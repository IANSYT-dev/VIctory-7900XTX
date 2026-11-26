//! Sites patched ahead of their first fault (`Patcher::patch_many`), as the
//! sites a run before remembered are: all under one stop of the other
//! threads, told back by `patched_sites` with the hash of what they were,
//! and from then on their stores go beside the guard without a fault.

#![cfg(windows)]

use std::arch::global_asm;
use std::sync::atomic::{AtomicU64, Ordering};
use victory_barrier::{Config, Patcher, instruction_hash, tables};
use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
    VirtualProtect,
};

global_asm!(
    r#"
    .text
    .fill 160, 1, 0x90
    .p2align 4
    .globl va_wide
va_wide:
    mov dword ptr [rcx + 0x100], edx
    ret
    .fill 160, 1, 0x90
    .globl va_short
va_short:
    mov dword ptr [rcx], edx
    ret
    .fill 160, 1, 0x90
"#
);

unsafe extern "C" {
    fn va_wide(dst: u64, value: u32);
    fn va_short(dst: u64, value: u32);
}

const LEN: usize = 1 << 20;
const BLOCK: u64 = 1 << 16;

static FAULTS: AtomicU64 = AtomicU64::new(0);

unsafe extern "system" fn count(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes a valid record.
    let r = unsafe { &*(*info).ExceptionRecord };
    if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION {
        FAULTS.fetch_add(1, Ordering::SeqCst);
    }
    EXCEPTION_CONTINUE_SEARCH
}

#[test]
fn sites_patched_ahead_never_fault() {
    // SAFETY: a pagefile-backed section and two views of it.
    let (title, ours) = unsafe {
        let s = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE,
            0,
            LEN as u32,
            std::ptr::null(),
        );
        assert!(!s.is_null());
        let a = MapViewOfFile(s, FILE_MAP_WRITE, 0, 0, LEN).Value as u64;
        let b = MapViewOfFile(s, FILE_MAP_WRITE, 0, 0, LEN).Value as u64;
        (a, b)
    };
    assert!(tables::track(ours, LEN as u64));
    for b in (0..LEN as u64).step_by(BLOCK as usize) {
        assert!(tables::set_delta(title + b, ours.wrapping_sub(title)));
    }
    // SAFETY: this test's image.
    let exe = unsafe { GetModuleHandleW(std::ptr::null()) } as u64;
    let config = Config {
        relocate: false,
        int3: true,
        images: vec![exe],
    };
    // SAFETY: the tables live for the process.
    let patcher: &'static Patcher = Box::leak(Box::new(
        unsafe { Patcher::new(tables::tables().unwrap(), config) }.unwrap(),
    ));
    victory_barrier::handler::install(patcher).unwrap();
    assert!(unsafe { patcher.prepare(exe) }.is_some());
    let (wide, short) = (va_wide as *const () as u64, va_short as *const () as u64);
    let mut want = vec![
        (wide, instruction_hash(wide, 6)),
        (short, instruction_hash(short, 2)),
    ];
    want.sort_unstable();
    assert_eq!(patcher.patch_many(&[wide, short]), 2);
    assert_eq!(patcher.patch_many(&[wide, short]), 0, "patched already");
    assert_eq!(patcher.patched_sites(), want);
    // SAFETY: a handler that lives as long as the process.
    assert!(!unsafe { AddVectoredExceptionHandler(1, Some(count)) }.is_null());
    let mut old = 0;
    // SAFETY: the title's view, guarded.
    assert!(unsafe { VirtualProtect(title as *const _, LEN, PAGE_READONLY, &mut old) } != 0);
    for round in 0..3u32 {
        // SAFETY: inside the guarded view; the stores go beside it.
        unsafe {
            va_wide(title + BLOCK, round + 10);
            va_short(title + 2 * BLOCK, round + 20);
        }
        // SAFETY: the views.
        unsafe {
            assert_eq!(
                std::ptr::read_volatile((title + BLOCK + 0x100) as *const u32),
                round + 10
            );
            assert_eq!(
                std::ptr::read_volatile((ours + 2 * BLOCK) as *const u32),
                round + 20
            );
        }
        let mut marked = Vec::new();
        tables::take(|b| marked.extend_from_slice(b));
        marked.sort_unstable();
        assert_eq!(
            marked,
            vec![(ours + BLOCK, 1), (ours + 2 * BLOCK, 1)],
            "round {round}"
        );
    }
    assert_eq!(FAULTS.load(Ordering::SeqCst), 0, "no store faulted");
    assert_eq!(patcher.stats().patched, 2);
}
