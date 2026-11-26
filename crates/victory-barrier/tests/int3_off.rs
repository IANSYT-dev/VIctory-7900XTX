//! With `int3` sites not asked for (the default), a store that only an
//! `int3` could send to its trampoline is refused, once, and its faults go
//! the guards' way from then on; a store a jump fits over is still patched.

#![cfg(windows)]

use std::arch::global_asm;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use victory_barrier::classify::Refusal;
use victory_barrier::{Config, Outcome, Patcher, tables};
use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_POINTERS,
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
    .globl vo_wide
vo_wide:
    mov dword ptr [rcx + 0x100], edx
    ret
    .fill 160, 1, 0x90
    .globl vo_short
vo_short:
    mov dword ptr [rcx], edx
    ret
    .fill 160, 1, 0x90
"#
);

unsafe extern "C" {
    fn vo_wide(dst: u64, value: u32);
    fn vo_short(dst: u64, value: u32);
}

const LEN: usize = 1 << 20;
const BLOCK: u64 = 1 << 16;

static STATE: OnceLock<(u64, u64, &'static Patcher)> = OnceLock::new();
static OPENED: AtomicU64 = AtomicU64::new(0);

unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes a valid record and context.
    let (r, c) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    let Some(&(title, ours, patcher)) = STATE.get() else {
        return 0;
    };
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION
        || r.NumberParameters < 2
        || r.ExceptionInformation[0] != 1
    {
        return 0;
    }
    let at = r.ExceptionInformation[1] as u64;
    if at < title || at >= title + LEN as u64 {
        return 0;
    }
    let block = at & !(BLOCK - 1);
    assert!(tables::set_delta(block, ours.wrapping_sub(title)));
    let mut rip = c.Rip;
    match patcher.on_write_fault(&mut rip) {
        Outcome::Patched | Outcome::Redirected => c.Rip = rip,
        Outcome::Refused(_) => {
            // The guards' way: open the block.
            OPENED.fetch_add(1, Ordering::SeqCst);
            let mut old = 0;
            // SAFETY: a block of the test's view.
            unsafe { VirtualProtect(block as *const _, BLOCK as usize, PAGE_READWRITE, &mut old) };
        }
    }
    EXCEPTION_CONTINUE_EXECUTION
}

#[test]
fn short_stores_are_left_to_the_guards() {
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
    // SAFETY: this test's image.
    let exe = unsafe { GetModuleHandleW(std::ptr::null()) } as u64;
    let config = Config {
        relocate: false,
        int3: false,
        images: vec![exe],
    };
    // SAFETY: the tables live for the process.
    let patcher: &'static Patcher = Box::leak(Box::new(
        unsafe { Patcher::new(tables::tables().unwrap(), config) }.unwrap(),
    ));
    assert!(unsafe { patcher.prepare(exe) }.is_some());
    assert!(STATE.set((title, ours, patcher)).is_ok());
    // SAFETY: a handler that lives as long as the process.
    assert!(!unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) }.is_null());
    for round in 0..3u32 {
        let mut old = 0;
        // SAFETY: the title's view, guarded again as after a refresh.
        assert!(unsafe { VirtualProtect(title as *const _, LEN, PAGE_READONLY, &mut old) } != 0);
        // SAFETY: inside the view.
        unsafe {
            vo_wide(title + BLOCK, round + 1);
            vo_short(title + 2 * BLOCK, round + 2);
        }
        // SAFETY: as above.
        unsafe {
            assert_eq!(
                std::ptr::read_volatile((title + BLOCK + 0x100) as *const u32),
                round + 1
            );
            assert_eq!(
                std::ptr::read_volatile((title + 2 * BLOCK) as *const u32),
                round + 2
            );
        }
        // The short store's block was opened, once a round; the wide one's
        // never was.
        assert_eq!(OPENED.load(Ordering::SeqCst), u64::from(round) + 1);
    }
    let stats = patcher.stats();
    assert_eq!(stats.patched, 1, "{stats:?}");
    assert_eq!(
        stats.refused_sites.get(&Refusal::Int3Off),
        Some(&1),
        "{stats:?}"
    );
    assert_eq!(stats.breakpoints, 0);
}
