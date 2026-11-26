//! Barriers end to end, the way `victory-gpu` uses them: a section mapped
//! twice (the title's view, guarded, and ours beside it), stores in this
//! test's own code (standing for the title's) that fault on the guard, a
//! handler that gives the block its delta and asks the patcher, and then the
//! same stores again, which must not fault, must land in the section, and
//! must be marked at our view's blocks.

#![cfg(windows)]

use std::arch::global_asm;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use victory_barrier::{Config, Outcome, Patcher, handler, tables};
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
    .globl vb_wide
// vb_wide(dst, value): [dst + 0x100] = value. Six bytes: a jump fits.
vb_wide:
    mov dword ptr [rcx + 0x100], edx
    ret
    .fill 160, 1, 0x90
    .globl vb_short
// vb_short(dst, value): [dst] = value. Two bytes, a `ret` after, no
// unwind data: an island if padding is near, else int3.
vb_short:
    mov dword ptr [rcx], edx
    ret
    .fill 160, 1, 0x90
    .globl vb_fill
// vb_fill(dst, byte, count): `rep stosb`.
vb_fill:
    push rdi
    mov rdi, rcx
    mov eax, edx
    mov rcx, r8
    rep stosb
    pop rdi
    ret
    .fill 160, 1, 0x90
"#
);

unsafe extern "C" {
    fn vb_wide(dst: u64, value: u32);
    fn vb_short(dst: u64, value: u32);
    fn vb_fill(dst: u64, byte: u32, count: u64);
}

const LEN: usize = 1 << 20;
const BLOCK: u64 = 1 << 16;

struct State {
    title: u64,
    ours: u64,
    patcher: &'static Patcher,
}

static STATE: OnceLock<State> = OnceLock::new();
static FAULTS: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);

unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes a valid record and context.
    let (r, c) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    let Some(s) = STATE.get() else { return 0 };
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION
        || r.NumberParameters < 2
        || r.ExceptionInformation[0] != 1
    {
        return 0;
    }
    let at = r.ExceptionInformation[1] as u64;
    if at < s.title || at >= s.title + LEN as u64 {
        return 0;
    }
    FAULTS.fetch_add(1, Ordering::SeqCst);
    // What victory-gpu's watch does: the block's stores go beside its guard.
    let block = at & !(BLOCK - 1);
    assert!(tables::set_delta(block, s.ours.wrapping_sub(s.title)));
    let mut rip = c.Rip;
    match s.patcher.on_write_fault(&mut rip) {
        Outcome::Patched | Outcome::Redirected => {
            c.Rip = rip;
        }
        Outcome::Refused(_) => {
            // The guards' way: open the block (and the test counts it).
            REFUSED.fetch_add(1, Ordering::SeqCst);
            let mut old = 0;
            // SAFETY: a block of the test's view.
            unsafe { VirtualProtect(block as *const _, BLOCK as usize, PAGE_READWRITE, &mut old) };
        }
    }
    EXCEPTION_CONTINUE_EXECUTION
}

fn read32(at: u64) -> u32 {
    // SAFETY: a mapped view of the test's section.
    unsafe { std::ptr::read_volatile(at as *const u32) }
}

#[test]
fn stores_go_beside_the_guard_and_are_marked() {
    // SAFETY: a pagefile-backed section and two views of it.
    let (title, ours) = unsafe {
        let section = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE,
            0,
            LEN as u32,
            std::ptr::null(),
        );
        assert!(!section.is_null());
        let a = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as u64;
        let b = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as u64;
        assert!(a != 0 && b != 0);
        (a, b)
    };
    assert!(tables::track(ours, LEN as u64));
    let t = tables::tables().expect("tables");
    // SAFETY: this test's image.
    let exe = unsafe { GetModuleHandleW(std::ptr::null()) } as u64;
    // SAFETY: the tables live for the process.
    let patcher: &'static Patcher = Box::leak(Box::new(
        unsafe {
            Patcher::new(
                t,
                Config {
                    relocate: false,
                    int3: true,
                    images: vec![exe],
                },
            )
        }
        .expect("patcher"),
    ));
    assert!(
        unsafe { patcher.prepare(exe) }.is_some(),
        "the scan of this test's image"
    );
    handler::install(patcher).expect("breakpoint handler");
    assert!(
        STATE
            .set(State {
                title,
                ours,
                patcher
            })
            .is_ok()
    );
    // SAFETY: a handler that lives as long as the process.
    assert!(!unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) }.is_null());
    let mut old = 0;
    // SAFETY: the title's view, guarded as the watch guards it.
    assert!(unsafe { VirtualProtect(title as *const _, LEN, PAGE_READONLY, &mut old) } != 0);

    let mut faults_after_first = 0;
    for round in 0..4u32 {
        let v = 0x1000 * (round + 1);
        // SAFETY: addresses inside the guarded view.
        unsafe {
            vb_wide(title + BLOCK, v + 1);
            vb_short(title + 2 * BLOCK + 8, v + 2);
            vb_fill(title + 3 * BLOCK + 16, 0x40 + round, 300);
        }
        for (at, want) in [
            (title + BLOCK + 0x100, v + 1),
            (title + 2 * BLOCK + 8, v + 2),
        ] {
            assert_eq!(read32(at), want, "round {round}: through the title's view");
            assert_eq!(
                read32(at - title + ours),
                want,
                "round {round}: through ours"
            );
        }
        let b = 0x40 + round;
        assert_eq!(
            read32(title + 3 * BLOCK + 16),
            u32::from_le_bytes([b as u8; 4])
        );
        assert_eq!(
            read32(title + 3 * BLOCK + 16 + 296),
            u32::from_le_bytes([b as u8; 4])
        );
        assert_eq!(
            read32(title + 3 * BLOCK + 16 + 300) & 0xFF,
            0,
            "past the fill"
        );
        let mut marked = Vec::new();
        tables::take(|w| marked.extend_from_slice(w));
        marked.sort_unstable();
        if round == 0 {
            faults_after_first = FAULTS.load(Ordering::SeqCst);
            // The first round's stores faulted, and were redone by their
            // trampolines: marked like the others.
            assert_eq!(faults_after_first, 3);
        } else {
            assert_eq!(
                FAULTS.load(Ordering::SeqCst),
                faults_after_first,
                "round {round}: no new faults"
            );
        }
        assert_eq!(
            marked,
            vec![
                (ours + BLOCK, 1),
                (ours + 2 * BLOCK, 1),
                (ours + 3 * BLOCK, 1)
            ],
            "round {round}: marks at our view's blocks, each in its first 4 KiB"
        );
    }
    assert_eq!(REFUSED.load(Ordering::SeqCst), 0);
    let stats = patcher.stats();
    eprintln!(
        "patched {} sites: {:?}, int3 hits {}",
        stats.patched, stats.strategies, stats.breakpoints
    );
    assert_eq!(stats.patched, 3);
}
