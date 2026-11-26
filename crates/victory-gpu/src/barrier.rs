//! Write barriers in the title's code (`VICTORY_BARRIERS=1`; off by default):
//! `victory_barrier` patches an instruction whose store faulted on a guarded
//! block so that it stores beside the guard, through our own view of the
//! page ([`crate::inplace::beside`]), and marks the block, as
//! [`crate::titlewrite`]'s store does at every fault, but once. The block
//! stays guarded and clean; the next asking takes the marks
//! ([`crate::watch`]'s `barrier_marks`) as it takes [`crate::watch::put_beside`]'s
//! bits.
//!
//! - A guarded block gets its delta (the distance to our view of its page)
//!   at its first write fault, and loses it when the title's mapping there
//!   changes ([`forget`]). A block without one stores where the title meant,
//!   and faults as before.
//! - Only the title's images are patched, and only once their scan is done
//!   (on a thread of its own, from [`start`]): until then, and for any store
//!   the patcher refuses, the fault goes on the way it always did.
//! - Marks go by our view's address, so a write through any view of a page
//!   is told at all of them.
//!
//! What it does not do (yet): memory the kernel's write watch tracks (no
//! fault names its stores); the title's own `VirtualProtect` of view memory
//! (a store a block made read-only by the title should fault on goes beside
//! it); title DLLs loaded after the start.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use victory_barrier::{Config, Outcome, Patcher, tables};

/// Guarded blocks: the console's 64 KiB page.
const BLOCK: usize = 1 << 16;

static PATCHER: OnceLock<&'static Patcher> = OnceLock::new();
static FAULTS: AtomicU64 = AtomicU64::new(0);
static DELTAS: AtomicU64 = AtomicU64::new(0);
/// Pieces of the title's `memcpy`, `memmove` and `memset` put beside their
/// guards by a block's delta ([`beside`]).
static COPIED: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// Barriers off for this thread's copies and faults (a test that
    /// compares the ways of copying).
    pub(crate) static OFF_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether barriers are asked for (and possible on this processor).
pub fn on() -> bool {
    #[cfg(test)]
    if OFF_IN_TEST.get() {
        return false;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_BARRIERS").is_some_and(|v| v == "1");
        let cpu = std::arch::is_x86_feature_detected!("bmi2") && std::arch::is_x86_feature_detected!("xsave");
        if asked && !cpu {
            tracing::warn!("VICTORY_BARRIERS: this processor lacks BMI2 or XSAVE; barriers stay off");
        }
        // Their marks are told as `put_beside` tells its bytes, which needs
        // the light refresh and the guard over every view of a page.
        let beside = crate::watch::beside_possible();
        if asked && !beside {
            tracing::warn!("VICTORY_BARRIERS: needs VICTORY_LIGHT_REFRESH and VICTORY_VIEW_GUARD on; barriers stay off");
        }
        asked && cpu && beside
    })
}

/// Starts barriers for the title's images (their bases): the scan of their
/// code runs on a thread of its own, and patching begins when it is done.
pub fn start(images: Vec<usize>) {
    if !on() {
        return;
    }
    let spawned = std::thread::Builder::new().name("barrier scan".into()).spawn(move || {
        let began = std::time::Instant::now();
        let Some(t) = tables::tables() else {
            tracing::warn!("VICTORY_BARRIERS: no memory for the tables; barriers stay off");
            return;
        };
        let relocate = std::env::var_os("VICTORY_BARRIER_RELOCATE").is_some_and(|v| v == "1");
        let int3 = std::env::var_os("VICTORY_BARRIER_INT3").is_some_and(|v| v == "1");
        let images: Vec<u64> = images.iter().map(|&b| b as u64).collect();
        // SAFETY: the tables are statics of `victory_barrier`.
        let patcher = match unsafe { Patcher::new(t, Config { relocate, int3, images: images.clone() }) } {
            Ok(p) => &*Box::leak(Box::new(p)),
            Err(e) => {
                tracing::warn!(error = %e, "VICTORY_BARRIERS: barriers stay off");
                return;
            }
        };
        let targets: Vec<Option<usize>> = images.iter().map(|&base| patcher.prepare(base)).collect();
        if let Err(e) = victory_barrier::handler::install(patcher) {
            tracing::warn!(error = %e, "VICTORY_BARRIERS: barriers stay off");
            return;
        }
        let _ = PATCHER.set(patcher);
        tracing::info!(images = images.len(), ?targets, ms = began.elapsed().as_millis() as u64, "VICTORY_BARRIERS: the title's code scanned; stores that fault on guarded view memory are patched from now on");
        say(patcher);
    });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "VICTORY_BARRIERS: no thread for the scan; barriers stay off");
    }
}

/// What the barriers did, every ten seconds while it changes.
fn say(patcher: &'static Patcher) {
    let _ = std::thread::Builder::new()
        .name("barrier line".into())
        .spawn(move || {
            let mut last = u64::MAX;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(10));
                let s = patcher.stats();
                let now = s.patched + s.breakpoints / 1024 + FAULTS.load(Ordering::Relaxed);
                if now == last {
                    continue;
                }
                last = now;
                tracing::info!(
                    patched = s.patched,
                    strategies = ?s.strategies,
                    refused = ?s.refused_sites,
                    int3_hits = s.breakpoints,
                    threads_moved = s.redirected_threads,
                    patching_ms = s.patch_ns / 1_000_000,
                    faults_seen = FAULTS.load(Ordering::Relaxed),
                copy_pieces_beside = COPIED.load(Ordering::Relaxed),
                    blocks_beside = DELTAS.load(Ordering::Relaxed),
                    "barriers"
                );
            }
        });
}

/// A store of the title's faulted at `at`, in a block we guard: the block's
/// stores go beside the guard from now on, and the instruction is patched if
/// it can be (then `true`: the thread goes on, maybe elsewhere, and does the
/// store through its trampoline). `false`: the fault goes on as before.
pub(crate) fn on_fault(
    context: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT,
    at: usize,
) -> bool {
    if !on() {
        return false;
    }
    FAULTS.fetch_add(1, Ordering::Relaxed);
    if give_delta(at).is_none() {
        return false;
    }
    let Some(patcher) = PATCHER.get() else {
        return false;
    };
    let mut rip = context.Rip;
    match patcher.on_write_fault(&mut rip) {
        Outcome::Patched | Outcome::Redirected => {
            context.Rip = rip;
            true
        }
        Outcome::Refused(_) => false,
    }
}

/// The distance from the guarded block at `at` to our view of its page,
/// given to the block now if it has none: what its stores add from now on.
/// `None` if the page has no view of ours (or the mapping is changing).
fn give_delta(at: usize) -> Option<usize> {
    let block = at & !(BLOCK - 1);
    let delta = tables::delta(block as u64) as usize;
    if delta != 0 {
        return Some(delta);
    }
    // No mapping changes while our view of the page is looked up and its
    // delta set; one that comes after clears it ([`forget`]).
    let _reading = crate::watch::title_memory_reading_now()?;
    let (page, _) = crate::inplace::beside_page(block)?;
    let delta = page.wrapping_sub(block);
    if !tables::track(page as u64, BLOCK as u64) || !tables::set_delta(block as u64, delta as u64) {
        return None;
    }
    DELTAS.fetch_add(1, Ordering::Relaxed);
    Some(delta)
}

/// For a piece of a copy of the title's ([`crate::titlewrite`]) into one
/// guarded block at `at`: what to add to its addresses to write beside the
/// guard, after which the caller writes there and then calls [`wrote`].
/// `None`: barriers are off, or the block cannot be given a delta.
///
/// Cheaper than [`crate::watch::put_beside`], which looks the page up under
/// the mapping's and the views' locks at every piece and sets a bit at
/// every view of it: once a block has its delta this is one table read, and
/// [`wrote`] three byte stores. And no share per block: nothing here costs
/// more for a block written often.
pub(crate) fn beside(at: usize) -> Option<usize> {
    if !on() {
        return None;
    }
    give_delta(at)
}

/// A piece of a copy was written at `[ours, ours + len)` (our view, inside
/// one block): its parts marked as a barrier marks.
pub(crate) fn wrote(ours: usize, len: usize) {
    if let Some(t) = tables::tables() {
        t.mark_range(ours as u64, len as u64);
        COPIED.fetch_add(1, Ordering::Relaxed);
    }
}

/// The title's mapping of `[at, at + len)` changes: its stores go where the
/// title means again.
pub(crate) fn forget(at: usize, len: usize) {
    if on() && len != 0 {
        tables::clear(at as u64, at.saturating_add(len) as u64);
    }
}

/// Takes the barriers' marks: `wrote` is given the title's addresses of the
/// blocks written (every view of each page written), each with the parts
/// written (a bit per 4 KiB), before any other taking can answer that nothing
/// was.
pub(crate) fn take(wrote: impl FnOnce(&[(usize, u16)])) {
    if !on() {
        return;
    }
    tables::take(|ours| {
        let views = crate::inplace::views_of_beside(ours);
        wrote(&views);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::global_asm;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_BASIC_INFORMATION, MapViewOfFile, PAGE_READONLY,
        PAGE_READWRITE, VirtualQuery,
    };

    global_asm!(
        r#"
        .text
        .fill 160, 1, 0x90
        .p2align 4
        .globl victory_barrier_test_store
    // (dst, value): [dst + 0x100] = value, six bytes: a jump fits.
    victory_barrier_test_store:
        mov dword ptr [rcx + 0x100], edx
        ret
        .fill 160, 1, 0x90
    "#
    );

    unsafe extern "C" {
        fn victory_barrier_test_store(dst: usize, value: u32);
    }

    fn protection(at: usize) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) };
        m.Protect
    }

    /// The title's store into guarded view memory faults once, is patched,
    /// and from then on goes beside the guard: the block stays read-only, the
    /// bytes are there, and an asking tells the block as written each time.
    /// Patches this test binary's code and turns barriers on for the whole
    /// process, so it runs alone:
    /// `VICTORY_BARRIERS=1 cargo test -p victory-gpu --lib barrier -- --ignored`.
    #[test]
    #[ignore = "VICTORY_BARRIERS=1, alone"]
    fn a_store_into_view_memory_goes_beside_its_guard() {
        assert!(on(), "run with VICTORY_BARRIERS=1");
        const LEN: usize = 1 << 20;
        // View memory as the runtime makes the title's.
        // SAFETY: a pagefile section of our own and a view of it.
        let (section, at) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as usize;
            assert_ne!(at, 0);
            crate::inplace::title_memory_section(section as usize, LEN as u64, true);
            crate::title_memory_remapped(at, LEN);
            crate::inplace::title_memory_view(at, LEN, section as usize, 0);
            (section, at)
        };
        // SAFETY: this test's image.
        let exe = unsafe {
            windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null())
        } as usize;
        start(vec![exe]);
        let began = std::time::Instant::now();
        while PATCHER.get().is_none() {
            assert!(began.elapsed().as_secs() < 120, "the scan never ended");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let store = |block: usize, value: u32| {
            // SAFETY: inside a view.
            unsafe { victory_barrier_test_store(block, value) };
        };
        let read = |at: usize| {
            // SAFETY: inside a view.
            unsafe { std::ptr::read_volatile((at + 0x100) as *const u32) }
        };
        let told = |at: usize, block: usize, asked: &mut u64, what: &str| {
            let (now, written) = crate::watch::writes(at, at + LEN, true).expect("a view");
            assert!(
                written
                    .iter()
                    .any(|&(page, w)| page == block / BLOCK && w > *asked),
                "{what}: told as written: {written:?} after {asked}"
            );
            *asked = now;
        };
        let block = at + 3 * BLOCK;
        let mut asked = crate::watch::writes(at, at + LEN, true).expect("a view").0;
        // Guarded, as after a refresh.
        let _ = crate::watch::writes(at, at + LEN, true);
        assert_eq!(protection(block), PAGE_READONLY, "guarded");
        for round in 0..4u32 {
            store(block, 0x5150 + round);
            assert_eq!(read(block), 0x5150 + round);
            assert_eq!(
                protection(block),
                PAGE_READONLY,
                "round {round}: still guarded"
            );
            told(at, block, &mut asked, &format!("round {round}"));
        }
        // Told by the 4 KiB part: a store in one part of the block, the journal
        // says that part alone was written (a refresh copies it, not the block).
        for part in [0usize, 5, 15, 9] {
            let before = asked;
            // (The store writes 0x100 bytes on from where it is given.)
            store(block + part * 4096, 0x5200 + part as u32);
            told(at, block, &mut asked, &format!("part {part}"));
            let parts = crate::watch::parts_written(&[(block / BLOCK, before)]);
            assert_eq!(parts, vec![1u16 << part], "part {part}: {parts:?}");
        }
        assert_eq!(PATCHER.get().unwrap().stats().patched, 1);
        assert_eq!(FAULTS.load(Ordering::Relaxed), 1, "one fault, the first");

        // The title shows the same pages at a second address.
        // SAFETY: a second view of the section.
        let at2 = unsafe {
            let at2 = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as usize;
            assert_ne!(at2, 0);
            crate::title_memory_remapped(at2, LEN);
            crate::inplace::title_memory_view(at2, LEN, section as usize, 0);
            at2
        };
        let block2 = at2 + 3 * BLOCK;
        let mut asked2 = crate::watch::writes(at2, at2 + LEN, true)
            .expect("a view")
            .0;
        let _ = crate::watch::writes(at2, at2 + LEN, true);
        let _ = crate::watch::writes(at, at + LEN, true);
        asked = crate::watch::writes(at, at + LEN, true).expect("a view").0;
        // Through the first address: told at the second too.
        store(block, 0x6000);
        assert_eq!(read(block2), 0x6000);
        told(
            at2,
            block2,
            &mut asked2,
            "a store through the first view, asked at the second",
        );
        // Through the second: its block gets its delta at its first fault.
        for round in 0..4u32 {
            store(block2, 0x7000 + round);
            assert_eq!(read(block), 0x7000 + round);
            told(
                at,
                block,
                &mut asked,
                &format!("round {round} through the second view, asked at the first"),
            );
            told(
                at2,
                block2,
                &mut asked2,
                &format!("round {round} through the second view, asked there"),
            );
        }
        assert_eq!(
            PATCHER.get().unwrap().stats().patched,
            1,
            "the same instruction"
        );
        assert_eq!(
            FAULTS.load(Ordering::Relaxed),
            2,
            "one more fault: the second view's block's first"
        );
    }
}
