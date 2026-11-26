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

/// A title image barriers patch.
struct Image {
    base: usize,
    size: usize,
    name: String,
    /// Its PE header's time stamp: which build of it.
    stamp: u32,
    /// Scanned (or being scanned) already.
    prepared: bool,
}

/// The title's images, in the order they came.
static IMAGES: std::sync::Mutex<Vec<Image>> = std::sync::Mutex::new(Vec::new());
/// Where the sites patched are remembered between runs.
static SITES_FILE: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();

/// The time stamp and size of the image's headers.
fn pe(base: usize) -> (u32, usize) {
    // SAFETY: the headers of a mapped image: `e_lfanew`, then the file
    // header's time stamp and the optional header's `SizeOfImage`.
    unsafe {
        let nt = base + ((base + 0x3c) as *const u32).read_unaligned() as usize;
        let stamp = ((nt + 8) as *const u32).read_unaligned();
        let size = ((nt + 0x50) as *const u32).read_unaligned() as usize;
        (stamp, size)
    }
}

fn register(base: usize, name: &str) {
    let (stamp, size) = pe(base);
    let mut images = IMAGES.lock().unwrap_or_else(|e| e.into_inner());
    if !images.iter().any(|i| i.base == base) {
        images.push(Image {
            base,
            size,
            name: name.to_ascii_lowercase(),
            stamp,
            prepared: false,
        });
    }
}

/// Starts barriers for the title's images (base, file name): the scan of
/// their code runs on a thread of its own, and patching begins when it is
/// done, first with the sites a run before patched (`sites`: the file they
/// are remembered in; patched under one stop of the other threads).
pub fn start(images: Vec<(usize, String)>, sites: Option<std::path::PathBuf>) {
    if !on() {
        return;
    }
    let _ = SITES_FILE.set(sites);
    for (base, name) in &images {
        register(*base, name);
    }
    let spawned = std::thread::Builder::new()
        .name("barrier scan".into())
        .spawn(move || {
            let Some(t) = tables::tables() else {
                tracing::warn!("VICTORY_BARRIERS: no memory for the tables; barriers stay off");
                return;
            };
            let relocate = std::env::var_os("VICTORY_BARRIER_RELOCATE").is_some_and(|v| v == "1");
            let int3 = std::env::var_os("VICTORY_BARRIER_INT3").is_some_and(|v| v == "1");
            // SAFETY: the tables are statics of `victory_barrier`. (Images are
            // added as each is scanned.)
            let patcher = match unsafe {
                Patcher::new(
                    t,
                    Config {
                        relocate,
                        int3,
                        images: Vec::new(),
                    },
                )
            } {
                Ok(p) => &*Box::leak(Box::new(p)),
                Err(e) => {
                    tracing::warn!(error = %e, "VICTORY_BARRIERS: barriers stay off");
                    return;
                }
            };
            if let Err(e) = victory_barrier::handler::install(patcher) {
                tracing::warn!(error = %e, "VICTORY_BARRIERS: barriers stay off");
                return;
            }
            let _ = PATCHER.set(patcher);
            prepare_pending(patcher);
            say(patcher);
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "VICTORY_BARRIERS: no thread for the scan; barriers stay off");
    }
}

/// The title mapped an image of its own after the start (`LoadLibrary`):
/// it is scanned and patched too.
pub fn image_loaded(base: usize, name: &str) {
    if !on() {
        return;
    }
    register(base, name);
    if let Some(&patcher) = PATCHER.get() {
        let _ = std::thread::Builder::new()
            .name("barrier scan".into())
            .spawn(move || prepare_pending(patcher));
    }
    // (Else the start's scan takes it.)
}

/// Scans the images not scanned yet, each then patched from the sites a run
/// before remembered for it.
fn prepare_pending(patcher: &'static Patcher) {
    loop {
        let next = {
            let mut images = IMAGES.lock().unwrap_or_else(|e| e.into_inner());
            images.iter_mut().find(|i| !i.prepared).map(|i| {
                i.prepared = true;
                (i.base, i.name.clone(), i.stamp, i.size)
            })
        };
        let Some((base, name, stamp, size)) = next else {
            return;
        };
        let began = std::time::Instant::now();
        // SAFETY: a title image, mapped for the rest of the run.
        let targets = unsafe { patcher.prepare(base as u64) };
        let scanned = began.elapsed();
        let wanted = remembered(base, &name, stamp, size);
        let patched = patcher.patch_many(&wanted);
        tracing::info!(
            image = %name,
            targets = ?targets,
            scan_ms = scanned.as_millis() as u64,
            remembered = wanted.len(),
            patched_ahead = patched,
            ahead_ms = (began.elapsed() - scanned).as_millis() as u64,
            "VICTORY_BARRIERS: scanned; its stores that fault on guarded view memory are patched from now on"
        );
    }
}

/// A line of the sites file: image name, its time stamp, the site's offset
/// in it and a hash of its instruction, in hexadecimal. Offsets and hashes,
/// no code of the title's.
fn parse(line: &str) -> Option<(&str, u32, usize, u64)> {
    let mut w = line.split_whitespace();
    let name = w.next()?;
    let stamp = u32::from_str_radix(w.next()?, 16).ok()?;
    let rva = usize::from_str_radix(w.next()?, 16).ok()?;
    let hash = u64::from_str_radix(w.next()?, 16).ok()?;
    Some((name, stamp, rva, hash))
}

fn sites_text() -> String {
    SITES_FILE
        .get()
        .cloned()
        .flatten()
        .and_then(|f| std::fs::read_to_string(f).ok())
        .unwrap_or_default()
}

/// The sites remembered for this build of the image whose instruction is
/// still the one patched before: their addresses.
fn remembered(base: usize, name: &str, stamp: u32, size: usize) -> Vec<u64> {
    sites_text()
        .lines()
        .filter_map(parse)
        .filter(|&(n, s, rva, _)| n == name && s == stamp && rva < size)
        .filter_map(|(_, _, rva, hash)| {
            let rip = (base + rva) as u64;
            let len = victory_barrier::instruction_len(rip, (base + size) as u64)?;
            (victory_barrier::instruction_hash(rip, len) == hash).then_some(rip)
        })
        .collect()
}

/// Writes the sites patched into the sites file, with what it held of
/// images not seen in this run (other builds, DLLs not loaded yet).
fn save(patcher: &Patcher) {
    let Some(file) = SITES_FILE.get().cloned().flatten() else {
        return;
    };
    let images: Vec<(usize, usize, String, u32)> = IMAGES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|i| (i.base, i.size, i.name.clone(), i.stamp))
        .collect();
    let ours = |name: &str, stamp: u32| images.iter().any(|i| i.2 == name && i.3 == stamp);
    let mut lines: std::collections::BTreeSet<String> = sites_text()
        .lines()
        .filter(|l| parse(l).is_some_and(|(n, s, _, _)| !ours(n, s)))
        .map(String::from)
        .collect();
    for (rip, hash) in patcher.patched_sites() {
        let rip = rip as usize;
        if let Some((base, _, name, stamp)) = images.iter().find(|i| rip >= i.0 && rip < i.0 + i.1)
        {
            lines.insert(format!("{name} {stamp:08x} {:x} {hash:016x}", rip - base));
        }
    }
    let text: String = lines.into_iter().map(|l| l + "\n").collect();
    let partial = file.with_extension("txt.partial");
    if std::fs::write(&partial, text)
        .and_then(|()| std::fs::rename(&partial, &file))
        .is_err()
    {
        tracing::warn!(file = %file.display(), "VICTORY_BARRIERS: the sites patched could not be remembered");
    }
}

/// What the barriers did, every ten seconds while it changes; and the sites
/// patched, remembered for the next run.
fn say(patcher: &'static Patcher) {
    let _ = std::thread::Builder::new()
        .name("barrier line".into())
        .spawn(move || {
            let (mut last, mut saved) = (u64::MAX, 0u64);
            loop {
                std::thread::sleep(std::time::Duration::from_secs(10));
                let s = patcher.stats();
                if s.patched != saved {
                    save(patcher);
                    saved = s.patched;
                }
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
        // The sites file, in a directory of the test's; the store's hash
        // before it is patched.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sites.txt");
        let store_at = victory_barrier_test_store as *const () as u64;
        let hash = victory_barrier::instruction_hash(store_at, 6);
        start(vec![(exe, "Test.exe".into())], Some(file.clone()));
        let began = std::time::Instant::now();
        while !PATCHER.get().is_some_and(|p| p.patches(exe as u64)) {
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
        // Remembered for the next run: the image (by name and build), the
        // store's offset in it, the hash of the instruction it was.
        save(PATCHER.get().unwrap());
        let text = std::fs::read_to_string(&file).unwrap();
        let (stamp, _) = pe(exe);
        let line = format!(
            "test.exe {stamp:08x} {:x} {hash:016x}",
            store_at as usize - exe
        );
        assert!(text.lines().any(|l| l == line), "{line:?} not in {text:?}");
        assert_eq!(
            parse(&line),
            Some(("test.exe", stamp, store_at as usize - exe, hash))
        );
        // What the file says of an image with another build is kept.
        std::fs::write(
            &file,
            format!(
                "{text}other.dll 00000001 10 0000000000000002
"
            ),
        )
        .unwrap();
        save(PATCHER.get().unwrap());
        assert!(
            std::fs::read_to_string(&file)
                .unwrap()
                .contains("other.dll 00000001 10")
        );
    }
}
