//! Write barriers patched into the title's code where it stores to guarded
//! view memory (`VICTORY_BARRIERS=1`; off by default; read in `victory-gpu`).
//!
//! `victory-gpu`'s watch guards view memory; the first store that faults on
//! a guard names its instruction. This crate makes that instruction go to a
//! trampoline that does the same store through our own view of the same page
//! (which no guard covers) and marks the block it wrote (see `layout`). From
//! then on that instruction costs a few dozen instructions instead of a
//! fault, and the block it wrote stays guarded, so a store from code that has
//! not been patched yet is still caught. It is `titlewrite`'s store beside
//! the guard, done once at the instruction instead of at every fault.
//!
//! Developed apart from VIctory as a prototype and benchmark with a
//! stand-in title, torture cases and a differential test against the CPU
//! (the `xbcompat` repository); this is its patcher, fitted to the fork.
//!
//! How the site reaches its trampoline ([`Strategy`]), in order of
//! preference:
//!
//! 1. `Direct`: the store is five bytes or more; a `jmp rel32` replaces it.
//! 2. `Island`: a `jmp rel8` (two bytes) to five bytes of `int3` padding
//!    between functions within 127 bytes, which hold the `jmp rel32`.
//! 3. `Relocate`: the `jmp rel32` covers the following instructions too,
//!    which move into the trampoline, where the scan proves nothing can
//!    branch into them: every indirect jump of the function is understood
//!    (`Text::relocation_proven`).
//! 4. `BlockStart`: where the bytes after the store are a branch target,
//!    the jump goes at the start of the store's basic block instead, and
//!    the block up to the store moves: nothing enters a block but at its
//!    start. Also proven only.
//! 5. `RelocateUnproven` (only with `Config::relocate`): the same where it
//!    is not proven, trusting the global list of branch targets, which a
//!    computed jump defeats.
//! 6. `Int3`: one `int3` byte; the breakpoint handler sends the thread to
//!    the trampoline. A kernel round trip per execution, but no protection
//!    change, and the guard stays closed.
//!
//! 1, 2 and 6 overwrite only the store's own bytes (and padding), so no
//! branch, return or saved context anywhere can land inside what they
//! changed; 3 and 4 rely on the proof, 5 on luck. No span may cover bytes
//! an earlier patch wrote: its trampoline jumps back into them.
//!
//! What is left to the guards ([`Refusal`]) is counted by reason, so a run
//! says how much of a title's storing code the barriers cover.

#![cfg(windows)]

pub mod classify;
pub mod emit;
pub mod handler;
pub mod layout;
pub mod string;
pub mod tables;
pub mod text;
pub mod threads;
pub mod unwind;

use classify::{Refusal, Store};
use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, OpKind};
use layout::{SLOT_SHIFT, Tables};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use string::StringSite;
use text::Text;
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    GetModuleHandleExW,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS, VirtualAlloc,
    VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Bytes kept for trampolines per area; an area serves code within reach.
const AREA: usize = 16 << 20;
/// The most a trampoline takes (a relocated span of string stores is the
/// longest).
const TRAMPOLINE_MAX: usize = 2048;
const JUMP: usize = 5;

#[derive(Clone, Debug, Default)]
pub struct Config {
    /// Allow relocating the instructions after a short store where the scan
    /// did not prove it safe (`VICTORY_BARRIER_RELOCATE=1`; a computed jump
    /// into a moved span crashes the title).
    pub relocate: bool,
    /// Allow `int3` sites (`VICTORY_BARRIER_INT3=1`). Off, a store that only
    /// an `int3` could send to its trampoline is refused and faults as
    /// before: an `int3` traps at every execution, a guard once a block per
    /// refresh, so a hot short store can cost more on `int3`.
    pub int3: bool,
    /// The images whose code is patched (the title's): their bases. Code
    /// of any other image, or no image, is left to the guards.
    pub images: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strategy {
    Direct,
    Island,
    /// Relocation the scan proved safe (`Text::relocation_proven`).
    Relocate,
    /// The jump at the start of the store's basic block, the block up to
    /// the store (and a `ret` right after it) moved: for a store followed
    /// by a branch target, in a function the scan proved closed.
    BlockStart,
    /// Relocation on the global branch targets alone (`Config::relocate`).
    RelocateUnproven,
    Int3,
}

/// What the fault handler should do after asking the patcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The site goes to its trampoline now: run the instruction again.
    Patched,
    /// The thread faulted inside a span that was patched meanwhile; its
    /// context now points at the relocated instruction.
    Redirected,
    /// Let the guards handle this store.
    Refused(Refusal),
}

struct Record {
    /// Where the bytes were written: the store, or its block's start.
    at: u64,
    /// The bytes written there, to see whether the code changed since.
    written: Vec<u8>,
    island: Option<(u64, [u8; 5])>,
    strategy: Strategy,
    moved: Vec<u64>,
    /// A hash of the store instruction's own bytes before it was patched.
    hash: u64,
}

/// A patch made ready ([`Patcher::stage`]) and not yet written.
struct Staged {
    rip: u64,
    site: u64,
    span_len: usize,
    site_bytes: Vec<u8>,
    island: Option<(u64, [u8; 5])>,
    strategy: Strategy,
    moved: Vec<(u64, u64)>,
    string: bool,
    hash: u64,
}

impl Staged {
    /// Whether a thread stopped at `ip` is inside what the site's bytes
    /// overwrite (not at its first instruction).
    fn in_span(&self, ip: u64) -> bool {
        ip > self.site && ip < self.site + self.span_len as u64
    }

    /// Where a thread stopped at `ip` in the span goes: its instruction's copy.
    fn moved_to(&self, ip: u64) -> Option<u64> {
        self.moved.iter().find(|m| m.0 == ip).map(|m| m.1)
    }
}

/// The length of the instruction at `ip`, decoded from no further than `end`;
/// `None` if none decodes there.
pub fn instruction_len(ip: u64, end: u64) -> Option<usize> {
    if ip >= end {
        return None;
    }
    // SAFETY: the caller's code, mapped up to `end`.
    let bytes = unsafe { std::slice::from_raw_parts(ip as *const u8, (end - ip).min(15) as usize) };
    let i = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE).decode();
    (!i.is_invalid()).then_some(i.len())
}

/// FNV-1a of the instruction at `ip` (its own bytes, `len` of them).
pub fn instruction_hash(ip: u64, len: usize) -> u64 {
    // SAFETY: an instruction's bytes, which the caller decoded there.
    let bytes = unsafe { std::slice::from_raw_parts(ip as *const u8, len) };
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

enum Site {
    Patched(Record),
    Refused(Refusal),
}

struct Area {
    start: u64,
    next: u64,
    end: u64,
}

struct Inner {
    sites: HashMap<u64, Site>,
    /// Relocated instruction addresses: old → new.
    moved: HashMap<u64, u64>,
    factory: InstructionInfoFactory,
    areas: Vec<Area>,
    texts: HashMap<u64, Arc<Text>>,
    islands: BTreeSet<u64>,
    /// Code bytes patches wrote: start → end.
    written: BTreeMap<u64, u64>,
    refusals: BTreeMap<Refusal, u64>,
    refused_faults: BTreeMap<Refusal, u64>,
    strategies: BTreeMap<Strategy, u64>,
    repatched: u64,
}

/// Counters a run reports.
#[derive(Clone, Debug, Default)]
pub struct PatchStats {
    pub patched: u64,
    pub strategies: BTreeMap<Strategy, u64>,
    pub refused_sites: BTreeMap<Refusal, u64>,
    pub refused_faults: BTreeMap<Refusal, u64>,
    pub string_sites: u64,
    pub redirected_threads: u64,
    pub repatched: u64,
    pub breakpoints: u64,
    pub patch_ns: u64,
    pub modules: Vec<(u64, usize, text::TargetStats)>,
}

pub struct Patcher {
    tables: Tables,
    config: Config,
    xsave_bytes: u32,
    inner: Mutex<Inner>,
    /// int3 sites → their trampolines, read by the breakpoint handler.
    breakpoints: RwLock<HashMap<u64, u64>>,
    /// The images patched (`Config::images`, then each [`Self::prepare`]d).
    images: RwLock<Vec<u64>>,
    /// Trampoline areas, read without the lock: (start, end).
    area_bounds: RwLock<Vec<(u64, u64)>>,
    string_sites: AtomicU64,
    redirected: AtomicU64,
    breakpoint_hits: AtomicU64,
    patch_ns: AtomicU64,
}

impl Patcher {
    /// # Safety
    ///
    /// `tables` stay valid while barriers run.
    pub unsafe fn new(tables: Tables, config: Config) -> Result<Self, String> {
        let images = config.images.clone();
        let cpu = std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("xsave");
        if !cpu {
            return Err("write barriers need BMI2 and XSAVE".into());
        }
        let _ = string::TABLES.set(tables);
        Ok(Self {
            tables,
            config,
            xsave_bytes: string::xsave_area_bytes(),
            inner: Mutex::new(Inner {
                sites: HashMap::new(),
                moved: HashMap::new(),
                factory: InstructionInfoFactory::new(),
                areas: Vec::new(),
                texts: HashMap::new(),
                islands: BTreeSet::new(),
                written: BTreeMap::new(),
                refusals: BTreeMap::new(),
                refused_faults: BTreeMap::new(),
                strategies: BTreeMap::new(),
                repatched: 0,
            }),
            breakpoints: RwLock::new(HashMap::new()),
            images: RwLock::new(images),
            area_bounds: RwLock::new(Vec::new()),
            string_sites: AtomicU64::new(0),
            redirected: AtomicU64::new(0),
            breakpoint_hits: AtomicU64::new(0),
            patch_ns: AtomicU64::new(0),
        })
    }

    /// Scans an image ahead of its first fault, and patches it from then on
    /// (the title, at load; a DLL it loads later). The scan, which can take
    /// seconds, holds no lock: faults meanwhile are refused (`NotPatchable`)
    /// and go on as before. The number of branch targets found.
    ///
    /// # Safety
    /// `module` is a loaded image (headers and sections mapped) that stays.
    pub unsafe fn prepare(&self, module: u64) -> Option<usize> {
        // SAFETY: by the caller.
        let text = Arc::new(unsafe { Text::read(module) }?);
        let targets = text.target_count();
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .texts
            .insert(module, text);
        let mut images = self.images.write().unwrap_or_else(|e| e.into_inner());
        if !images.contains(&module) {
            images.push(module);
        }
        Some(targets)
    }

    /// Whether the image at `base` is scanned and its stores are patched.
    pub fn patches(&self, base: u64) -> bool {
        self.images
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&base)
    }

    pub fn in_trampolines(&self, ip: u64) -> bool {
        self.area_bounds
            .read()
            .unwrap()
            .iter()
            .any(|&(s, e)| ip >= s && ip < e)
    }

    /// The code region `rip` is in, scanned once.
    fn text_of(&self, inner: &mut Inner, rip: u64) -> Option<Arc<Text>> {
        let mut module = std::ptr::null_mut();
        // SAFETY: looks up the image containing `rip`, without a reference.
        let found = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                rip as *const u16,
                &mut module,
            )
        } != 0;
        if found {
            let base = module as u64;
            // (The host's own image is never among them in VIctory; a test's
            // is, standing for the title.)
            if !self
                .images
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&base)
            {
                return None;
            }
            let text = match inner.texts.get(&base) {
                Some(t) => t.clone(),
                None => {
                    // SAFETY: a loaded image.
                    let t = Arc::new(unsafe { Text::read(base) }?);
                    inner.texts.insert(base, t.clone());
                    t
                }
            };
            return (text.contains(rip) || rip == base).then_some(text);
        }
        // Executable memory that is no image (generated code, our own
        // trampolines and stubs, other layers' stubs): not ours to patch.
        None
    }

    /// Whether any site went to `int3`.
    pub fn has_breakpoints(&self) -> bool {
        !self
            .breakpoints
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// The trampoline for an int3 site at `rip` (the int3's address or the
    /// one after it), for the breakpoint handler.
    pub fn breakpoint(&self, rip: u64) -> Option<u64> {
        let map = self.breakpoints.read().unwrap();
        let t = map
            .get(&rip)
            .or_else(|| map.get(&rip.wrapping_sub(1)))
            .copied();
        if t.is_some() {
            self.breakpoint_hits.fetch_add(1, Ordering::Relaxed);
        }
        t
    }

    /// Called by the fault handler for a write fault on guarded graphics
    /// memory at instruction `*rip`. May change `*rip`.
    pub fn on_write_fault(&self, rip: &mut u64) -> Outcome {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&new) = inner.moved.get(rip) {
            *rip = new;
            self.redirected.fetch_add(1, Ordering::Relaxed);
            return Outcome::Redirected;
        }
        let Some(text) = self.text_of(&mut inner, *rip) else {
            *inner
                .refused_faults
                .entry(Refusal::NotPatchable)
                .or_default() += 1;
            return Outcome::Refused(Refusal::NotPatchable);
        };
        match inner.sites.get(rip) {
            Some(Site::Patched(record)) => {
                if self.still_written(*rip, record) {
                    return Outcome::Patched;
                }
                // The code changed under the patch (JIT, self-modifying
                // code): forget the old patch and patch what is there now.
                let Some(Site::Patched(old)) = inner.sites.remove(rip) else {
                    unreachable!()
                };
                for m in old.moved {
                    inner.moved.remove(&m);
                }
                if let Some((island, _)) = old.island {
                    inner.islands.remove(&island);
                }
                inner.written.remove(&old.at);
                self.breakpoints.write().unwrap().remove(rip);
                inner.repatched += 1;
            }
            Some(Site::Refused(r)) => {
                let r = *r;
                *inner.refused_faults.entry(r).or_default() += 1;
                return Outcome::Refused(r);
            }
            None => {}
        }
        let began = std::time::Instant::now();
        let result = self.patch(&mut inner, &text, *rip);
        self.patch_ns
            .fetch_add(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match result {
            Ok((record, moved_to)) => {
                *inner.strategies.entry(record.strategy).or_default() += 1;
                inner.sites.insert(*rip, Site::Patched(record));
                // A block-start patch moved the store itself: the faulting
                // thread goes to its copy.
                match moved_to {
                    Some(new) => {
                        *rip = new;
                        Outcome::Redirected
                    }
                    None => Outcome::Patched,
                }
            }
            Err(r) => {
                self.settle(&mut inner, *rip, &Err(r));
                Outcome::Refused(r)
            }
        }
    }

    fn still_written(&self, _rip: u64, record: &Record) -> bool {
        // SAFETY: code bytes we wrote earlier; still mapped (the title
        // faulted there).
        let site =
            unsafe { std::slice::from_raw_parts(record.at as *const u8, record.written.len()) };
        let island_ok = record.island.is_none_or(
            |(at, bytes)| unsafe { std::slice::from_raw_parts(at as *const u8, 5) } == bytes,
        );
        site == record.written && island_ok
    }

    fn decode(text: &Text, ip: u64) -> Instruction {
        let end = text.section_end(ip);
        // SAFETY: `ip` is inside an executable region; read at most 15
        // bytes and not past its end.
        let bytes =
            unsafe { std::slice::from_raw_parts(ip as *const u8, (end - ip).min(15) as usize) };
        Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE).decode()
    }

    /// An area within reach of `near` with room for a trampoline.
    fn area_near(&self, inner: &mut Inner, near: u64) -> Result<u64, Refusal> {
        let reach = (1u64 << 31) - AREA as u64 - 4096;
        if let Some(a) = inner
            .areas
            .iter()
            .find(|a| a.start.abs_diff(near) < reach && a.next + TRAMPOLINE_MAX as u64 <= a.end)
        {
            return Ok(a.next);
        }
        let step = 64u64 << 20;
        // Below the code first: unwind data refers to the code's function
        // entries by offsets from the area, which must be positive.
        for k in 1..30u64 {
            for at in [near.saturating_sub(k * step), near + k * step] {
                let at = at & !0xFFFF;
                let slots = [at >> SLOT_SHIFT, (at + AREA as u64) >> SLOT_SHIFT];
                let _ = slots;
                if at < 0x10000 {
                    continue;
                }
                // SAFETY: asks for fresh memory at a hint; null if taken.
                let got = unsafe {
                    VirtualAlloc(
                        at as *const _,
                        AREA,
                        MEM_RESERVE | MEM_COMMIT,
                        PAGE_EXECUTE_READWRITE,
                    )
                } as u64;
                if got != 0 {
                    unwind::register_area(got, AREA as u64);
                    inner.areas.push(Area {
                        start: got,
                        next: got,
                        end: got + AREA as u64 - unwind::UNWIND_BYTES,
                    });
                    self.area_bounds
                        .write()
                        .unwrap()
                        .push((got, got + AREA as u64));
                    return Ok(got);
                }
            }
        }
        Err(Refusal::Full)
    }

    /// The span relocation would move: the store and what follows it until
    /// five bytes are covered.
    /// The span that ends at `store` and starts at the start of its basic
    /// block, if that covers five bytes: nothing can enter a block but at
    /// its start, so a jump there and the block moved, store included,
    /// needs nothing after the store. Also takes a `ret` right after it.
    /// Only in a function the scan proved closed (its block starts known).
    pub fn backward_span(text: &Text, store: Instruction) -> Option<Vec<Instruction>> {
        let (begin, end, _) = text.runtime_function(store.ip())?;
        if !text.relocation_proven(store.ip()) {
            return None;
        }
        Self::backward_span_in(text, store, (begin, end))
    }

    /// `backward_span` within known bounds of the code around the store.
    pub fn backward_span_in(
        text: &Text,
        store: Instruction,
        (begin, end): (u64, u64),
    ) -> Option<Vec<Instruction>> {
        // SAFETY: the function's bytes in an executable section.
        let bytes =
            unsafe { std::slice::from_raw_parts(begin as *const u8, (end - begin) as usize) };
        let code: Vec<Instruction> = Decoder::with_ip(64, bytes, begin, DecoderOptions::NONE)
            .into_iter()
            .collect();
        let k = code.iter().position(|i| i.ip() == store.ip())?;
        // Walk back only as far as five bytes need. Each instruction taken in
        // must not be a branch target (the span's first may be), and the one
        // before must fall into it: plain code, or a conditional branch,
        // which only leaves the span (the encoder re-targets it). Not past a
        // call: the instruction after it is where other threads return to.
        let mut j = k;
        let mut bytes_covered = store.len() as u64;
        while bytes_covered < JUMP as u64 && j > 0 {
            let ip = code[j].ip();
            let before = &code[j - 1];
            let falls_in = match before.flow_control() {
                FlowControl::Next => true,
                FlowControl::ConditionalBranch => {
                    before.op_kind(0) == iced_x86::OpKind::NearBranch64
                }
                _ => false,
            };
            if text.targets_between(ip - 1, ip + 1) || !falls_in {
                break;
            }
            j -= 1;
            bytes_covered += code[j].len() as u64;
        }
        let mut span: Vec<Instruction> = code[j..=k].to_vec();
        if bytes_covered < JUMP as u64 {
            match code.get(k + 1) {
                Some(r)
                    if r.flow_control() == FlowControl::Return
                        && !text.targets_between(r.ip() - 1, r.ip() + 1) =>
                {
                    bytes_covered += r.len() as u64;
                    span.push(*r);
                }
                _ => return None,
            }
        }
        (bytes_covered >= JUMP as u64).then_some(span)
    }

    pub fn span(text: &Text, first: Instruction) -> Result<Vec<Instruction>, Refusal> {
        let rip = first.ip();
        let mut span = vec![first];
        let mut len = first.len();
        while len < JUMP {
            let last = span.last().unwrap();
            if matches!(
                last.flow_control(),
                FlowControl::UnconditionalBranch | FlowControl::Return
            ) {
                return Err(Refusal::Span);
            }
            let next = Self::decode(text, rip + len as u64);
            if next.is_invalid() || !text.contains(next.ip()) {
                return Err(Refusal::Span);
            }
            match next.flow_control() {
                FlowControl::Next | FlowControl::Return => {}
                FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch
                    if next.op_kind(0) == OpKind::NearBranch64 => {}
                _ => return Err(Refusal::Span),
            }
            len += next.len();
            span.push(next);
        }
        if text.targets_between(rip, rip + len as u64) {
            return Err(Refusal::Span);
        }
        Ok(span)
    }

    /// Whether code bytes in `[from, to)` were written by an earlier patch.
    fn overlaps_patch(inner: &Inner, from: u64, to: u64) -> bool {
        inner
            .written
            .range(..to)
            .next_back()
            .is_some_and(|(_, &end)| end > from)
    }

    /// Everything for patching the store at `rip` but the writing: the
    /// strategy, the trampoline (put in its area, with unwind data), the
    /// bytes for the site. The bytes it will write are reserved against later
    /// spans and islands, and an int3 site's trampoline is told to the
    /// breakpoint handler; [`Self::unstage`] gives those back.
    fn stage(&self, inner: &mut Inner, text: &Text, rip: u64) -> Result<Staged, Refusal> {
        let first = Self::decode(text, rip);
        let store = classify::classify(&first, &mut inner.factory, self.xsave_bytes)?;
        let len = first.len();
        let island = if len < JUMP {
            text.island_near(rip + 2, &inner.islands)
        } else {
            None
        };
        // A span may not cover an earlier patch's bytes.
        let clear = |span: &Vec<Instruction>| {
            let end = span.last().map_or(0, |i| i.next_ip());
            !Self::overlaps_patch(inner, span[0].ip(), end)
        };
        let relocatable = if text.image && len < JUMP && island.is_none() {
            Self::span(text, first).ok().filter(|s| clear(s))
        } else {
            None
        };
        // A leaf without `.pdata`, bounded and checked: as good as a closed
        // function for moving code (see `Text::leaf_proven`).
        let leaf = if text.image && len < JUMP && island.is_none() && !text.relocation_proven(rip) {
            text.leaf_proven(rip)
        } else {
            None
        };
        let proven = text.relocation_proven(rip) || leaf.is_some();
        let block = if text.image && len < JUMP && island.is_none() {
            Self::backward_span(text, first)
                .or_else(|| leaf.and_then(|b| Self::backward_span_in(text, first, b)))
                .filter(|s| clear(s))
        } else {
            None
        };
        let (strategy, span) = if len >= JUMP {
            (Strategy::Direct, vec![first])
        } else if island.is_some() {
            (Strategy::Island, vec![first])
        } else if let Some(span) = relocatable.clone().filter(|_| proven) {
            (Strategy::Relocate, span)
        } else if let Some(span) = block {
            (Strategy::BlockStart, span)
        } else if let Some(span) = relocatable.filter(|_| self.config.relocate) {
            (Strategy::RelocateUnproven, span)
        } else if self.config.int3 {
            (Strategy::Int3, vec![first])
        } else {
            return Err(Refusal::Int3Off);
        };
        // Where the jump goes: the store, or its block's start.
        let site = span[0].ip();
        let span_len: usize = span.iter().map(|i| i.len()).sum();
        if strategy == Strategy::Int3 && std::env::var_os("VICTORY_BARRIER_DEBUG").is_some() {
            let span = match Self::span(text, first) {
                Ok(_) => "span ok".to_string(),
                Err(_) => {
                    "span refused (a branch target, or a ret/jmp, within 5 bytes)".to_string()
                }
            };
            eprintln!(
                "int3 site {} ({} bytes): {}; {}",
                symbol(rip),
                len,
                text.why_not_proven(rip),
                span
            );
        }

        // The store, and any other store the span covers, gets a barrier.
        let mut stores = Vec::with_capacity(span.len());
        for instr in &span {
            stores.push(if instr.ip() == rip {
                Some(store)
            } else {
                match classify::classify(instr, &mut inner.factory, self.xsave_bytes) {
                    Ok(s @ Store::Inline(_)) => Some(s),
                    _ => None,
                }
            });
        }
        let site_info: *const StringSite = match store {
            Store::String { kind, element, rep } => {
                Box::leak(Box::new(StringSite { kind, element, rep }))
            }
            Store::Inline(_) => std::ptr::null(),
        };
        let at = self.area_near(inner, rip)?;
        let encoded = emit::trampoline(at, &span, &stores, self.tables, site_info)?;
        if encoded.code.len() > TRAMPOLINE_MAX {
            return Err(Refusal::Encode);
        }
        // What to write where.
        let (site_bytes, island_bytes) = match strategy {
            Strategy::Direct
            | Strategy::Relocate
            | Strategy::BlockStart
            | Strategy::RelocateUnproven => {
                let mut b = emit::jump32(site, at).ok_or(Refusal::Encode)?.to_vec();
                b.resize(span_len, 0xCC);
                (b, None)
            }
            Strategy::Island => {
                let island = island.unwrap();
                let mut b = emit::jump8(rip, island).ok_or(Refusal::Encode)?.to_vec();
                b.resize(len, 0xCC);
                (
                    b,
                    Some((island, emit::jump32(island, at).ok_or(Refusal::Encode)?)),
                )
            }
            Strategy::Int3 => (vec![0xCC], None),
        };
        // SAFETY: fresh trampoline memory nobody executes yet.
        unsafe {
            std::ptr::copy_nonoverlapping(encoded.code.as_ptr(), at as *mut u8, encoded.code.len())
        };
        let chained = if text.image {
            text.runtime_function(rip)
        } else {
            None
        };
        unwind::describe(at, &encoded, text.base, chained);

        // An int3 site's trampoline is known before its int3 is written
        // (inserting may allocate, which nothing may do while the others
        // are stopped: one may hold the heap's lock).
        if strategy == Strategy::Int3 {
            self.breakpoints.write().unwrap().insert(rip, at);
        }
        let area = inner.areas.iter_mut().find(|a| a.next == at).unwrap();
        area.next = (at + encoded.code.len() as u64 + 15) & !15;
        inner.written.insert(site, site + site_bytes.len() as u64);
        if let Some((island, _)) = island_bytes {
            inner.islands.insert(island);
        }
        Ok(Staged {
            rip,
            site,
            span_len,
            site_bytes,
            island: island_bytes,
            strategy,
            moved: encoded.moved,
            string: matches!(store, Store::String { .. }),
            hash: instruction_hash(rip, len),
        })
    }

    /// Gives back what [`Self::stage`] reserved, for a patch not written.
    /// (The trampoline's room is not reused.)
    fn unstage(&self, inner: &mut Inner, staged: &Staged) {
        inner.written.remove(&staged.site);
        if let Some((island, _)) = staged.island {
            inner.islands.remove(&island);
        }
        if staged.strategy == Strategy::Int3 {
            self.breakpoints.write().unwrap().remove(&staged.rip);
        }
    }

    /// Writes `staged` patches with the other threads stopped once. A thread
    /// stopped inside a span goes to its instruction's copy; a patch with a
    /// thread where it cannot go (between instructions) is not written
    /// (`ThreadInSpan`, tried again at its next fault). For each: its
    /// record, and where a block-start patch moved the store itself.
    fn commit(
        &self,
        inner: &mut Inner,
        staged: Vec<Staged>,
    ) -> Vec<(u64, Result<(Record, Option<u64>), Refusal>)> {
        // Made before the others stop: nothing may allocate meanwhile.
        let mut outcome: Vec<Option<Refusal>> = vec![None; staged.len()];
        let mut frozen = threads::Frozen::others();
        frozen.each_rip(|ip| {
            for (k, s) in staged.iter().enumerate() {
                if s.in_span(ip) && s.moved_to(ip).is_none() {
                    outcome[k] = Some(Refusal::ThreadInSpan);
                }
            }
        });
        let redirects = frozen.redirect(|ip| {
            staged
                .iter()
                .enumerate()
                .find(|(k, s)| outcome[*k].is_none() && s.in_span(ip))
                .map(|(_, s)| s.moved_to(ip))
        });
        if redirects.is_none() {
            // A thread could not be moved: nothing is written. (Threads
            // moved already run the copies, which do the same.)
            outcome
                .iter_mut()
                .for_each(|o| *o = Some(Refusal::ThreadInSpan));
        }
        let write = |to: u64, bytes: &[u8]| -> Result<(), Refusal> {
            let mut old: PAGE_PROTECTION_FLAGS = 0;
            // SAFETY: code bytes of the site or its island; other threads
            // are stopped outside them.
            unsafe {
                if VirtualProtect(
                    to as *const _,
                    bytes.len(),
                    PAGE_EXECUTE_READWRITE,
                    &mut old,
                ) == 0
                {
                    return Err(Refusal::Encode);
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, bytes.len());
                VirtualProtect(to as *const _, bytes.len(), old, &mut old);
                FlushInstructionCache(GetCurrentProcess(), to as *const _, bytes.len());
            }
            Ok(())
        };
        for (k, s) in staged.iter().enumerate() {
            if outcome[k].is_none() {
                let written = s
                    .island
                    .map_or(Ok(()), |(island, bytes)| write(island, &bytes))
                    .and_then(|()| write(s.site, &s.site_bytes));
                outcome[k] = written.err();
            }
        }
        drop(frozen);
        self.redirected
            .fetch_add(redirects.unwrap_or(0), Ordering::Relaxed);
        staged
            .into_iter()
            .zip(outcome)
            .map(|(s, refused)| {
                if let Some(r) = refused {
                    self.unstage(inner, &s);
                    return (s.rip, Err(r));
                }
                inner.moved.extend(s.moved.iter().copied());
                if s.string {
                    self.string_sites.fetch_add(1, Ordering::Relaxed);
                }
                let moved_to = (s.site != s.rip).then(|| s.moved_to(s.rip)).flatten();
                let record = Record {
                    at: s.site,
                    written: s.site_bytes,
                    island: s.island,
                    strategy: s.strategy,
                    moved: s.moved.iter().map(|m| m.0).collect(),
                    hash: s.hash,
                };
                (s.rip, Ok((record, moved_to)))
            })
            .collect()
    }

    fn patch(
        &self,
        inner: &mut Inner,
        text: &Text,
        rip: u64,
    ) -> Result<(Record, Option<u64>), Refusal> {
        let staged = self.stage(inner, text, rip)?;
        self.commit(inner, vec![staged]).pop().expect("one patch").1
    }

    /// Notes what became of patching the store at `rip`.
    fn settle(&self, inner: &mut Inner, rip: u64, result: &Result<Record, Refusal>) {
        match result {
            Ok(record) => *inner.strategies.entry(record.strategy).or_default() += 1,
            Err(r) => {
                *inner.refused_faults.entry(*r).or_default() += 1;
                // A thread in the way is a matter of timing: try again next
                // time. Everything else is a property of the site.
                if *r != Refusal::ThreadInSpan {
                    inner.sites.insert(rip, Site::Refused(*r));
                    *inner.refusals.entry(*r).or_default() += 1;
                }
            }
        }
    }

    /// Patches the stores at `rips` that are not patched yet (the sites a run
    /// before this one patched, [`Self::patched_sites`]), with the other
    /// threads stopped once for all of them. A store inside another's moved
    /// span is left (its barrier is in that span's trampoline). How many
    /// were patched.
    pub fn patch_many(&self, rips: &[u64]) -> usize {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let began = std::time::Instant::now();
        let mut staged = Vec::new();
        for &rip in rips {
            if inner.sites.contains_key(&rip) || Self::overlaps_patch(&inner, rip, rip + 1) {
                continue;
            }
            let Some(text) = self.text_of(&mut inner, rip) else {
                continue;
            };
            match self.stage(&mut inner, &text, rip) {
                Ok(s) => staged.push(s),
                Err(r) => self.settle(&mut inner, rip, &Err(r)),
            }
        }
        let mut done = 0;
        for (rip, result) in self.commit(&mut inner, staged) {
            let result = result.map(|(record, _)| record);
            self.settle(&mut inner, rip, &result);
            if let Ok(record) = result {
                inner.sites.insert(rip, Site::Patched(record));
                done += 1;
            }
        }
        self.patch_ns
            .fetch_add(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
        done
    }

    /// The stores patched so far: where, and a hash of the instruction they
    /// were ([`instruction_hash`]), to be patched ahead next time
    /// ([`Self::patch_many`]) where that instruction is still there.
    pub fn patched_sites(&self) -> Vec<(u64, u64)> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut sites: Vec<(u64, u64)> = inner
            .sites
            .iter()
            .filter_map(|(&rip, site)| match site {
                Site::Patched(record) => Some((rip, record.hash)),
                Site::Refused(_) => None,
            })
            .collect();
        sites.sort_unstable();
        sites
    }

    pub fn stats(&self) -> PatchStats {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        PatchStats {
            patched: inner.strategies.values().sum(),
            strategies: inner.strategies.clone(),
            refused_sites: inner.refusals.clone(),
            refused_faults: inner.refused_faults.clone(),
            string_sites: self.string_sites.load(Ordering::Relaxed),
            redirected_threads: self.redirected.load(Ordering::Relaxed),
            repatched: inner.repatched,
            breakpoints: self.breakpoint_hits.load(Ordering::Relaxed),
            patch_ns: self.patch_ns.load(Ordering::Relaxed),
            modules: inner
                .texts
                .values()
                .map(|t| (t.base, t.target_count(), t.stats))
                .collect(),
        }
    }
}

/// "image+offset" of a code address, for diagnostics. (Not dbghelp: the
/// title has no symbols, and VIctory's crash reports use dbghelp.)
fn symbol(at: u64) -> String {
    let mut module = std::ptr::null_mut();
    // SAFETY: looks up the image holding `at`, without a reference.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            at as *const u16,
            &mut module,
        )
    } != 0;
    if found {
        format!("image {:#x}+{:#x}", module as u64, at - module as u64)
    } else {
        format!("{at:#x}")
    }
}
