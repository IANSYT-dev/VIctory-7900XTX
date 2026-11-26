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
pub mod shape;
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
            area_bounds: RwLock::new(Vec::new()),
            string_sites: AtomicU64::new(0),
            redirected: AtomicU64::new(0),
            breakpoint_hits: AtomicU64::new(0),
            patch_ns: AtomicU64::new(0),
        })
    }

    /// Scans an image ahead of its first fault (the title, at load).
    pub fn prepare(&self, module: u64) -> Option<usize> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        self.text_of(&mut inner, module).map(|t| t.target_count())
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
            if !self.config.images.contains(&base) {
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
                *inner.refused_faults.entry(r).or_default() += 1;
                // A thread in the way is a matter of timing: try again next
                // time. Everything else is a property of the site.
                if r != Refusal::ThreadInSpan {
                    inner.sites.insert(*rip, Site::Refused(r));
                    *inner.refusals.entry(r).or_default() += 1;
                }
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

    fn patch(
        &self,
        inner: &mut Inner,
        text: &Text,
        rip: u64,
    ) -> Result<(Record, Option<u64>), Refusal> {
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
        let forget_breakpoint = |this: &Self| {
            if strategy == Strategy::Int3 {
                this.breakpoints.write().unwrap().remove(&rip);
            }
        };
        // Everything allocated; now stop the world for the write.
        let mut frozen = threads::Frozen::others();
        let in_span = |ip: u64| ip > site && ip < site + span_len as u64;
        let redirects = frozen.redirect(|ip| {
            in_span(ip).then(|| encoded.moved.iter().find(|m| m.0 == ip).map(|m| m.1))
        });
        let Some(redirects) = redirects else {
            drop(frozen);
            forget_breakpoint(self);
            return Err(Refusal::ThreadInSpan);
        };
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
        let written = island_bytes
            .map_or(Ok(()), |(island, bytes)| write(island, &bytes))
            .and_then(|()| write(site, &site_bytes));
        drop(frozen);
        if let Err(e) = written {
            forget_breakpoint(self);
            return Err(e);
        }

        let area = inner.areas.iter_mut().find(|a| a.next == at).unwrap();
        area.next = (at + encoded.code.len() as u64 + 15) & !15;
        inner.moved.extend(encoded.moved.iter().copied());
        if let Some((island, _)) = island_bytes {
            inner.islands.insert(island);
        }
        inner.written.insert(site, site + site_bytes.len() as u64);
        self.redirected.fetch_add(redirects, Ordering::Relaxed);
        if matches!(store, Store::String { .. }) {
            self.string_sites.fetch_add(1, Ordering::Relaxed);
        }
        let moved_to = (site != rip)
            .then(|| encoded.moved.iter().find(|m| m.0 == rip).map(|m| m.1))
            .flatten();
        Ok((
            Record {
                at: site,
                written: site_bytes,
                island: island_bytes,
                strategy,
                moved: encoded.moved.iter().map(|m| m.0).collect(),
            },
            moved_to,
        ))
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

/// Where every store instruction of an image would end up, without
/// running it: `xbc survey`.
#[derive(Default, Debug)]
pub struct Survey {
    pub instructions: u64,
    /// Stores through rsp: the stack, which the patcher never sees.
    pub stack_stores: u64,
    pub strategies: BTreeMap<Strategy, u64>,
    pub refused: BTreeMap<Refusal, u64>,
    /// Why short stores ended on int3.
    pub int3_why: BTreeMap<&'static str, u64>,
    /// Anonymised shapes around what is left on int3, by kind: how many
    /// times each shape occurs.
    pub shapes: BTreeMap<&'static str, HashMap<String, u64>>,
}

/// Classifies every instruction of `text`'s code sections (a linear sweep,
/// so data in code adds some noise) as the patcher would on a fault.
pub fn survey(text: &Text) -> Survey {
    let mut s = Survey::default();
    for (window, jmp) in &text.unread {
        let mut list: Vec<Instruction> = window.iter().rev().take(10).rev().copied().collect();
        list.push(*jmp);
        *s.shapes
            .entry("indirect jump not read")
            .or_default()
            .entry(shape::render(&list, text, None))
            .or_default() += 1;
    }
    let mut recent: Vec<Instruction> = Vec::with_capacity(8);
    let mut factory = InstructionInfoFactory::new();
    let xsave = string::xsave_area_bytes();
    let taken = BTreeSet::new();
    for &(start, end) in &text.sections {
        // SAFETY: an executable section of a loaded image.
        let bytes =
            unsafe { std::slice::from_raw_parts(start as *const u8, (end - start) as usize) };
        for instr in Decoder::with_ip(64, bytes, start, DecoderOptions::NONE) {
            s.instructions += 1;
            if recent.len() == 6 {
                recent.remove(0);
            }
            let before = recent.clone();
            recent.push(instr);
            match classify::classify(&instr, &mut factory, xsave) {
                Err(Refusal::NotAStore | Refusal::Undecodable) => {}
                Err(Refusal::Stack) => s.stack_stores += 1,
                Err(r) => {
                    if std::env::var_os("VICTORY_BARRIER_DEBUG").is_some()
                        && r == Refusal::NoScratch
                    {
                        use iced_x86::Formatter;
                        let mut t = String::new();
                        iced_x86::IntelFormatter::new().format(&instr, &mut t);
                        eprintln!("refused NoScratch at +{:#x}: {t}", instr.ip() - text.base);
                    }
                    *s.refused.entry(r).or_default() += 1
                }
                Ok(_) => {
                    let rip = instr.ip();
                    let strategy = if instr.len() >= JUMP {
                        Strategy::Direct
                    } else if text.island_near(rip + 2, &taken).is_some() {
                        Strategy::Island
                    } else {
                        let span = Patcher::span(text, instr);
                        let leaf = if text.relocation_proven(rip) {
                            None
                        } else {
                            text.leaf_proven(rip)
                        };
                        let proven = text.relocation_proven(rip) || leaf.is_some();
                        if span.is_ok() && proven {
                            Strategy::Relocate
                        } else {
                            if Patcher::backward_span(text, instr)
                                .or_else(|| {
                                    leaf.and_then(|b| Patcher::backward_span_in(text, instr, b))
                                })
                                .is_some()
                            {
                                Strategy::BlockStart
                            } else {
                                let mut why = if proven {
                                    // What stops the block-start jump.
                                    if before.last().is_some_and(|b| {
                                        b.flow_control() == FlowControl::Call
                                            || b.flow_control() == FlowControl::IndirectCall
                                    }) {
                                        "store right after a call"
                                    } else if text.is_target(rip) {
                                        "store is a branch target"
                                    } else {
                                        "branch target or ret within 5 bytes"
                                    }
                                } else {
                                    text.unproven_kind(rip)
                                };
                                if why == "no .pdata (leaf or assembly)"
                                    && text.leaf_piece(rip).is_some()
                                {
                                    why = "no .pdata (a leaf beside an unread jump)";
                                }
                                *s.int3_why.entry(why).or_default() += 1;
                                if matches!(
                                    why,
                                    "branch target or ret within 5 bytes"
                                        | "store right after a call"
                                        | "store is a branch target"
                                        | "no .pdata (leaf or assembly)"
                                        | "no .pdata (a leaf beside an unread jump)"
                                ) {
                                    // The store, what led to it and what follows.
                                    let mut list = before.clone();
                                    list.push(instr);
                                    let mut next = instr.next_ip();
                                    for _ in 0..3 {
                                        let i = Patcher::decode(text, next);
                                        if i.is_invalid() {
                                            break;
                                        }
                                        next = i.next_ip();
                                        list.push(i);
                                    }
                                    *s.shapes
                                        .entry(why)
                                        .or_default()
                                        .entry(shape::render(&list, text, Some(rip)))
                                        .or_default() += 1;
                                }
                                Strategy::Int3
                            }
                        }
                    };
                    *s.strategies.entry(strategy).or_default() += 1;
                }
            }
        }
    }
    s
}

/// "function+offset" from the debug symbols, for diagnostics.
fn symbol(at: u64) -> String {
    use windows_sys::Win32::System::Diagnostics::Debug::{
        SYMBOL_INFO, SymFromAddr, SymInitialize, SymSetOptions,
    };
    static INIT: std::sync::Once = std::sync::Once::new();
    // SAFETY: dbghelp on this process, once.
    INIT.call_once(|| unsafe {
        SymSetOptions(0x2 | 0x4);
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned()))
            .unwrap_or_default();
        let dir = std::ffi::CString::new(dir).unwrap_or_default();
        SymInitialize(GetCurrentProcess(), dir.as_ptr() as *const u8, 1);
    });
    #[repr(C)]
    struct Buf {
        info: SYMBOL_INFO,
        name: [u8; 512],
    }
    // SAFETY: plain data, filled in by dbghelp.
    let mut b: Buf = unsafe { std::mem::zeroed() };
    b.info.SizeOfStruct = size_of::<SYMBOL_INFO>() as u32;
    b.info.MaxNameLen = 512;
    let mut displacement = 0u64;
    // SAFETY: as above.
    if unsafe { SymFromAddr(GetCurrentProcess(), at, &mut displacement, &mut b.info) } != 0 {
        // SAFETY: a NUL-terminated name dbghelp wrote.
        let name = unsafe { std::ffi::CStr::from_ptr(b.info.Name.as_ptr()) }.to_string_lossy();
        format!("{name}+{displacement:#x}")
    } else {
        format!("{at:#x}")
    }
}
