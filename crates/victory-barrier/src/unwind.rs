//! Unwind data for trampolines, so a stack walk through one works.
//!
//! A thread stopped in a trampoline (a profiler, a crash reporter, a
//! title's watchdog walking its threads' stacks) is unwound by Windows with
//! the function table entry for its RIP. Trampoline memory has none, so
//! Windows would take the frame for a leaf function and read the return
//! address from `[rsp]`: one of the barrier's pushed registers.
//!
//! Each trampoline therefore gets entries for every range of its code where
//! the stack differs (after the red-zone skip, after each push, while the
//! string helper's frame is set up), each describing what the trampoline
//! pushed, chained (`UNW_FLAG_CHAININFO`) to the entry of the function the
//! site is in. Unwinding applies the trampoline's codes and then the
//! function's, as if the thread were still at the site. Windows finds the
//! entries through a function table callback over each trampoline area.
//!
//! The jump back to the site is `jmp [rip+0]` with the target after it:
//! Windows takes a `jmp rel32` out of a function at RIP for a tail call
//! (an epilog) and would unwind it as one.
//!
//! Not covered: an island's `jmp` (one instruction, in padding outside any
//! function), JIT code (no function entry to chain to), and exception
//! handlers of the function (a chained entry carries none: an exception at
//! the store itself is not given to the function's own `__except`).

use crate::emit::Encoded;
use iced_x86::{Code, Register};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The end of each area holds its unwind data.
pub const UNWIND_BYTES: u64 = 4 << 20;
const ENTRIES: usize = 1 << 18;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RuntimeFunction {
    begin: u32,
    end: u32,
    unwind: u32,
}

struct Table {
    base: u64,
    entries: Box<[RuntimeFunction]>,
    len: AtomicUsize,
    /// Next free byte of unwind data (an address in the area).
    data: Mutex<(u64, u64)>,
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    fn RtlInstallFunctionTableCallback(
        id: u64,
        base: u64,
        length: u32,
        callback: unsafe extern "system" fn(u64, *mut c_void) -> *mut RuntimeFunction,
        context: *mut c_void,
        dll: *const u16,
    ) -> u8;
}

static TABLES: Mutex<Vec<&'static Table>> = Mutex::new(Vec::new());

unsafe extern "system" fn lookup(pc: u64, context: *mut c_void) -> *mut RuntimeFunction {
    // SAFETY: the context is the leaked table registered with the callback.
    let t = unsafe { &*(context as *const Table) };
    let n = t.len.load(Ordering::Acquire);
    let rva = (pc - t.base) as u32;
    let entries = &t.entries[..n];
    let i = entries.partition_point(|e| e.begin <= rva);
    match i.checked_sub(1).map(|i| &entries[i]) {
        Some(e) if rva < e.end => e as *const RuntimeFunction as *mut RuntimeFunction,
        _ => std::ptr::null_mut(),
    }
}

/// Registers the function table callback for a new trampoline area.
pub fn register_area(base: u64, bytes: u64) {
    let table: &'static Table = Box::leak(Box::new(Table {
        base,
        entries: vec![RuntimeFunction::default(); ENTRIES].into_boxed_slice(),
        len: AtomicUsize::new(0),
        data: Mutex::new((base + bytes - UNWIND_BYTES, base + bytes)),
    }));
    // VICTORY_BARRIER_NO_UNWIND=1 leaves the area without unwind data, to see what a
    // stack walk does then.
    if std::env::var_os("VICTORY_BARRIER_NO_UNWIND").is_some() {
        return;
    }
    // SAFETY: the area and the table live as long as the process.
    unsafe {
        RtlInstallFunctionTableCallback(
            base | 3,
            base,
            bytes as u32,
            lookup,
            table as *const Table as *mut c_void,
            std::ptr::null(),
        );
    }
    TABLES.lock().unwrap().push(table);
}

/// What the stack holds above the site's own frame at a point of a
/// trampoline.
#[derive(Clone, PartialEq, Eq)]
struct State {
    /// Pushed in order: Some(register) or None (8 opaque bytes; the red
    /// zone counts as 16 of them).
    pushed: Vec<Option<Register>>,
    /// rbx holds rsp as it was after this many items.
    frame: Option<usize>,
}

fn register_number(r: Register) -> u8 {
    (r as u32 - Register::RAX as u32) as u8
}

/// UNWIND_INFO bytes for a state, chained to `chain` if given.
fn unwind_info(state: &State, chain: Option<RuntimeFunction>) -> Vec<u8> {
    const PUSH_NONVOL: u8 = 0;
    const ALLOC_SMALL: u8 = 2;
    const SET_FPREG: u8 = 3;
    // (op, info) in reverse order of what the trampoline did: setting the
    // frame register was done last, so it comes first.
    let mut codes: Vec<(u8, u8)> = Vec::new();
    if state.frame.is_some() {
        codes.push((SET_FPREG, 0));
    }
    let upto = state.frame.unwrap_or(state.pushed.len());
    let mut opaque = 0u8;
    let flush = |codes: &mut Vec<(u8, u8)>, opaque: &mut u8| {
        while *opaque > 0 {
            let n = (*opaque).min(16);
            codes.push((ALLOC_SMALL, n - 1));
            *opaque -= n;
        }
    };
    for item in state.pushed[..upto].iter().rev() {
        match item {
            Some(r) => {
                flush(&mut codes, &mut opaque);
                codes.push((PUSH_NONVOL, register_number(*r)));
            }
            None => opaque += 1,
        }
    }
    flush(&mut codes, &mut opaque);
    let flags = if chain.is_some() { 4u8 } else { 0 };
    // Frame register rbx, at offset 0 from rsp when it was set.
    let frame_register = if state.frame.is_some() {
        register_number(Register::RBX)
    } else {
        0
    };
    let mut b = vec![1 | flags << 3, 0, codes.len() as u8, frame_register];
    for (op, info) in &codes {
        // Offset 0: with a prolog of size 0, every code applies.
        b.push(0);
        b.push(op | info << 4);
    }
    if codes.len() % 2 == 1 {
        b.extend_from_slice(&[0, 0]);
    }
    if let Some(c) = chain {
        b.extend_from_slice(&c.begin.to_le_bytes());
        b.extend_from_slice(&c.end.to_le_bytes());
        b.extend_from_slice(&c.unwind.to_le_bytes());
    }
    b
}

/// Describes the trampoline at `at` (encoded as `encoded`) to Windows.
/// `chained` is the site's function: (begin, end, unwind data), absolute.
pub fn describe(at: u64, encoded: &Encoded, _image_base: u64, chained: Option<(u64, u64, u64)>) {
    let tables = TABLES.lock().unwrap();
    let Some(table) = tables
        .iter()
        .find(|t| at >= t.base && at < t.base + (1 << 31))
    else {
        return;
    };
    let rva = |x: u64| u32::try_from(x.wrapping_sub(table.base)).ok();
    let chain = match chained {
        Some((b, e, u)) => match (rva(b), rva(e), rva(u)) {
            (Some(begin), Some(end), Some(unwind)) => Some(RuntimeFunction { begin, end, unwind }),
            // Code below the area: offsets would be negative. Unwinding a
            // thread here stays as wrong as without this.
            _ => return,
        },
        None => None,
    };
    // The state after each instruction.
    let mut state = State {
        pushed: Vec::new(),
        frame: None,
    };
    let mut ranges: Vec<(u32, State)> = vec![(0, state.clone())];
    for (k, i) in encoded.instructions.iter().enumerate() {
        match i.code() {
            Code::Push_r64 => state.pushed.push(Some(i.op0_register())),
            Code::Pushfq => state.pushed.push(None),
            Code::Pop_r64 | Code::Popfq => {
                state.pushed.pop();
            }
            Code::Lea_r64_m if i.op0_register() == Register::RSP && state.frame.is_none() => {
                let by = i.memory_displacement64() as i64;
                if by < 0 {
                    state
                        .pushed
                        .extend(std::iter::repeat_n(None, (-by / 8) as usize));
                } else {
                    let keep = state.pushed.len().saturating_sub((by / 8) as usize);
                    state.pushed.truncate(keep);
                }
            }
            Code::Mov_r64_rm64
                if i.op0_register() == Register::RBX && i.op1_register() == Register::RSP =>
            {
                state.frame = Some(state.pushed.len());
            }
            Code::Mov_r64_rm64
                if i.op0_register() == Register::RSP && i.op1_register() == Register::RBX =>
            {
                state.frame = None;
            }
            _ => {}
        }
        let next = encoded
            .offsets
            .get(k + 1)
            .copied()
            .unwrap_or(encoded.code.len() as u32);
        if ranges.last().unwrap().1 != state {
            ranges.push((next, state.clone()));
        }
    }
    let end = encoded.code.len() as u32;
    let mut data = table.data.lock().unwrap();
    let mut entries = Vec::new();
    for (n, (start, s)) in ranges.iter().enumerate() {
        let stop = ranges.get(n + 1).map_or(end, |r| r.0);
        if stop <= *start {
            continue;
        }
        let info = unwind_info(s, chain);
        let place = (data.0 + 3) & !3;
        if place + info.len() as u64 > data.1 {
            return;
        }
        // SAFETY: unused unwind data space of the area.
        unsafe { std::ptr::copy_nonoverlapping(info.as_ptr(), place as *mut u8, info.len()) };
        data.0 = place + info.len() as u64;
        let (Some(b), Some(u)) = (rva(at + u64::from(*start)), rva(place)) else {
            return;
        };
        entries.push(RuntimeFunction {
            begin: b,
            end: b + (stop - start),
            unwind: u,
        });
    }
    let n = table.len.load(Ordering::Relaxed);
    if n + entries.len() > ENTRIES {
        return;
    }
    // Entries are appended in address order (trampolines are allocated
    // upwards), written before the length that publishes them.
    let slots = table.entries.as_ptr() as *mut RuntimeFunction;
    for (k, e) in entries.iter().enumerate() {
        // SAFETY: past the published length; nobody reads these yet.
        unsafe { slots.add(n + k).write(*e) };
    }
    table.len.store(n + entries.len(), Ordering::Release);
}
