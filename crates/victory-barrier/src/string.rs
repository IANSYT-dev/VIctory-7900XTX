//! `movs` and `stos`, done for a patched site by host code.
//!
//! A `rep movsb` can write megabytes through one instruction, across slots
//! and across graphics and ordinary memory, so instead of an inline barrier
//! its trampoline saves the thread's state (general registers, flags, and
//! the vector state with `xsave`, since the copy below may use it) and calls
//! [`store`], which does what the instruction would have done: through our
//! view where the destination is guarded view memory, marking its blocks,
//! and updating `rdi`, `rsi` and `rcx` as the instruction would.

use crate::classify::StringKind;
use crate::layout::{BLOCK_SHIFT, Tables};
use iced_x86::Register;
use std::sync::OnceLock;

/// The barrier tables, set once by `Patcher::new`.
pub(crate) static TABLES: OnceLock<Tables> = OnceLock::new();

/// Sets the tables string trampolines use, for code that builds trampolines
/// without a `Patcher` (`xbc difftest`). The first call wins.
pub fn set_tables(tables: Tables) {
    let _ = TABLES.set(tables);
}

/// What a string site does: one per patched site, never freed.
#[repr(C)]
pub struct StringSite {
    pub kind: StringKind,
    pub element: u8,
    pub rep: bool,
}

/// The order the trampoline pushes registers in (after `pushfq`). The
/// saved state on the stack therefore reads, from its lowest address,
/// in the reverse order: see [`Saved`].
pub(crate) const SAVED_PUSH_ORDER: [Register; 15] = [
    Register::RAX,
    Register::RCX,
    Register::RDX,
    Register::RBX,
    Register::RBP,
    Register::RSI,
    Register::RDI,
    Register::R8,
    Register::R9,
    Register::R10,
    Register::R11,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

/// The thread's registers as the trampoline left them on its stack.
#[repr(C)]
pub struct Saved {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rbp: u64,
    pub rbx: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rax: u64,
    pub rflags: u64,
}

const DF: u64 = 1 << 10;

/// Bytes the trampoline reserves for `xsave`: the size this CPU and OS use
/// for everything they enable, plus room to align to 64.
pub fn xsave_area_bytes() -> u32 {
    static BYTES: OnceLock<u32> = OnceLock::new();
    *BYTES.get_or_init(|| {
        // Leaf 0xD exists wherever XSAVE does, which the patcher checks
        // before it patches.
        let size = std::arch::x86_64::__cpuid_count(0xD, 0).ebx;
        (size + 64 + 63) & !63
    })
}

const BLOCK: u64 = 1 << BLOCK_SHIFT;

/// `[from, to)` cut where its 64 KiB blocks end: each piece has one delta.
fn pieces(from: u64, to: u64) -> impl Iterator<Item = (u64, u64)> {
    let mut at = from;
    std::iter::from_fn(move || {
        (at < to).then(|| {
            let end = ((at & !(BLOCK - 1)) + BLOCK).min(to);
            let piece = (at, end);
            at = end;
            piece
        })
    })
}

/// Marks where the bytes of `[from, to)` went, after they were written: a
/// mark per block, at our address of it.
fn mark(tables: &Tables, from: u64, to: u64) {
    for (at, _) in pieces(from, to) {
        tables.mark(tables.resolve(at));
    }
}

/// Where the bytes `[from, to)` are written: through one run of addresses,
/// if each block's goes on from the one before (one view, as a range of the
/// title's in one section is).
fn writer(tables: &Tables, from: u64, to: u64) -> Option<u64> {
    let base = tables.resolve(from);
    pieces(from, to)
        .all(|(at, _)| tables.resolve(at) == base.wrapping_add(at - from))
        .then_some(base)
}

/// Called by a string trampoline.
///
/// # Safety
///
/// Called only from a trampoline, with the state it saved.
pub(crate) unsafe extern "C" fn store(saved: &mut Saved, site: &StringSite) {
    let tables = TABLES.get().expect("patcher tables");
    let element = u64::from(site.element);
    let count = if site.rep { saved.rcx } else { 1 };
    if count == 0 {
        return;
    }
    let bytes = count * element;
    let down = saved.rflags & DF != 0;
    // The lowest address written and one past the highest.
    let (from, to) = if down {
        (saved.rdi + element - bytes, saved.rdi + element)
    } else {
        (saved.rdi, saved.rdi + bytes)
    };
    match site.kind {
        StringKind::Stos => {
            // Every element holds the same value, so byte `k` of the range
            // is byte `k % element` of it, whichever way the copy runs.
            let value = saved.rax.to_le_bytes();
            for (at, end) in pieces(from, to) {
                let w = tables.resolve(at);
                if element == 1 {
                    // SAFETY: the title's instruction would have written here.
                    unsafe { std::ptr::write_bytes(w as *mut u8, value[0], (end - at) as usize) }
                } else {
                    for k in 0..end - at {
                        // SAFETY: as above, a byte at a time.
                        unsafe {
                            std::ptr::write_volatile(
                                (w + k) as *mut u8,
                                value[((at + k - from) % element) as usize],
                            )
                        }
                    }
                }
            }
        }
        StringKind::Movs => {
            let src_from = if down {
                saved.rsi + element - bytes
            } else {
                saved.rsi
            };
            // Both ends as the barrier sees them: view memory through our
            // view. Overlap has to be judged there, not between a title
            // address and ours of the same bytes, which a memmove would
            // take for two separate buffers.
            let (dst_w, src_w) = (
                writer(tables, from, to),
                writer(tables, src_from, src_from + bytes),
            );
            let overlap = match (dst_w, src_w) {
                (Some(d), Some(s)) => s < d + bytes && d < s + bytes,
                _ => true,
            };
            match (dst_w, src_w) {
                (Some(d), Some(s)) if !overlap => {
                    // SAFETY: the instruction's own source and destination.
                    unsafe {
                        std::ptr::copy_nonoverlapping(s as *const u8, d as *mut u8, bytes as usize)
                    }
                }
                _ => {
                    // Element by element in the instruction's own order: a
                    // memmove when the copy runs away from the overlap, a
                    // replication when it runs into it, as the CPU does.
                    for k in 0..count {
                        let k = if down { count - 1 - k } else { k };
                        for b in 0..element {
                            let off = k * element + b;
                            // SAFETY: as above, one byte at a time.
                            unsafe {
                                let byte = std::ptr::read_volatile(
                                    tables.resolve(src_from + off) as *const u8
                                );
                                std::ptr::write_volatile(
                                    tables.resolve(from + off) as *mut u8,
                                    byte,
                                );
                            }
                        }
                    }
                }
            }
            saved.rsi = if down {
                saved.rsi.wrapping_sub(bytes)
            } else {
                saved.rsi + bytes
            };
        }
    }
    mark(tables, from, to);
    saved.rdi = if down {
        saved.rdi.wrapping_sub(bytes)
    } else {
        saved.rdi + bytes
    };
    if site.rep {
        saved.rcx = 0;
    }
}
