//! The barrier's tables (`layout`), made and kept here for the process.
//!
//! Every slot starts out with a shared table of zero deltas (read-only) and
//! shared junk marks, so a barrier can index any address. A slot gets its
//! own deltas when a block in it is given one ([`set_delta`]), and its own
//! marks when one of our views lies in it ([`track`]). Nothing is ever
//! given back: a slot's deltas are 8 MiB of address space, its marks 1 MiB,
//! and only the pages written come to be.
//!
//! [`take`] finds and clears the marks. Who asks whether memory was written
//! takes them first, and must not answer "nothing" while another thread is
//! between clearing a mark and telling of it: so taking is done under a
//! lock, and the check for "no marks at all" that skips the lock also looks
//! at whether a taking is under way.

use crate::layout::{
    BLOCK_SHIFT, BLOCK_SHR, BLOCKS_PER_SLOT, COARSE_SHIFT, COARSES_PER_SLOT, MID_SHIFT, MIDS_PER_SLOT,
    SLOT_SHIFT, SLOTS, Tables, index, slot,
};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

static DELTA: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static FINE: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static MID: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static COARSE: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
/// Slots with marks of their own, a bit each.
static TRACKED: [AtomicU64; SLOTS / 64] = [const { AtomicU64::new(0) }; SLOTS / 64];
/// The shared arrays: (zero deltas, junk fine, junk mid, junk coarse).
static SHARED: OnceLock<[u64; 4]> = OnceLock::new();
/// Serialises making a slot's arrays, and taking marks.
static MAKING: Mutex<()> = Mutex::new(());
static TAKING: Mutex<()> = Mutex::new(());
/// Takings between clearing a mark and having told of it.
static UNDER_WAY: AtomicUsize = AtomicUsize::new(0);

fn alloc(bytes: usize, read_only: bool) -> Option<u64> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc,
    };
    // SAFETY: fresh memory of ours, zeroed by Windows.
    let at = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            bytes,
            MEM_COMMIT | MEM_RESERVE,
            if read_only {
                PAGE_READONLY
            } else {
                PAGE_READWRITE
            },
        )
    } as u64;
    (at != 0).then_some(at)
}

/// The tables, made on first use. `None` if Windows gave no memory.
pub fn tables() -> Option<Tables> {
    let shared = match SHARED.get() {
        Some(s) => *s,
        None => {
            let _making = MAKING.lock().unwrap_or_else(|e| e.into_inner());
            match SHARED.get() {
                Some(s) => *s,
                None => {
                    let s = [
                        alloc(BLOCKS_PER_SLOT * 8, true)?,
                        alloc(BLOCKS_PER_SLOT, false)?,
                        alloc(MIDS_PER_SLOT, false)?,
                        alloc(COARSES_PER_SLOT.max(64), false)?,
                    ];
                    for k in 0..SLOTS {
                        DELTA[k].store(s[0], Ordering::Release);
                        FINE[k].store(s[1], Ordering::Release);
                        MID[k].store(s[2], Ordering::Release);
                        COARSE[k].store(s[3], Ordering::Release);
                    }
                    let _ = SHARED.set(s);
                    s
                }
            }
        }
    };
    let _ = shared;
    Some(Tables {
        delta: DELTA.as_ptr().cast(),
        fine: FINE.as_ptr().cast(),
        mid: MID.as_ptr().cast(),
        coarse: COARSE.as_ptr().cast(),
    })
}

/// The slot's own deltas, made if need be.
fn own_deltas(s: usize) -> Option<*const AtomicU64> {
    let shared = SHARED.get()?[0];
    let now = DELTA[s].load(Ordering::Acquire);
    if now != shared {
        return Some(now as *const AtomicU64);
    }
    let _making = MAKING.lock().unwrap_or_else(|e| e.into_inner());
    let now = DELTA[s].load(Ordering::Acquire);
    if now != shared {
        return Some(now as *const AtomicU64);
    }
    let made = alloc(BLOCKS_PER_SLOT * 8, false)?;
    DELTA[s].store(made, Ordering::Release);
    Some(made as *const AtomicU64)
}

/// Stores into the 64 KiB block at `block` go to `block + delta` from now on.
/// `false` if no table could be made.
pub fn set_delta(block: u64, delta: u64) -> bool {
    if tables().is_none() {
        return false;
    }
    let Some(deltas) = own_deltas(slot(block)) else {
        return false;
    };
    // SAFETY: the slot's array holds every block's delta.
    unsafe { (*deltas.add(index(block, BLOCK_SHR))).store(delta, Ordering::SeqCst) };
    true
}

/// The delta of the block at `at`.
pub fn delta(at: u64) -> u64 {
    tables().map_or(0, |t| t.resolve(at).wrapping_sub(at))
}

/// Stores into `[from, to)` go where the title meant from now on. Any
/// range: what lies past user space (where no delta can be) is left out, and
/// only slots with deltas of their own are looked at.
pub fn clear(from: u64, to: u64) {
    /// The end of the 47-bit user address space.
    const USER_END: u64 = 1 << 47;
    let Some(shared) = SHARED.get().map(|s| s[0]) else {
        return;
    };
    let (from, to) = (from.min(USER_END), to.min(USER_END));
    if from >= to {
        return;
    }
    for s in (from >> SLOT_SHIFT)..=((to - 1) >> SLOT_SHIFT) {
        let deltas = DELTA[s as usize].load(Ordering::Acquire);
        if deltas == shared {
            continue;
        }
        let deltas = deltas as *const AtomicU64;
        let base = s << SLOT_SHIFT;
        let (lo, hi) = (from.max(base), to.min(base + (1 << SLOT_SHIFT)));
        for b in index(lo, BLOCK_SHR)..=index(hi - 1, BLOCK_SHR) {
            // SAFETY: as in `set_delta`.
            let delta = unsafe { &*deltas.add(b) };
            // (Not written where it is 0: a page of the table never given a
            // delta stays untouched.)
            if delta.load(Ordering::Relaxed) != 0 {
                delta.store(0, Ordering::SeqCst);
            }
        }
    }
}

/// Our view `[from, from + len)` takes stores: its slots get marks of their
/// own. `false` if no table could be made.
pub fn track(from: u64, len: u64) -> bool {
    let Some(shared) = tables().and_then(|_| SHARED.get().copied()) else {
        return false;
    };
    let _making = MAKING.lock().unwrap_or_else(|e| e.into_inner());
    let (first, last) = (from >> SLOT_SHIFT, (from + len.max(1) - 1) >> SLOT_SHIFT);
    for s in first..=last {
        let s = s as usize & (SLOTS - 1);
        if FINE[s].load(Ordering::Acquire) != shared[1] {
            continue;
        }
        let (Some(fine), Some(mid), Some(coarse)) = (
            alloc(BLOCKS_PER_SLOT, false),
            alloc(MIDS_PER_SLOT, false),
            alloc(COARSES_PER_SLOT.max(64), false),
        ) else {
            return false;
        };
        // The finest first: a barrier that sees the slot's coarse array
        // marks the others of the same slot.
        FINE[s].store(fine, Ordering::Release);
        MID[s].store(mid, Ordering::Release);
        COARSE[s].store(coarse, Ordering::Release);
        TRACKED[s / 64].fetch_or(1 << (s % 64), Ordering::SeqCst);
    }
    true
}

fn tracked() -> impl Iterator<Item = usize> {
    (0..SLOTS / 64).flat_map(|w| {
        let mut bits = TRACKED[w].load(Ordering::SeqCst);
        std::iter::from_fn(move || {
            (bits != 0).then(|| {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                w * 64 + b
            })
        })
    })
}

fn bytes<'a>(at: u64, n: usize) -> &'a [AtomicU8] {
    // SAFETY: a mark array of ours, of at least `n` bytes, never freed.
    unsafe { std::slice::from_raw_parts(at as *const AtomicU8, n) }
}

/// Whether a mark is set anywhere, or a taking is under way.
fn any() -> bool {
    tracked().any(|s| {
        bytes(COARSE[s].load(Ordering::Acquire), COARSES_PER_SLOT)
            .iter()
            .any(|b| b.load(Ordering::SeqCst) != 0)
    }) || UNDER_WAY.load(Ordering::SeqCst) != 0
}

/// Takes every mark: `wrote` is told the addresses of the 64 KiB blocks of
/// our views that stores went to since (if any), and returns before any
/// other taking can answer that nothing was written. Who reads a block's
/// bytes after this returns finds every store whose mark was taken.
pub fn take(wrote: impl FnOnce(&[u64])) {
    if !any() {
        return;
    }
    let _taking = TAKING.lock().unwrap_or_else(|e| e.into_inner());
    UNDER_WAY.fetch_add(1, Ordering::SeqCst);
    let mut blocks = Vec::new();
    for s in tracked() {
        let slot_base = (s as u64) << SLOT_SHIFT;
        let (fine, mid, coarse) = (
            FINE[s].load(Ordering::Acquire),
            MID[s].load(Ordering::Acquire),
            COARSE[s].load(Ordering::Acquire),
        );
        for (c, cb) in bytes(coarse, COARSES_PER_SLOT).iter().enumerate() {
            if cb.load(Ordering::SeqCst) == 0 || cb.swap(0, Ordering::SeqCst) == 0 {
                continue;
            }
            let mids_per = 1 << (COARSE_SHIFT - MID_SHIFT);
            for m in c * mids_per..(c + 1) * mids_per {
                let mb = &bytes(mid, MIDS_PER_SLOT)[m];
                if mb.load(Ordering::SeqCst) == 0 || mb.swap(0, Ordering::SeqCst) == 0 {
                    continue;
                }
                let fines_per = 1 << (MID_SHIFT - BLOCK_SHIFT);
                for f in m * fines_per..(m + 1) * fines_per {
                    let fb = &bytes(fine, BLOCKS_PER_SLOT)[f];
                    if fb.load(Ordering::SeqCst) != 0 && fb.swap(0, Ordering::SeqCst) != 0 {
                        blocks.push(slot_base + ((f as u64) << BLOCK_SHIFT));
                    }
                }
            }
        }
    }
    if !blocks.is_empty() {
        wrote(&blocks);
    }
    UNDER_WAY.fetch_sub(1, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_are_taken_once_by_block() {
        // An address range of ours to stand for a view.
        let view = alloc(4 << 20, false).unwrap();
        assert!(track(view, 4 << 20));
        let t = tables().unwrap();
        t.mark(view + 5);
        t.mark(view + (3 << 16) + 100);
        t.mark(view + (3 << 16) + 200);
        let mut got = Vec::new();
        take(|b| got.extend_from_slice(b));
        assert_eq!(got, vec![view, view + (3 << 16)]);
        let mut again = Vec::new();
        take(|b| again.extend_from_slice(b));
        assert!(again.is_empty());
    }

    #[test]
    fn clearing_any_range_ends() {
        let at = 0x7123_4570_0000u64;
        assert!(set_delta(at, 0x1000_0000));
        clear(u64::MAX - 5, u64::MAX);
        clear(0, u64::MAX);
        assert_eq!(delta(at), 0);
        assert!(set_delta(at, 0x1000_0000));
        clear(at - (1 << 40), at + 1);
        assert_eq!(delta(at), 0);
    }

    #[test]
    fn deltas_resolve_and_clear() {
        let t = tables().unwrap();
        let at = 0x7123_4560_0000u64;
        assert_eq!(t.resolve(at + 7), at + 7);
        assert!(set_delta(at, 0x1000_0000));
        assert_eq!(t.resolve(at + 7), at + 7 + 0x1000_0000);
        assert_eq!(t.resolve(at + (1 << 16)), at + (1 << 16));
        clear(at, at + 1);
        assert_eq!(t.resolve(at + 7), at + 7);
    }
}
