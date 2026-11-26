//! The tables a write barrier reads, and how it indexes them.
//!
//! A barrier has to run between two instructions of the title's own code,
//! so it may not change the flags (the title can test flags a store sits
//! between: `cmp`, store, `jne`) and may not branch on anything it computes.
//! It therefore only uses `lea`, `mov`, `shrx` and `shlx`, and finds both
//! the address to store at and the bytes to mark by table lookups that need
//! no range check:
//!
//! ```text
//! slot   = ea >> 36                      (64 GiB slots; user space has 2048)
//! blocks = DELTA[slot]                   one u64 per 64 KiB block of the slot
//! delta  = blocks[(ea << 28) >> 44]      0, or our view's address - the title's
//! w      = ea + delta                    where the store goes
//! PART[w's slot][(w << 28) >> 40]   = 1  a byte per 4 KiB part
//! FINE[w's slot][(w << 28) >> 44]   = 1  a byte per 64 KiB block
//! MID[w's slot][(w << 28) >> 50]    = 1  a byte per 4 MiB
//! COARSE[w's slot][(w << 28) >> 58] = 1  a byte per GiB
//! ```
//!
//! A block of the title's view memory that a write faulted on gets the delta
//! that sends its stores to our own view of the same page, which no guard
//! covers (`victory-gpu`'s `inplace::beside`). Everything else has delta 0,
//! so a patched instruction that also writes ordinary memory (a `memcpy`
//! loop does both) stores where it always did, and a store the title expects
//! to fault still faults.
//!
//! The marks go by `w`, our address of the page, not the title's: the title
//! shows one page at several addresses, and a mark at the one address that
//! is the page's own tells of a write through any of them. Slots with none
//! of our views mark a shared junk map.
//!
//! Four levels: the parts say which 4 KiB of a block were written (a
//! refresh copies those, not the block), and the three above them make
//! finding the marks when there are none (which an asking does every time)
//! read 64 bytes a slot, and when there are some read only under the marked
//! GiB, 4 MiB and block. A barrier marks the finest first (store, part,
//! fine, mid, coarse); who takes marks takes the coarsest first and reads
//! the bytes after (coarse, mid, fine, parts, bytes), so a mark it misses was
//! made after it took that level, and is found next time (the argument for
//! one level, at each level: see `tables::take`). A block whose fine mark is
//! found with no part marked was written before an earlier taking took its
//! parts: nothing is left to take of it.

/// Bits of an address below its slot index.
pub const SLOT_SHIFT: u32 = 36;
/// Slots covering the 47-bit user address space.
pub const SLOTS: usize = 1 << (47 - SLOT_SHIFT);
/// The part marks' unit: 4 KiB, the host's page.
pub const PART_SHIFT: u32 = 12;
/// The delta's unit and the fine marks': 64 KiB, the console's page.
pub const BLOCK_SHIFT: u32 = 16;
/// Parts of a block.
pub const PARTS_PER_BLOCK: usize = 1 << (BLOCK_SHIFT - PART_SHIFT);
/// The middle marks' unit: 4 MiB.
pub const MID_SHIFT: u32 = 22;
/// The coarse marks' unit: 1 GiB.
pub const COARSE_SHIFT: u32 = 30;
/// Part marks per slot.
pub const PARTS_PER_SLOT: usize = 1 << (SLOT_SHIFT - PART_SHIFT);
/// Deltas, and fine marks, per slot.
pub const BLOCKS_PER_SLOT: usize = 1 << (SLOT_SHIFT - BLOCK_SHIFT);
pub const MIDS_PER_SLOT: usize = 1 << (SLOT_SHIFT - MID_SHIFT);
pub const COARSES_PER_SLOT: usize = 1 << (SLOT_SHIFT - COARSE_SHIFT);
/// `(ea << INDEX_SHL) >> X_SHR` is the index of `ea`'s unit within its slot.
pub const INDEX_SHL: u32 = 64 - SLOT_SHIFT;
pub const PART_SHR: u32 = INDEX_SHL + PART_SHIFT;
pub const BLOCK_SHR: u32 = INDEX_SHL + BLOCK_SHIFT;
pub const MID_SHR: u32 = INDEX_SHL + MID_SHIFT;
pub const COARSE_SHR: u32 = INDEX_SHL + COARSE_SHIFT;
/// The largest store an inline barrier marks by its first and last byte:
/// at most two units of every level (the classifier takes none larger).
pub const INLINE_STORE_MAX: u32 = 1 << PART_SHIFT;

/// Where the barrier's tables are. Each holds `SLOTS` u64s, the address of
/// the slot's array: `delta` its per-block deltas, the others its marks.
#[derive(Clone, Copy, Debug)]
pub struct Tables {
    pub delta: *const u64,
    pub part: *const u64,
    pub fine: *const u64,
    pub mid: *const u64,
    pub coarse: *const u64,
}

// SAFETY: the slot tables' entries are written atomically, each before
// anything can need them; the per-block deltas are changed atomically, a
// u64 at a time.
unsafe impl Send for Tables {}
unsafe impl Sync for Tables {}

#[inline]
pub(crate) fn index(ea: u64, shr: u32) -> usize {
    ((ea << INDEX_SHL) >> shr) as usize
}

#[inline]
pub(crate) fn slot(ea: u64) -> usize {
    (ea >> SLOT_SHIFT) as usize & (SLOTS - 1)
}

impl Tables {
    /// Where a store at `ea` goes, as the barrier computes it.
    #[inline]
    pub fn resolve(&self, ea: u64) -> u64 {
        // SAFETY: the table holds SLOTS entries; each slot's array holds
        // BLOCKS_PER_SLOT.
        unsafe {
            let blocks = std::ptr::read_volatile(self.delta.add(slot(ea))) as *const u64;
            ea.wrapping_add(std::ptr::read_volatile(blocks.add(index(ea, BLOCK_SHR))))
        }
    }

    /// Marks every part of `[w, w + len)` (stores went there, inside one
    /// block of ours), as barriers mark.
    pub fn mark_range(&self, w: u64, len: u64) {
        if len == 0 {
            return;
        }
        let mut at = w;
        while at < w + len {
            self.mark(at);
            at = (at | ((1 << PART_SHIFT) - 1)) + 1;
        }
    }

    /// Marks `w` (an address stores went to) as a barrier does, the finest
    /// level first.
    #[inline]
    pub fn mark(&self, w: u64) {
        use std::sync::atomic::{AtomicU8, Ordering};
        for (table, shr) in [
            (self.part, PART_SHR),
            (self.fine, BLOCK_SHR),
            (self.mid, MID_SHR),
            (self.coarse, COARSE_SHR),
        ] {
            // SAFETY: as above; a slot's mark arrays hold every index.
            unsafe {
                let map = std::ptr::read_volatile(table.add(slot(w))) as *mut u8;
                AtomicU8::from_ptr(map.add(index(w, shr))).store(1, Ordering::Release);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_are_within_the_slot() {
        let slot_base = 0x6000_0000_0000u64;
        assert_eq!(slot(slot_base), 0x600);
        for block in [0u64, 1, 15, 16, (BLOCKS_PER_SLOT - 1) as u64] {
            let ea = slot_base + (block << BLOCK_SHIFT) + 123;
            assert_eq!(index(ea, BLOCK_SHR), block as usize);
            assert_eq!(
                index(ea, MID_SHR),
                (block >> (MID_SHIFT - BLOCK_SHIFT)) as usize
            );
            assert_eq!(
                index(ea, COARSE_SHR),
                (block >> (COARSE_SHIFT - BLOCK_SHIFT)) as usize
            );
            assert!(index(ea, COARSE_SHR) < COARSES_PER_SLOT && index(ea, MID_SHR) < MIDS_PER_SLOT);
        }
    }

    /// Marking a store's first and last byte marks every unit of every level
    /// it writes, for stores up to `INLINE_STORE_MAX`, at every offset.
    #[test]
    fn first_and_last_cover_the_store() {
        let base = 0x6000_0000_0000u64 + (5 << BLOCK_SHIFT);
        for shr in [PART_SHR, BLOCK_SHR, MID_SHR, COARSE_SHR] {
            for ea in (base - 70_000..base + 70_000).step_by(997) {
                for size in [1u64, 2, 8, 64, 2696, 4095, u64::from(INLINE_STORE_MAX)] {
                    let (a, b) = (index(ea, shr), index(ea + size - 1, shr));
                    let units = (ea..ea + size).step_by(61).chain([ea + size - 1]);
                    assert!(
                        units.map(|x| index(x, shr)).all(|u| u == a || u == b),
                        "{ea:#x}+{size}"
                    );
                }
            }
        }
    }
}
