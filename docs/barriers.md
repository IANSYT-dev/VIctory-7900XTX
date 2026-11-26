# Write barriers in the title's code

`VICTORY_BARRIERS=1` (off by default). **Not yet run with the title**: what
is below is what the code does and what its tests show, not a measurement
of the game.

## What it does

View memory (the title's physical-page views) is guarded in 64 KiB blocks
(`crate::watch` in `victory-gpu`). Every first write to a guarded block
faults; the handler opens the block, or with `VICTORY_STORE_BESIDE=1` does
the store through our own view of the page (`titlewrite`), at every fault.

With barriers on, the first fault at an instruction also **patches that
instruction** (crate `victory-barrier`): it jumps to a trampoline that does
the same store through our view of the page (`inplace::beside`) and marks
the block. From then on that instruction never faults, the block stays
guarded and clean, and the next asking takes the marks as it takes
`put_beside`'s bits. It is `titlewrite`'s store beside the guard, decided
once per instruction instead of at every fault, and without a decode in the
handler.

How the instruction reaches its trampoline, in order of preference:

1. a `jmp rel32` over the store, when it is 5 bytes or more;
2. a `jmp rel8` to 5 bytes of `int3` padding nearby, which hold the `jmp`;
3. the instructions after the store moved into the trampoline too, only
   where the scan of the title's code proves nothing branches into them
   (every indirect jump of the function read, its exception handler
   understood, or a leaf function bounded by padding);
4. the same at the start of the store's basic block, when a branch target
   follows the store;
5. one `int3`, which a breakpoint handler sends to the trampoline: only
   with `VICTORY_BARRIER_INT3=1`. An `int3` traps at every execution, where a
   guard faults once a block per refresh, so a hot short store can cost more
   on `int3` than it did before; by default such a store is refused
   (`Int3Off`) and faults as it always did.

Other threads stopped inside code being moved are moved to its copy.
Trampolines carry unwind data, so stack walks and C++ exceptions pass
through them. `rep movs`/`rep stos` go to a helper that copies block by
block.

## How it fits

| Where | What |
|---|---|
| `crates/victory-barrier` | the patcher: store classification, the scan of the title's code, trampolines, unwind data, thread freezing, the tables |
| `victory-gpu/src/barrier.rs` | the switch, the scan's thread, the fault hook, deltas cleared on remapping, marks to views |
| `watch.rs`, `on_write` | asks `barrier::on_fault` before `titlewrite::store` |
| `watch.rs`, `published` and `guard` | take the marks first (`barrier_marks`): `BESIDE` set and `PUBLISHED` cleared at every view of each page written, as `put_beside` does |
| `watch.rs`, `title_memory_changed` and `title_memory_freed` | clear the deltas of the range first |
| `inplace.rs`, `views_of_beside` | our view's page back to every title address that shows it |
| `apps/victory/src/run.rs` | starts the scan after VIctory's own code patches |

- **Deltas.** A barrier finds where to store by two table lookups (no
  branch, no flag changed): a block's delta is our view's address minus the
  title's, set at the block's first write fault, 0 elsewhere. A store with
  delta 0 goes where the title meant and faults as before.
- **Marks.** A byte per 64 KiB block, per 4 MiB and per GiB, at our view's
  address (one per page, whichever address the title wrote through). Taking
  them when there are none reads 64 bytes per 64 GiB slot holding our views.
  The order (store, fine, middle, coarse mark; taken coarse first, under a
  lock that "nothing written" waits for) means a mark missed by one taking is
  found by the next.
- **Other patches.** A function with a jump out of the image (VIctory's
  `write_sites`, `descriptor_sites` stubs) is never moved, and what follows
  such a jump counts as a branch target.
- **`clzero`** is zeroed with plain stores (it is AMD's only; #UD on Intel).

## The title's `memcpy`, `memmove` and `memset`

These are imports the host serves, so no barrier is ever patched into them,
and by `titlewrite`'s count they were a third of the write faults by time.
With barriers on, the host binds them to `titlewrite`'s wrappers (as with
`VICTORY_COPY_BESIDE=1`), and a piece of a copy into a guarded block goes
beside the guard by the block's delta (given to the block there if it has
none), then marks it: one table read and three byte stores, no lock, no
share per block. Without a delta it goes the way it did (`put_beside` with
`VICTORY_COPY_BESIDE=1`, else a fault).

`VICTORY_BARRIERS=1 cargo test -p victory-gpu --release --lib copies_three_ways -- --ignored --nocapture`
copies 256 bytes into each of 64 guarded blocks of a section shown twice, a
frame at a time, then asks about both views (Ryzen 5 7600, Windows 10; one
run, the copies' time summed over the threads):

| copies a block, threads | plain (fault, open, guard again) | `put_beside` | barriers' delta | askings: plain / `put_beside` / barriers |
|---|---:|---:|---:|---|
| 1, 1 | 2.95 us a copy | 0.124 us | 0.088 us | 50 / 4 / 9 us a frame |
| 64, 1 | 0.086 us | 0.079 us | 0.045 us | 56 / 56 / 7 us |
| 1, 8 | 7.65 us | 0.270 us | 0.166 us | 51 / 4 / 7 us |
| 64, 8 | 0.171 us | 0.178 us | 0.100 us | 51 / 52 / 7 us |

- Against a fault, a copy beside the guard is 20-50 times cheaper, either
  way, and the block keeps its guard, so the asking guards nothing again.
- Against `put_beside`, the delta is somewhat cheaper per copy, and the
  difference grows with threads (no locks). With more than 32 copies a block
  between askings, `put_beside` opens the block (its share), and the copies
  and the asking cost what the plain way does; the delta has no share.
- With one copy a block, the asking costs ~5 us more (64 blocks): the marks
  are turned into bits at every view there, on the thread that asks, where
  `put_beside` does it on the title's thread at the copy. About 60 ns a
  block written, on the thread that prepares submissions.

## Switches

| Switch | |
|---|---|
| `VICTORY_BARRIERS=1` | on (needs BMI2, XSAVE, and `VICTORY_LIGHT_REFRESH`/`VICTORY_VIEW_GUARD` not 0) |
| `VICTORY_BARRIER_INT3=1` | also patch stores only an `int3` can reach (see above) |
| `VICTORY_BARRIER_RELOCATE=1` | also move code the scan did not prove safe (a computed jump into it crashes the title) |
| `VICTORY_BARRIER_DEBUG=1` | log why each `int3` site is one |
| `VICTORY_BARRIER_NO_UNWIND=1` | trampolines without unwind data (diagnostics only) |

Every 10 s while it changes, a `barriers` line: sites patched by strategy,
refusals by reason, `int3` hits, threads moved, time spent patching, faults
seen and blocks given a delta.

## What a first run should look at

- The `barriers` line: how many sites, and how many refused as `Int3Off`
  (with `VICTORY_BARRIER_INT3=1`: whether `int3_hits` grows fast; a hot
  `int3` site costs a trap per execution).
- Write faults a frame (the fault book, `VICTORY_FAULT_BOOK`) and the
  title threads' time in them, against a run without the switch.
- Frame times, and anything stale on screen.

## Not done

- Memory the kernel's write watch tracks: no fault names its stores.
- The title's own `VirtualProtect` of view memory: a block the title makes
  read-only keeps its delta, so a store it expects to fault goes beside.
- Title DLLs loaded after the start are not patched.

## Tests

- `cargo test -p victory-barrier`: classification, the tables, a test that
  `int3` sites are refused by default, and a test
  that maps a section twice, guards one view, lets stores in the test's own
  code fault, patches them, and checks bytes, marks and that they never
  fault again.
- `VICTORY_BARRIERS=1 cargo test -p victory-gpu --lib barrier::tests -- --ignored`
  (alone: it patches the test binary): the same through `watch`, `inplace`
  and `writes`, including a second view of the same pages.
- `copies_three_ways` (above), which also checks the bytes at both views and
  that both are told written, every frame, each way.
- `VICTORY_BARRIERS=1 cargo test -p victory-gpu --lib`: all of
  `victory-gpu`'s tests with barriers on (no patching: nothing starts the
  scan), as without.

The patcher was developed apart from VIctory, with a stand-in title, about
30 torture cases and a differential test that runs every store form against
its trampoline on the CPU (the `xbcompat` prototype).
