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
- **Marks.** A byte per 4 KiB part, per 64 KiB block, per 4 MiB and per
  GiB, at our view's address (one per page, whichever address the title
  wrote through). Taking them when there are none reads 64 bytes per 64 GiB
  slot holding our views. The order (store, part, block, 4 MiB, GiB mark;
  taken coarsest first, under a lock that "nothing written" waits for) means
  a mark missed by one taking is found by the next.
- **Parts.** A block that stayed guarded and was written only beside its
  guard by barriers (or the copies below) is told with the 4 KiB parts they
  marked, not whole (`watch::guard_parts`), so the journal holds those parts
  and a refresh copies them, as it already does for the parts the kernel's
  write watch names. A block `put_beside` wrote is told whole
  (`BESIDE_WHOLE`), as is one a fault opened.
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

| copies a block, threads | plain (fault, open, guard again) | `put_beside` | barriers' delta |
|---|---:|---:|---:|
| 1, 1 | 2.68 us a copy | 0.112 us | 0.098 us |
| 64, 1 | 0.080 us | 0.081 us | 0.049 us |
| 1, 8 | 6.40 us | 0.254 us | 0.187 us |
| 64, 8 | 0.163 us | 0.195 us | 0.166 us |

Then a host copy of each view is refreshed as a submission does
(`Tracked::take`): what it takes, and the time of both takes:

| copies a block | plain | `put_beside` | barriers |
|---|---:|---:|---:|
| 1 (256 bytes) | 8192 KiB, 54-59 us | 8192 KiB, 10-13 us | **512 KiB**, 24-27 us |
| 64 (16 KiB) | 8192 KiB, 62-63 us | 8192 KiB, 60-64 us | **2560 KiB**, 27 us |

- Against a fault, a copy beside the guard is 20-50 times cheaper, either
  way, and the block keeps its guard, so the asking guards nothing again.
- What a refresh copies (and sends up, and the card hashes again) is the
  parts written: here 16 times less for small copies, 3 times less for 16
  KiB ones. The askings cost more than `put_beside`'s (the journal keeps
  parts; ~14 us for 128 blocks here), which the bytes not copied pay for
  many times over.
- Against `put_beside`, the delta is somewhat cheaper per copy, and the
  difference grows with threads (no locks). With more than 32 copies a block
  between askings, `put_beside` opens the block (its share), and the copies
  and the asking cost what the plain way does; the delta has no share.
- The marks are turned into bits at every view on the thread that asks,
  where `put_beside` does it on the title's thread at the copy.

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

## Sites remembered between runs

Patching a site stops the process's other threads for a moment, so the
first minutes of a run, when most sites are met, would stutter. The sites
patched are written to `<cache>/barriers/sites.txt` every 10 s while they
change: a line each, the image's file name and PE time stamp, the site's
offset in it and a hash of its instruction (offsets and hashes, no code of
the title's). The next run patches them as soon as the image is scanned,
all under one stop of the threads (`Patcher::patch_many`); a line whose image
is another build, or whose instruction is no longer the one hashed, is left
out. A store the title only reaches later is patched at its first fault, as
before.

Title DLLs the title loads itself (`LoadLibrary`) are scanned and patched
like its first images, from their own remembered sites.

## What a first run should look at

- The `barriers` line: how many sites, and how many refused as `Int3Off`
  (with `VICTORY_BARRIER_INT3=1`: whether `int3_hits` grows fast; a hot
  `int3` site costs a trap per execution).
- Write faults a frame (the fault book, `VICTORY_FAULT_BOOK`) and the
  title threads' time in them, against a run without the switch.
- Frame times, and anything stale on screen.

## Not done

- Memory the kernel's write watch tracks (`XMemVirtualAlloc`'s private
  memory): no fault names its stores, and private memory cannot be mapped a
  second time, so a barrier has nowhere to store beside a guard. Made of
  sections the way `xmem` makes physical pages, it would be view memory, and
  the guards, our views, `kernel_writing` and barriers would take it as they
  are; only what copies take is guarded, so memory the card never reads does
  not fault. That would replace its askings (5.4 ms a frame on the
  submitting thread, 4.4 of it in the kernel). Open: a barrier costs every
  patched store into it some 25 instructions, where the write watch costs
  nothing (some 29 MB a frame are written there); a view cannot be
  decommitted (whether the title decommits is not known); Wine keeps the
  write watch; `VICTORY_INPLACE`'s private-memory heaps (off by default)
  would not open. A few days' work behind a switch of its own, worth doing
  once a run says what those cost.
- The title's own `VirtualProtect` of view memory: a block the title makes
  read-only keeps its delta, so a store it expects to fault goes beside.

## Tests

- `cargo test -p victory-barrier`: classification, the tables, a test that
  `int3` sites are refused by default, one of sites patched ahead under one
  stop, and one that maps a section twice, guards one view, lets stores in
  the test's own code fault, patches them, and checks bytes, marks and that
  they never fault again.
- `tests/difftest.rs`: every store form the classifier accepts, in this
  test's code and Windows DLLs, run as the original instruction and as its
  trampoline from the same random state; registers, flags, vector state,
  memory, faults and the marks (by `tables::take`, at 4 KiB parts) must
  agree, and three ways of breaking a trampoline on purpose must be seen. Any
  images, every instance: `VICTORY_BARRIER_DIFFTEST="a.exe;b.dll"
  VICTORY_BARRIER_DIFFTEST_PER_FORM=100000 cargo test -p victory-barrier
  --release --test difftest -- --ignored --nocapture`. Windows' DLLs and the
  prototype's stand-in title: 307 forms, all 117,862 instances, 471,368 paired
  runs (2,852 faulting alike), no difference. Its debug build found that a
  jump table's bound compared to a register was read as a constant (a table
  of one entry, its other targets missed); such a table is now read
  unbounded.
- `VICTORY_BARRIERS=1 cargo test -p victory-gpu --lib barrier::tests -- --ignored`
  (alone: it patches the test binary): the same through `watch`, `inplace`
  and `writes`, including a second view of the same pages, the 4 KiB part
  each store wrote, and the sites file.
- `copies_three_ways` (above), which also checks the bytes at both views and
  that both are told written, every frame, each way.
- `VICTORY_BARRIERS=1 cargo test -p victory-gpu --lib`: all of
  `victory-gpu`'s tests with barriers on (no patching: nothing starts the
  scan), as without.

The patcher was developed apart from VIctory, with a stand-in title, about
30 torture cases and a differential test that runs every store form against
its trampoline on the CPU (the `xbcompat` prototype).
