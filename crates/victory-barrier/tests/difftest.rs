//! Every store form the classifier accepts, run twice from the same random
//! state on this CPU: once as the original instruction, once as the
//! trampoline the patcher builds for it. The CPU is the oracle: whatever the
//! original does, the trampoline must do too.
//!
//! Compared after each pair of runs:
//! - all general registers (rsp aside), the arithmetic flags and DF;
//! - the vector state (x87, SSE, AVX, AVX-512 masks and registers), by XSAVE;
//! - the memory the store can reach (a 64 KiB window of a test arena mapped
//!   twice, the title's view and ours, at the distance a guarded block's
//!   delta gives);
//! - that every byte the store changed is covered by a mark of its 4 KiB
//!   part at our view's address, taken as an asking takes them
//!   (`tables::take`): what the watch relies on;
//! - faults: an original that faults (misaligned `movaps`, `#GP` from a
//!   privileged form, `#UD`) must fault the same way as its trampoline.
//!
//! The corpus is real code: each instruction the classifier accepts in the
//! images given, grouped by form (code and addressing shape), a few instances
//! of each, re-encoded into a buffer near the arena with its address
//! registers solved so the store lands in the arena, everything else random.
//! The sabotage runs break the trampolines on purpose (marks to the junk
//! maps, flags changed, the store skipped) to show the comparison sees each.
//!
//! `cargo test -p victory-barrier --test difftest` runs this test binary's
//! own code and a few Windows DLLs. Any images, every instance:
//! `VICTORY_BARRIER_DIFFTEST="a.exe;b.dll" VICTORY_BARRIER_DIFFTEST_PER_FORM=100000
//! cargo test -p victory-barrier --release --test difftest -- --ignored --nocapture`.

#![cfg(windows)]

use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, Decoder, DecoderOptions, Formatter, Instruction,
    InstructionBlock, InstructionInfoFactory, IntelFormatter, MemoryOperand, Mnemonic, Register,
};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use victory_barrier::classify::{self, Address, Store};
use victory_barrier::layout::{SLOT_SHIFT, SLOTS, Tables};
use victory_barrier::string::{self, StringSite};
use victory_barrier::text::Text;
use victory_barrier::{emit, tables};
use windows_sys::Win32::Foundation::{EXCEPTION_BREAKPOINT, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_POINTERS,
    FlushInstructionCache,
};
use windows_sys::Win32::System::LibraryLoader::{DONT_RESOLVE_DLL_REFERENCES, LoadLibraryExW};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_COMMIT, MEM_RESERVE, MapViewOfFile3, PAGE_EXECUTE_READWRITE,
    PAGE_READWRITE, SEC_COMMIT, VirtualAlloc,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const BLOCK: u64 = 1 << 16;

const TITLE_SLOT: u64 = 0x600;
const WRITER_DELTA: u64 = 1 << SLOT_SHIFT;
/// The test arena: 1 MiB at 1 MiB into the title slot.
const ARENA: u64 = (TITLE_SLOT << SLOT_SHIFT) + (1 << 20);
const ARENA_BYTES: usize = 1 << 20;
/// Stores land around the arena's middle; this much around it is compared.
const WINDOW: u64 = ARENA + (ARENA_BYTES as u64) / 2 - 32 * 1024;
const WINDOW_BYTES: usize = 64 * 1024;
const XSAVE: usize = 4096;
const GPR: usize = XSAVE;
const FLAGS: usize = XSAVE + 16 * 8;

/// What a run starts from or ends with.
#[repr(C, align(64))]
#[derive(Clone)]
struct State {
    xsave: [u8; XSAVE],
    gpr: [u64; 16],
    flags: u64,
}

const REGS: [Register; 16] = [
    Register::RAX,
    Register::RCX,
    Register::RDX,
    Register::RBX,
    Register::RSP,
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

fn reg_index(r: Register) -> usize {
    REGS.iter()
        .position(|&x| x == r.full_register())
        .unwrap_or(4)
}

/// Globals the generated code reads and writes, at fixed addresses.
#[repr(C, align(64))]
struct Globals {
    host: [u8; XSAVE],
    saved_rsp: u64,
    out: u64,
    target: u64,
}

static RUNNING: AtomicBool = AtomicBool::new(false);
static FAULT: AtomicU32 = AtomicU32::new(0);
static RECOVERY: AtomicU64 = AtomicU64::new(0);

unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    if !RUNNING.load(Ordering::Acquire) {
        return 0;
    }
    // SAFETY: the system's record and context; RECOVERY is set before runs.
    unsafe {
        let code = (*(*info).ExceptionRecord).ExceptionCode as u32;
        if code == EXCEPTION_BREAKPOINT as u32 {
            return 0;
        }
        FAULT.store(code, Ordering::Release);
        (*(*info).ContextRecord).Rip = RECOVERY.load(Ordering::Acquire);
    }
    EXCEPTION_CONTINUE_EXECUTION
}

fn op<T>(r: Result<T, iced_x86::IcedError>) -> T {
    r.expect("encoding the harness")
}

fn encode(list: &[Instruction], at: u64) -> Vec<u8> {
    op(BlockEncoder::encode(
        64,
        InstructionBlock::new(list, at),
        BlockEncoderOptions::NONE,
    ))
    .code_buffer
}

fn write_code(at: u64, bytes: &[u8]) {
    // SAFETY: our own RWX buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len());
        FlushInstructionCache(GetCurrentProcess(), at as *const _, bytes.len());
    }
}

const CALLEE_SAVED: [Register; 8] = [
    Register::RBX,
    Register::RBP,
    Register::RDI,
    Register::RSI,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
];

fn mem(base: Register, disp: i64) -> MemoryOperand {
    MemoryOperand::with_base_displ(base, disp)
}

fn mov_imm(r: Register, v: u64) -> Instruction {
    op(Instruction::with2(Code::Mov_r64_imm64, r, v))
}

/// The code that loads a State and jumps to the address in `cell` (a
/// u64 next to the code: a RIP-relative jump has to reach it).
fn runner(g: u64, cell: u64) -> Vec<Instruction> {
    use Register::*;
    let mut l = Vec::new();
    for r in CALLEE_SAVED {
        l.push(op(Instruction::with1(Code::Push_r64, r)));
    }
    l.push(mov_imm(RAX, g));
    l.push(op(Instruction::with2(
        Code::Mov_rm64_r64,
        mem(RAX, XSAVE as i64),
        RSP,
    )));
    l.push(op(Instruction::with2(
        Code::Mov_rm64_r64,
        mem(RAX, XSAVE as i64 + 8),
        RDX,
    )));
    l.push(op(Instruction::with2(Code::Mov_r64_rm64, RBX, RCX)));
    // Keep the host's vector state (xmm6-15 and the control words are
    // callee-saved), then load the test's.
    l.push(op(Instruction::with2(Code::Mov_r32_imm32, EAX, u32::MAX)));
    l.push(op(Instruction::with2(Code::Mov_r32_imm32, EDX, u32::MAX)));
    l.push(mov_imm(RCX, g));
    l.push(op(Instruction::with1(Code::Xsave64_mem, mem(RCX, 0))));
    l.push(op(Instruction::with1(Code::Xrstor64_mem, mem(RBX, 0))));
    l.push(op(Instruction::with1(
        Code::Push_rm64,
        mem(RBX, FLAGS as i64),
    )));
    l.push(Instruction::with(Code::Popfq));
    for (n, r) in REGS.iter().enumerate() {
        if *r != RSP && *r != RBX {
            l.push(op(Instruction::with2(
                Code::Mov_r64_rm64,
                *r,
                mem(RBX, (GPR + 8 * n) as i64),
            )));
        }
    }
    l.push(op(Instruction::with2(
        Code::Mov_r64_rm64,
        RBX,
        mem(RBX, (GPR + 8 * 3) as i64),
    )));
    l.push(op(Instruction::with1(
        Code::Jmp_rm64,
        mem(RIP, cell as i64),
    )));
    l
}

/// The tail every run returns to: saves the State, restores the host.
fn epilogue(g: u64, l: &mut Vec<Instruction>, result: u32) {
    use Register::*;
    l.push(mov_imm(RAX, g));
    l.push(op(Instruction::with2(
        Code::Mov_r64_rm64,
        RSP,
        mem(RAX, XSAVE as i64),
    )));
    l.push(op(Instruction::with2(Code::Mov_r32_imm32, EAX, u32::MAX)));
    l.push(op(Instruction::with2(Code::Mov_r32_imm32, EDX, u32::MAX)));
    l.push(mov_imm(RCX, g));
    l.push(op(Instruction::with1(Code::Xrstor64_mem, mem(RCX, 0))));
    l.push(Instruction::with(Code::Cld));
    for r in CALLEE_SAVED.iter().rev() {
        l.push(op(Instruction::with1(Code::Pop_r64, *r)));
    }
    l.push(op(Instruction::with2(Code::Mov_r32_imm32, EAX, result)));
    l.push(Instruction::with(Code::Retnq));
}

/// The return stub after the instruction: saves registers, flags, vectors.
fn stub(g: u64) -> Vec<Instruction> {
    use Register::*;
    let mut l = vec![
        Instruction::with(Code::Pushfq),
        op(Instruction::with1(Code::Push_r64, RAX)),
    ];
    l.push(mov_imm(RAX, g));
    l.push(op(Instruction::with2(
        Code::Mov_r64_rm64,
        RAX,
        mem(RAX, XSAVE as i64 + 8),
    )));
    for (n, r) in REGS.iter().enumerate() {
        if *r != RSP && *r != RAX {
            l.push(op(Instruction::with2(
                Code::Mov_rm64_r64,
                mem(RAX, (GPR + 8 * n) as i64),
                *r,
            )));
        }
    }
    l.push(op(Instruction::with1(Code::Pop_rm64, mem(RAX, GPR as i64))));
    l.push(op(Instruction::with1(
        Code::Pop_rm64,
        mem(RAX, FLAGS as i64),
    )));
    l.push(op(Instruction::with2(Code::Mov_r64_rm64, RBX, RAX)));
    l.push(op(Instruction::with2(Code::Mov_r32_imm32, EAX, u32::MAX)));
    l.push(op(Instruction::with2(Code::Mov_r32_imm32, EDX, u32::MAX)));
    l.push(op(Instruction::with1(Code::Xsave64_mem, mem(RBX, 0))));
    epilogue(g, &mut l, 0);
    l
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// XSAVE component offsets and sizes (CPUID leaf 0xD), and XCR0.
struct Layout {
    xcr0: u64,
    components: Vec<(u32, usize, usize)>,
}

fn layout() -> Layout {
    // SAFETY: XGETBV is available wherever XSAVE is, which the patcher needs.
    let xcr0 = unsafe { std::arch::x86_64::_xgetbv(0) };
    let mut components = Vec::new();
    for i in [2u32, 5, 6, 7] {
        if xcr0 & (1 << i) != 0 {
            let r = std::arch::x86_64::__cpuid_count(0xD, i);
            components.push((i, r.ebx as usize, r.eax as usize));
        }
    }
    Layout { xcr0, components }
}

/// Vector state as compared: components the header marks as in their
/// initial state count as zeros, and the x87 instruction and data pointers
/// (which name the last x87 instruction's address) are left out.
fn vector_view(s: &State, l: &Layout) -> Vec<u8> {
    let bv = u64::from_le_bytes(s.xsave[512..520].try_into().unwrap());
    let mut v = Vec::new();
    let part = |v: &mut Vec<u8>, bit: u32, bytes: &[u8]| {
        if bv & (1 << bit) != 0 {
            v.extend_from_slice(bytes);
        } else {
            v.extend(std::iter::repeat_n(0, bytes.len()));
        }
    };
    part(&mut v, 0, &s.xsave[0..6]);
    v.extend_from_slice(&s.xsave[24..28]);
    part(&mut v, 0, &s.xsave[32..160]);
    part(&mut v, 1, &s.xsave[160..416]);
    for &(i, off, size) in &l.components {
        part(&mut v, i, &s.xsave[off..off + size]);
    }
    v
}

/// Where a run left things.
struct Outcome {
    fault: u32,
    state: State,
    memory: Vec<u8>,
}

struct Harness {
    globals: u64,
    cell: u64,
    runner: u64,
    tables: Tables,
    layout: Layout,
    template: State,
    teb: u64,
    code_next: u64,
    code_end: u64,
}

impl Harness {
    fn new() -> Result<Harness, String> {
        // The arena: one section, the title's view and ours, at a fixed
        // distance, as a guarded block's delta sends a barrier there.
        // SAFETY: a fresh section mapped at free addresses.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                ARENA_BYTES as u32,
                std::ptr::null(),
            );
            for at in [ARENA, ARENA + WRITER_DELTA] {
                let v = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    0,
                    ARENA_BYTES,
                    0,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                if v.Value as u64 != at {
                    return Err(format!("cannot map the test arena at {at:#x}"));
                }
            }
        }
        // The process's tables, as VIctory uses them: the arena's blocks
        // given their delta, our view's slot its marks.
        let tables = tables::tables().ok_or("no tables")?;
        for b in (0..ARENA_BYTES as u64).step_by(BLOCK as usize) {
            if !tables::set_delta(ARENA + b, WRITER_DELTA) {
                return Err("no delta table".into());
            }
        }
        if !tables::track(ARENA + WRITER_DELTA, ARENA_BYTES as u64) {
            return Err("no marks".into());
        }
        string::set_tables(tables);

        // Code, below the arena's slot so RIP-relative stores reach it.
        let mut code = 0u64;
        for k in 1..16u64 {
            let hint = (TITLE_SLOT << SLOT_SHIFT) - k * (64 << 20);
            // SAFETY: fresh memory at a hint.
            code = unsafe {
                VirtualAlloc(
                    hint as *const _,
                    32 << 20,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_EXECUTE_READWRITE,
                )
            } as u64;
            if code != 0 {
                break;
            }
        }
        if code == 0 {
            return Err("no code memory near the arena".into());
        }
        let globals = Box::leak(Box::new(Globals {
            host: [0; XSAVE],
            saved_rsp: 0,
            out: 0,
            target: 0,
        })) as *mut Globals as u64;
        let runner_at = code;
        let cell = code + 12288;
        write_code(runner_at, &encode(&runner(globals, cell), runner_at));
        let recovery_at = code + 4096;
        let mut rec = Vec::new();
        epilogue(globals, &mut rec, 1);
        write_code(recovery_at, &encode(&rec, recovery_at));
        RECOVERY.store(recovery_at, Ordering::Release);
        // A tiny function that XSAVEs the current state, for a template.
        let snap_at = code + 8192;
        {
            use Register::*;
            let l = vec![
                op(Instruction::with2(Code::Mov_r32_imm32, EAX, u32::MAX)),
                op(Instruction::with2(Code::Mov_r32_imm32, EDX, u32::MAX)),
                op(Instruction::with1(Code::Xsave64_mem, mem(RCX, 0))),
                Instruction::with(Code::Retnq),
            ];
            write_code(snap_at, &encode(&l, snap_at));
        }
        let mut template = State {
            xsave: [0; XSAVE],
            gpr: [0; 16],
            flags: 0x202,
        };
        // SAFETY: the snapshot function takes a 64-byte aligned area.
        unsafe {
            let snap = std::mem::transmute::<u64, extern "win64" fn(*mut u8)>(snap_at);
            snap(template.xsave.as_mut_ptr());
            AddVectoredExceptionHandler(1, Some(on_fault));
        }
        let layout = layout();
        // Every user component present, so random values load.
        let bv = layout.xcr0 & 0xE7;
        template.xsave[512..520].copy_from_slice(&bv.to_le_bytes());
        let teb: u64;
        // SAFETY: gs:[0x30] is the TEB's own address.
        unsafe { std::arch::asm!("mov {}, gs:[0x30]", out(reg) teb) };
        Ok(Harness {
            globals,
            cell,
            runner: runner_at,
            tables,
            layout,
            template,
            teb,
            code_next: code + (1 << 20),
            code_end: code + (32 << 20),
        })
    }

    fn alloc_code(&mut self, bytes: u64) -> Option<u64> {
        let at = self.code_next;
        if at + bytes > self.code_end {
            return None;
        }
        self.code_next = (at + bytes + 63) & !63;
        Some(at)
    }

    fn run(&self, target: u64, start: &State, memory: &[u8]) -> Outcome {
        // SAFETY: the arena's title view is mapped read-write here.
        unsafe { std::ptr::copy_nonoverlapping(memory.as_ptr(), WINDOW as *mut u8, WINDOW_BYTES) };
        let mut out = start.clone();
        // SAFETY: the globals and the generated runner set up in `new`.
        let fault = unsafe {
            *(self.cell as *mut u64) = target;
            FAULT.store(0, Ordering::Release);
            RUNNING.store(true, Ordering::Release);
            let run = std::mem::transmute::<u64, extern "win64" fn(*const State, *mut State) -> u32>(
                self.runner,
            );
            let r = run(start, &mut out);
            RUNNING.store(false, Ordering::Release);
            if r == 0 {
                0
            } else {
                FAULT.load(Ordering::Acquire)
            }
        };
        // SAFETY: as above.
        let memory =
            unsafe { std::slice::from_raw_parts(WINDOW as *const u8, WINDOW_BYTES) }.to_vec();
        Outcome {
            fault,
            state: out,
            memory,
        }
    }

    /// Takes (and so clears) the marks: the parts written, by block of our
    /// view.
    fn take_marks(&self) -> BTreeMap<u64, u16> {
        let mut marked = BTreeMap::new();
        tables::take(|blocks| {
            for &(block, parts) in blocks {
                *marked.entry(block).or_insert(0) |= parts;
            }
        });
        marked
    }
}

/// Whether the title's byte at `at` is covered by a mark: its part, at our
/// view's address of it.
fn marked(marks: &BTreeMap<u64, u16>, at: u64) -> bool {
    let w = at + WRITER_DELTA;
    marks
        .get(&(w & !(BLOCK - 1)))
        .is_some_and(|parts| parts >> ((w >> 12) & 15) & 1 != 0)
}

/// A store found in an image: where, and its form.
struct Sample {
    image: String,
    rva: u64,
    instr: Instruction,
}

fn form(i: &Instruction, s: &Store) -> String {
    match s {
        Store::String { .. } => format!("{:?} rep={}", i.code(), i.has_rep_prefix()),
        Store::Inline(x) => format!(
            "{:?} {:?} base={} index={} same={} bit={}",
            i.code(),
            match x.address {
                Address::Operand => "operand",
                Address::Gs => "gs",
                Address::Absolute(_) => "rip",
                Address::Implicit { .. } => "implicit",
            },
            i.memory_base() != Register::None,
            i.memory_index() != Register::None,
            i.memory_base() != Register::None
                && i.memory_base().full_register() == i.memory_index().full_register(),
            x.bit_offset.is_some(),
        ),
    }
}

fn collect(images: &[String], per_form: usize) -> (BTreeMap<String, Vec<Sample>>, usize) {
    let mut forms: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
    let mut factory = InstructionInfoFactory::new();
    let xsave = string::xsave_area_bytes();
    let mut stores = 0;
    for name in images {
        let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
        // SAFETY: mapped for reading; no entry point runs.
        let module = unsafe {
            LoadLibraryExW(
                wide.as_ptr(),
                std::ptr::null_mut(),
                DONT_RESOLVE_DLL_REFERENCES,
            )
        } as u64;
        if module == 0 {
            eprintln!("difftest: cannot load {name}");
            continue;
        }
        // SAFETY: a loaded image.
        let Some(text) = (unsafe { Text::read(module) }) else {
            continue;
        };
        let file = name.rsplit(['\\', '/']).next().unwrap_or(name).to_string();
        for &(start, end) in &text.sections {
            // SAFETY: an executable section.
            let bytes =
                unsafe { std::slice::from_raw_parts(start as *const u8, (end - start) as usize) };
            for i in Decoder::with_ip(64, bytes, start, DecoderOptions::NONE) {
                let Ok(store) = classify::classify(&i, &mut factory, xsave) else {
                    continue;
                };
                stores += 1;
                let list = forms.entry(form(&i, &store)).or_default();
                if list.len() < per_form {
                    list.push(Sample {
                        image: file.clone(),
                        rva: i.ip() - module,
                        instr: i,
                    });
                }
            }
        }
    }
    (forms, stores)
}

/// One instance set up at its place in the code buffer: the re-encoded
/// original and its trampoline.
struct Prepared {
    original: u64,
    trampoline: u64,
    instr: Instruction,
    store: Store,
    _site: Option<Box<StringSite>>,
}

/// Ways to break a trampoline on purpose, to show the comparison catches
/// each kind of mistake.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sabotage {
    None,
    /// Marks go to the junk maps: memory right, marks missing.
    Marks,
    /// `test rsp, rsp` first: the flags change.
    Flags,
    /// Straight to the return stub: the store never happens.
    Skip,
}

impl Sabotage {
    fn name(self) -> &'static str {
        match self {
            Sabotage::None => "none",
            Sabotage::Marks => "marks",
            Sabotage::Flags => "flags",
            Sabotage::Skip => "skip",
        }
    }
}

fn prepare(
    h: &mut Harness,
    sample: &Sample,
    factory: &mut InstructionInfoFactory,
    sabotage: Sabotage,
) -> Result<Prepared, String> {
    let at = h.alloc_code(1024).ok_or("code buffer full")?;
    let mut i = sample.instr;
    if i.is_ip_rel_memory_operand() {
        // Aim RIP-relative stores at the arena.
        i.set_memory_displacement64(WINDOW + WINDOW_BYTES as u64 / 2);
    } else if i.memory_base() == Register::None
        && i.memory_index() == Register::None
        && i.op_kind(0) == iced_x86::OpKind::Memory
    {
        // Absolute: a 64-bit moffs can be aimed; a [disp32] cannot reach.
        if !matches!(
            i.code(),
            Code::Mov_moffs8_AL
                | Code::Mov_moffs16_AX
                | Code::Mov_moffs32_EAX
                | Code::Mov_moffs64_RAX
        ) {
            return Err("skip: an absolute 32-bit address cannot reach the arena".into());
        }
        i.set_memory_displacement64(WINDOW + WINDOW_BYTES as u64 / 2);
    }
    let bytes = op_result(BlockEncoder::encode(
        64,
        InstructionBlock::new(&[i], at),
        BlockEncoderOptions::NONE,
    ))?
    .code_buffer;
    write_code(at, &bytes);
    let instr = Decoder::with_ip(64, &bytes, at, DecoderOptions::NONE).decode();
    let stub_code = encode(&stub(h.globals), instr.next_ip());
    write_code(instr.next_ip(), &stub_code);
    let store = classify::classify(&instr, factory, string::xsave_area_bytes())
        .map_err(|r| format!("re-encoded form refused: {r:?}"))?;
    let site = match store {
        Store::String { kind, element, rep } => Some(Box::new(StringSite { kind, element, rep })),
        Store::Inline(_) => None,
    };
    let tramp = h.alloc_code(2048).ok_or("code buffer full")?;
    let site_ptr = site
        .as_deref()
        .map_or(std::ptr::null(), |s| s as *const StringSite);
    let mut tables = h.tables;
    if sabotage == Sabotage::Marks {
        // SAFETY: slot 0's entry, which holds a junk map.
        let junk = |t: *const u64| -> *const u64 {
            Box::leak(vec![unsafe { *t }; SLOTS].into_boxed_slice()).as_ptr()
        };
        // Every slot, ours too, gets slot 0's (junk) maps.
        tables = Tables {
            part: junk(tables.part),
            fine: junk(tables.fine),
            mid: junk(tables.mid),
            coarse: junk(tables.coarse),
            ..tables
        };
    }
    let encoded = emit::trampoline(tramp, &[instr], &[Some(store)], tables, site_ptr)
        .map_err(|r| format!("no trampoline: {r:?}"))?;
    write_code(tramp, &encoded.code);
    let entry = match sabotage {
        Sabotage::Flags => {
            let pre = h.alloc_code(64).ok_or("code buffer full")?;
            let l = [
                op(Instruction::with2(
                    Code::Test_rm64_r64,
                    Register::RSP,
                    Register::RSP,
                )),
                op(Instruction::with_branch(Code::Jmp_rel32_64, tramp)),
            ];
            write_code(pre, &encode(&l, pre));
            pre
        }
        Sabotage::Skip => instr.next_ip(),
        _ => tramp,
    };
    Ok(Prepared {
        original: at,
        trampoline: entry,
        instr,
        store,
        _site: site,
    })
}

fn op_result<T>(r: Result<T, iced_x86::IcedError>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

/// A random start for `p`: registers, flags, vectors, with the address
/// registers solved so the store lands in the window. None if it cannot.
fn start(h: &Harness, p: &Prepared, rng: &mut Rng, memory: &[u8]) -> Option<State> {
    let mut s = h.template.clone();
    for g in &mut s.gpr {
        *g = rng.next();
    }
    // Flags: CF PF AF ZF SF OF at random, DF at random, IF and bit 1 set.
    s.flags = 0x202 | (rng.next() & 0x8D5) | if rng.below(4) == 0 { 0x400 } else { 0 };
    let l = &h.layout;
    for k in (160..416).step_by(8) {
        s.xsave[k..k + 8].copy_from_slice(&rng.next().to_le_bytes());
    }
    for &(_, off, size) in &l.components {
        for k in (off..off + size).step_by(8) {
            s.xsave[k..k + 8].copy_from_slice(&rng.next().to_le_bytes());
        }
    }
    // Aligned half the time (so aligned-only forms run), anywhere otherwise.
    let center = WINDOW + WINDOW_BYTES as u64 / 2;
    let mut target = center - 2048 + rng.below(4096);
    if rng.below(2) == 0 {
        target &= !63;
    }
    let i = &p.instr;
    match p.store {
        Store::String { .. } => {
            s.gpr[reg_index(Register::RDI)] = target;
            s.gpr[reg_index(Register::RSI)] = center - 8192 + rng.below(16384);
            if i.has_rep_prefix() {
                s.gpr[reg_index(Register::RCX)] = rng.below(48);
            }
        }
        Store::Inline(x) => {
            match x.address {
                Address::Absolute(_) => {}
                Address::Implicit { reg, .. } => s.gpr[reg_index(reg)] = target,
                Address::Operand | Address::Gs => {
                    let (b, ix, scale) = (
                        i.memory_base(),
                        i.memory_index(),
                        u64::from(i.memory_index_scale()),
                    );
                    let disp = i.memory_displacement64();
                    let gs = if x.address == Address::Gs { h.teb } else { 0 };
                    let want = target.wrapping_sub(disp).wrapping_sub(gs);
                    match (b, ix) {
                        (Register::None, Register::None) => return None,
                        (Register::None, ix) => {
                            let v = want / scale;
                            s.gpr[reg_index(ix)] = v;
                        }
                        (b, Register::None) => s.gpr[reg_index(b)] = want,
                        (b, ix) if b.full_register() == ix.full_register() => {
                            s.gpr[reg_index(b)] = want / (1 + scale);
                        }
                        (b, ix) => {
                            let iv = rng.below(32);
                            s.gpr[reg_index(ix)] = iv;
                            s.gpr[reg_index(b)] = want.wrapping_sub(iv * scale);
                        }
                    }
                }
            }
            if let Some((reg, _)) = x.bit_offset {
                if [i.memory_base(), i.memory_index()]
                    .iter()
                    .any(|r| *r != Register::None && r.full_register() == reg.full_register())
                {
                    return None;
                }
                let bits = rng.below(8 * 8192) as i64 - 4 * 8192;
                s.gpr[reg_index(reg)] = bits as u64;
            }
            // cmpxchg: half the time, make the comparison succeed.
            if matches!(
                i.mnemonic(),
                Mnemonic::Cmpxchg | Mnemonic::Cmpxchg8b | Mnemonic::Cmpxchg16b
            ) && rng.below(2) == 0
            {
                let addr_regs = [
                    i.memory_base().full_register(),
                    i.memory_index().full_register(),
                ];
                if !addr_regs.contains(&Register::RAX) && !addr_regs.contains(&Register::RDX) {
                    // The address the store will use, from the solved registers.
                    let ea = target;
                    if ea >= WINDOW && ea + 16 <= WINDOW + WINDOW_BYTES as u64 {
                        let off = (ea - WINDOW) as usize;
                        s.gpr[0] = u64::from_le_bytes(memory[off..off + 8].try_into().unwrap());
                        s.gpr[2] =
                            u64::from_le_bytes(memory[off + 8..off + 16].try_into().unwrap());
                    }
                }
            }
        }
    }
    Some(s)
}

/// What differs between the original's run and the trampoline's, if
/// anything.
fn compare(
    h: &Harness,
    a: &Outcome,
    b: &Outcome,
    initial: &[u8],
    marks: &BTreeMap<u64, u16>,
) -> Option<String> {
    if a.fault != 0 || b.fault != 0 {
        return (a.fault != b.fault).then(|| format!("fault {:#x} vs {:#x}", a.fault, b.fault));
    }
    for (n, r) in REGS.iter().enumerate() {
        if *r != Register::RSP && a.state.gpr[n] != b.state.gpr[n] {
            return Some(format!(
                "{r:?} {:#x} vs {:#x}",
                a.state.gpr[n], b.state.gpr[n]
            ));
        }
    }
    if (a.state.flags ^ b.state.flags) & 0xCD5 != 0 {
        return Some(format!(
            "flags {:#x} vs {:#x}",
            a.state.flags, b.state.flags
        ));
    }
    if vector_view(&a.state, &h.layout) != vector_view(&b.state, &h.layout) {
        return Some("vector state".into());
    }
    if let Some(k) = (0..WINDOW_BYTES).find(|&k| a.memory[k] != b.memory[k]) {
        return Some(format!(
            "memory at +{k:#x}: {:#x} vs {:#x}",
            a.memory[k], b.memory[k]
        ));
    }
    if let Some(k) =
        (0..WINDOW_BYTES).find(|&k| b.memory[k] != initial[k] && !marked(marks, WINDOW + k as u64))
    {
        return Some(format!("byte +{k:#x} written but not marked"));
    }
    None
}

/// What a run over a corpus came to.
#[derive(Debug, Default)]
struct Report {
    forms: usize,
    instances: usize,
    runs: usize,
    faulted: usize,
    skipped: usize,
    failed_forms: usize,
    failures: Vec<String>,
}

/// The one harness of the process (its arena is at fixed addresses).
static HARNESS: Mutex<Option<Harness>> = Mutex::new(None);

fn difftest(images: &[String], per_form: usize, trials: usize, sabotage: Sabotage) -> Report {
    let mut guard = HARNESS.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(Harness::new().expect("the harness"));
    }
    let h = guard.as_mut().unwrap();
    let (forms, _) = collect(images, per_form);
    let mut factory = InstructionInfoFactory::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let fmt = |i: &Instruction| {
        let mut t = String::new();
        IntelFormatter::new().format(i, &mut t);
        t
    };
    let mut r = Report {
        forms: forms.len(),
        instances: forms.values().map(Vec::len).sum(),
        ..Report::default()
    };
    for (key, samples) in &forms {
        let mut form_failed = false;
        for sample in samples {
            // The original cannot run on a CPU without the instruction (the
            // trampoline can: it stores plainly).
            if sample.instr.code() == Code::Clzeroq
                && std::arch::x86_64::__cpuid(0x8000_0008).ebx & 1 == 0
            {
                r.skipped += trials;
                continue;
            }
            // Each instance's code is dead once its trials are done: reuse it.
            let mark = h.code_next;
            let p = match prepare(h, sample, &mut factory, sabotage) {
                Ok(p) => p,
                Err(e) if e.starts_with("skip:") => {
                    r.skipped += trials;
                    continue;
                }
                Err(e) => {
                    r.failures.push(format!(
                        "{}+{:#x} `{}` [{key}]: {e}",
                        sample.image,
                        sample.rva,
                        fmt(&sample.instr)
                    ));
                    form_failed = true;
                    continue;
                }
            };
            for _ in 0..trials {
                let initial: Vec<u8> = (0..WINDOW_BYTES / 8)
                    .flat_map(|_| rng.next().to_le_bytes())
                    .collect();
                let Some(s) = start(h, &p, &mut rng, &initial) else {
                    r.skipped += 1;
                    continue;
                };
                let a = h.run(p.original, &s, &initial);
                let _ = h.take_marks();
                let b = h.run(p.trampoline, &s, &initial);
                let marks = h.take_marks();
                r.runs += 1;
                if a.fault != 0 {
                    r.faulted += 1;
                }
                if let Some(why) = compare(h, &a, &b, &initial, &marks) {
                    if !form_failed {
                        r.failures.push(format!(
                            "{}+{:#x} `{}` [{key}]: {why}",
                            sample.image,
                            sample.rva,
                            fmt(&p.instr)
                        ));
                    }
                    form_failed = true;
                    break;
                }
            }
            h.code_next = mark;
        }
        r.failed_forms += usize::from(form_failed);
    }
    r
}

/// This test binary and a few of Windows' own DLLs (MSVC output, the
/// closest thing at hand to a title's code); those not present are left out.
fn default_images() -> Vec<String> {
    let me = std::env::current_exe()
        .expect("this test")
        .to_string_lossy()
        .into_owned();
    std::iter::once(me)
        .chain(["ntdll.dll", "kernelbase.dll", "ucrtbase.dll"].map(String::from))
        .collect()
}

#[test]
fn every_store_form_does_what_the_cpu_does() {
    let images = default_images();
    let r = difftest(&images, 3, 8, Sabotage::None);
    eprintln!("{r:?}");
    assert!(
        r.forms > 100 && r.runs > 1000,
        "too little was compared: {r:?}"
    );
    assert_eq!(
        r.failed_forms, 0,
        "trampolines differ from their instructions: {:#?}",
        r.failures
    );
    // Each way of breaking a trampoline is seen.
    for sabotage in [Sabotage::Marks, Sabotage::Flags, Sabotage::Skip] {
        let broken = difftest(&images, 1, 4, sabotage);
        assert!(
            broken.failed_forms * 2 > broken.forms,
            "{}: only {} of {} forms failed",
            sabotage.name(),
            broken.failed_forms,
            broken.forms
        );
    }
}

/// Every instance of the images in `VICTORY_BARRIER_DIFFTEST` (`;`-separated
/// paths), `VICTORY_BARRIER_DIFFTEST_PER_FORM` of each form (3).
#[test]
#[ignore = "VICTORY_BARRIER_DIFFTEST=<images>"]
fn every_store_form_of_the_images_asked_for() {
    let images: Vec<String> = std::env::var("VICTORY_BARRIER_DIFFTEST")
        .expect("VICTORY_BARRIER_DIFFTEST=<images>")
        .split(';')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let per_form = std::env::var("VICTORY_BARRIER_DIFFTEST_PER_FORM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let r = difftest(&images, per_form, 4, Sabotage::None);
    println!(
        "{} forms, {} instances, {} paired runs ({} where the original faulted), {} skipped; {} forms failed",
        r.forms, r.instances, r.runs, r.faulted, r.skipped, r.failed_forms
    );
    for f in r.failures.iter().take(40) {
        println!("- {f}");
    }
    assert_eq!(r.failed_forms, 0);
}
