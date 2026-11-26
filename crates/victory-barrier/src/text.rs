//! A code image as far as patching needs it: where its code is, what in it
//! might be branched to, and where it has padding nobody executes.
//!
//! A patch that overwrites only the store's own bytes is safe whatever
//! branches where: nothing can enter an instruction in the middle. Only
//! relocation (taking the next instructions into the trampoline) needs to
//! know that nothing branches into them, and that is undecidable in
//! general: a computed jump can go anywhere. This module collects what can
//! be seen statically:
//!
//! - direct jumps, conditional jumps and calls (a linear sweep);
//! - code addresses the code takes (`lea r, [rip+x]`, any RIP-relative
//!   operand, a 64-bit immediate) that land in code: function pointers,
//!   labels a title resumes at after an exception, manual return addresses;
//! - absolute pointers into code in the image's data (`.reloc`): vtables,
//!   absolute jump tables;
//! - relative jump tables, found from the indirect jump that uses them
//!   (`lea base, [rip+T]` … `movsxd r, [base+i*4]` … `add r, base`, `jmp
//!   r`; MSVC's image-base-relative form is the same shape), bounded by the
//!   `cmp i, n` / `ja` in front, or else read while entries land in the
//!   jump's own function;
//! - function starts (`.pdata`).
//!
//! What it cannot see: targets computed at run time from anything but a
//! table (an offset added to a function's address, an obfuscated pointer).

use iced_x86::{Code, Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind};

#[derive(Clone, Copy, Debug, Default)]
pub struct TargetStats {
    pub direct: usize,
    pub address_taken: usize,
    pub relocations: usize,
    pub jump_tables: usize,
    pub table_entries: usize,
    pub functions: usize,
}

pub struct Text {
    pub base: u64,
    /// Executable sections: (start, end).
    pub sections: Vec<(u64, u64)>,
    targets: Vec<u64>,
    /// `.pdata` entries (begin, end, unwind data), absolute, sorted.
    functions: Vec<(u64, u64, u64)>,
    /// Per function: nothing can branch into its middle that the scan did
    /// not see (see `relocation_proven`).
    closed: Vec<bool>,
    /// Why a function is not closed: an unread indirect jump's address, or
    /// 0 for an exception handler.
    open_why: Vec<u64>,
    /// Each function's exception handler.
    handlers: Vec<Handler>,
    /// Addresses entered from outside the control flow the scan follows:
    /// function starts, taken code addresses, `.reloc` pointers, exception
    /// continuations. Sorted.
    external: Vec<u64>,
    /// Edges from jumps through tables the scan read: (jump, target).
    table_edges: Vec<(u64, u64)>,
    /// Handlers not understood: address, functions using it.
    pub unknown_handlers: std::collections::BTreeMap<u64, usize>,
    /// Indirect jumps left unread: the instructions before, the jump.
    pub unread: Vec<(Vec<Instruction>, Instruction)>,
    size: u64,
    pub stats: TargetStats,
    /// Whether this is an image (with headers, tables, padding) or
    /// anonymous executable memory (JIT code), of which nothing is known.
    pub image: bool,
    /// Whether a stretch of code with no `.pdata` holds no indirect jump
    /// but tail calls (see `leaf_proven`), by (start, end).
    gaps: std::sync::Mutex<std::collections::HashMap<(u64, u64), bool>>,
}

fn u16_at(at: u64) -> u16 {
    // SAFETY: inside the mapped image's headers, which the caller checked.
    unsafe { (at as *const u16).read_unaligned() }
}
fn u32_at(at: u64) -> u32 {
    // SAFETY: as above.
    unsafe { (at as *const u32).read_unaligned() }
}
fn u64_at(at: u64) -> u64 {
    // SAFETY: as above.
    unsafe { (at as *const u64).read_unaligned() }
}

const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;

/// The RIP-relative addresses `reg` can hold at `at` in `function`, if on
/// every control-flow path back from `at` the last write to `reg` is a
/// `lea reg, [rip+x]`. None if any path reaches something else first: a
/// write of another kind, a call (which clobbers it), or an instruction
/// entered from outside the flow (the function's start, a taken address).
///
/// The flow is direct branches, fall-through and the edges of jump tables
/// already read. An instruction with no predecessor in that flow is taken
/// as unreachable: that is optimistic while some of the function's jumps
/// are unread, which is why `Text::read` checks every answer again once
/// they are, and why a function with a jump left unread is not closed at
/// all.
fn reaching_leas(
    text: &Text,
    function: (u64, u64),
    at: u64,
    reg: iced_x86::Register,
) -> Option<Vec<u64>> {
    use iced_x86::{InstructionInfoFactory, OpAccess};
    use std::collections::HashMap;
    let dbg = std::env::var_os("VICTORY_BARRIER_DEBUG_REACH").is_some();
    // SAFETY: the function's bytes in an executable section.
    let bytes = unsafe {
        std::slice::from_raw_parts(function.0 as *const u8, (function.1 - function.0) as usize)
    };
    let code: Vec<Instruction> = Decoder::with_ip(64, bytes, function.0, DecoderOptions::NONE)
        .into_iter()
        .collect();
    let index: HashMap<u64, usize> = code.iter().enumerate().map(|(k, i)| (i.ip(), k)).collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); code.len()];
    let edge = |from: usize, to: u64, preds: &mut Vec<Vec<usize>>| -> Option<()> {
        if to < function.0 || to >= function.1 {
            return Some(());
        }
        // Into the middle of an instruction: the linear decode is out of
        // step with the code. Give up.
        let Some(&k) = index.get(&to) else {
            if dbg {
                eprintln!(
                    "reach: an edge into the middle of an instruction (+{:#x})",
                    to - function.0
                );
            }
            return None;
        };
        preds[k].push(from);
        Some(())
    };
    for (k, i) in code.iter().enumerate() {
        let falls = !matches!(
            i.flow_control(),
            FlowControl::UnconditionalBranch
                | FlowControl::IndirectBranch
                | FlowControl::Return
                | FlowControl::Interrupt
        );
        if falls && k + 1 < code.len() {
            preds[k + 1].push(k);
        }
        if matches!(
            i.flow_control(),
            FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch
        ) && i.op_kind(0) == OpKind::NearBranch64
        {
            edge(k, i.near_branch_target(), &mut preds)?;
        }
    }
    for &(jmp, to) in &text.table_edges {
        if let Some(&k) = index.get(&jmp) {
            edge(k, to, &mut preds)?;
        }
    }
    let &start = index.get(&at)?;
    let reg = reg.full_register();
    let mut factory = InstructionInfoFactory::new();
    let mut found = Vec::new();
    let mut seen = vec![false; code.len()];
    let mut stack: Vec<usize> = preds[start].clone();
    while let Some(k) = stack.pop() {
        if std::mem::replace(&mut seen[k], true) {
            continue;
        }
        let i = &code[k];
        if i.code() == Code::Lea_r64_m && i.op0_register() == reg {
            if !i.is_ip_rel_memory_operand() {
                return None;
            }
            found.push(i.ip_rel_memory_address());
            continue;
        }
        let writes = factory.info(i).used_registers().iter().any(|u| {
            u.register().full_register() == reg
                && !matches!(
                    u.access(),
                    OpAccess::Read | OpAccess::CondRead | OpAccess::NoMemAccess
                )
        });
        let clobbered_by_call = matches!(
            i.flow_control(),
            FlowControl::Call | FlowControl::IndirectCall
        ) && matches!(
            reg,
            iced_x86::Register::RAX
                | iced_x86::Register::RCX
                | iced_x86::Register::RDX
                | iced_x86::Register::R8
                | iced_x86::Register::R9
                | iced_x86::Register::R10
                | iced_x86::Register::R11
        );
        if writes || clobbered_by_call {
            if dbg {
                eprintln!("reach: {reg:?} written at +{:#x}", i.ip() - function.0);
            }
            return None;
        }
        if i.ip() == function.0 || text.is_external(i.ip()) {
            if dbg {
                eprintln!(
                    "reach: +{:#x} is entered from outside the flow",
                    i.ip() - function.0
                );
            }
            return None;
        }
        stack.extend(preds[k].iter().copied());
    }
    found.sort_unstable();
    found.dedup();
    (!found.is_empty()).then_some(found)
}

thread_local! {
    /// The handler address `read_handler` last looked at.
    static UNKNOWN_HANDLER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// `open_why` of a function another layer's patch jumps out of.
const FOREIGN: u64 = u64::MAX;

/// Whether an instruction's encoding has a REX prefix with W set (after
/// any legacy prefixes).
fn rex_w(i: &Instruction) -> bool {
    // SAFETY: the instruction's own bytes, at most 15.
    let bytes = unsafe { std::slice::from_raw_parts(i.ip() as *const u8, i.len()) };
    let rex = bytes.iter().find(|&&b| {
        !matches!(
            b,
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3
        )
    });
    rex.is_some_and(|&b| (0x48..=0x4F).contains(&b))
}

/// What a function's exception handler allows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handler {
    /// No language handler.
    None,
    /// MSVC C++ (`__CxxFrameHandler3`, magic 0x1993052x): cleanups run in
    /// separate funclets and unwinding goes on to the caller; a catch's
    /// funclet returns its continuation with `lea rax, [rip+x]`, which the
    /// scan takes as a target. Nothing resumes the function elsewhere.
    CxxFh3,
    /// SEH (`__C_specific_handler`): each `__except` block's start (its
    /// scope record's jump target) becomes a target.
    CSpecific,
    /// `/GS` cookie check (`__GSHandlerCheck`): checks the stack cookie and
    /// returns ExceptionContinueSearch. Never resumes the function.
    GsCheck,
    /// MSVC C++ FH4 (`__CxxFrameHandler4`): cleanups run in funclets; each
    /// catch's continuation, read from its compressed tables, becomes a
    /// target.
    CxxFh4,
    /// Anything else (a custom handler, a wrapper not recognised): the
    /// function may be resumed anywhere.
    Unknown,
}

/// The import a `jmp [rip+x]` stub at `at` goes to, by name.
fn import_name(base: u64, at: u64) -> Option<String> {
    // SAFETY: code of a loaded image.
    let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, 15) };
    let i = Decoder::with_ip(64, bytes, at, DecoderOptions::NONE).decode();
    if i.mnemonic() != Mnemonic::Jmp
        || i.op_kind(0) != OpKind::Memory
        || !i.is_ip_rel_memory_operand()
    {
        return None;
    }
    let slot = i.ip_rel_memory_address();
    let nt = base + u64::from(u32_at(base + 0x3C));
    let imports = u64::from(u32_at(nt + 24 + 112 + 8));
    if imports == 0 {
        return None;
    }
    let mut desc = base + imports;
    while u32_at(desc + 12) != 0 {
        let (names, iat) = (u64::from(u32_at(desc)), u64::from(u32_at(desc + 16)));
        let names = if names != 0 { names } else { iat };
        let mut k = 0;
        loop {
            let entry = u64_at(base + names + k * 8);
            if entry == 0 {
                break;
            }
            if base + iat + k * 8 == slot {
                if entry >> 63 != 0 {
                    return None;
                }
                // SAFETY: an IMAGE_IMPORT_BY_NAME: a hint, then the name.
                let name = unsafe { std::ffi::CStr::from_ptr((base + entry + 2) as *const i8) };
                return Some(name.to_string_lossy().into_owned());
            }
            k += 1;
        }
        desc += 20;
    }
    None
}

/// The export of `base` at `at`, by name.
fn export_name(base: u64, at: u64) -> Option<String> {
    let nt = base + u64::from(u32_at(base + 0x3C));
    let exports = u64::from(u32_at(nt + 24 + 112));
    if exports == 0 {
        return None;
    }
    let dir = base + exports;
    let (count, functions, names, ordinals) = (
        u64::from(u32_at(dir + 24)),
        base + u64::from(u32_at(dir + 28)),
        base + u64::from(u32_at(dir + 32)),
        base + u64::from(u32_at(dir + 36)),
    );
    for k in 0..count {
        let ordinal = u64::from(u16_at(ordinals + k * 2));
        if base + u64::from(u32_at(functions + ordinal * 4)) == at {
            // SAFETY: a NUL-terminated export name in the image.
            let name = unsafe {
                std::ffi::CStr::from_ptr((base + u64::from(u32_at(names + k * 4))) as *const i8)
            };
            return Some(name.to_string_lossy().into_owned());
        }
    }
    None
}

/// Handler kinds by the runtime's names for them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Named {
    Gs,
    Fh3,
    Fh4,
    Seh,
}

fn named(name: &str) -> Option<Named> {
    Some(match name {
        "__GSHandlerCheck" => Named::Gs,
        "__CxxFrameHandler3" | "__GSHandlerCheck_EH" => Named::Fh3,
        "__CxxFrameHandler4" | "__GSHandlerCheck_EH4" => Named::Fh4,
        "__C_specific_handler" | "__GSHandlerCheck_SEH" => Named::Seh,
        _ => return None,
    })
}

/// What the handler at `at` is: by its import or export name, by the
/// `__GSHandlerCheck` body, or (a heuristic) by the one named handler its
/// code calls: the `/GS` wrappers (`__GSHandlerCheck_EH4` and friends)
/// check the cookie, then pass the same dispatcher context, and so the same
/// handler data, to the real handler, which decides where to resume.
fn identify(base: u64, at: u64, depth: u32) -> Option<Named> {
    if let Some(k) = import_name(base, at)
        .or_else(|| export_name(base, at))
        .as_deref()
        .and_then(named)
    {
        return Some(k);
    }
    if is_gs_check(at) {
        return Some(Named::Gs);
    }
    if depth == 0 {
        return None;
    }
    // SAFETY: code of a loaded image, up to a short handler's length.
    let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, 256) };
    let mut inner = None;
    for i in Decoder::with_ip(64, bytes, at, DecoderOptions::NONE)
        .into_iter()
        .take(96)
    {
        let call = matches!(
            i.flow_control(),
            FlowControl::Call | FlowControl::UnconditionalBranch
        ) && i.op_kind(0) == OpKind::NearBranch64;
        if call {
            match identify(base, i.near_branch_target(), depth - 1) {
                Some(Named::Gs) | None => {}
                Some(k) if inner.is_none_or(|x| x == k) => inner = Some(k),
                // Two different real handlers: not a wrapper this knows.
                Some(_) => return None,
            }
        }
        if matches!(
            i.flow_control(),
            FlowControl::Return | FlowControl::UnconditionalBranch
        ) && !call
        {
            break;
        }
    }
    inner
}

/// Whether the code at `at` is MSVC's `__GSHandlerCheck`: `sub rsp,28h;
/// mov r8,[r9+38h]; mov rcx,rdx; mov rdx,r9; call __GSHandlerCheckCommon;
/// mov eax,1; add rsp,28h; ret`. It returns ExceptionContinueSearch.
fn is_gs_check(at: u64) -> bool {
    use iced_x86::Formatter;
    // SAFETY: code of a loaded image.
    let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, 32) };
    let mut f = iced_x86::IntelFormatter::new();
    let mut shape = Vec::new();
    for i in Decoder::with_ip(64, bytes, at, DecoderOptions::NONE)
        .into_iter()
        .take(8)
    {
        let mut t = String::new();
        if i.flow_control() == FlowControl::Call && i.op_kind(0) == OpKind::NearBranch64 {
            t.push_str("call");
        } else {
            f.format(&i, &mut t);
        }
        shape.push(t);
    }
    shape
        == [
            "sub rsp,28h",
            "mov r8,[r9+38h]",
            "mov rcx,rdx",
            "mov rdx,r9",
            "call",
            "mov eax,1",
            "add rsp,28h",
            "ret",
        ]
}

/// FH4's compressed unsigned integer (`ReadUnsigned` in the MSVC
/// runtime's `ehdata4_export.h`): the low 4 bits of the first byte give
/// the length (1-5 bytes); the value is the u32 ending at the last byte,
/// shifted right. Advances `at`.
fn fh4_unsigned(at: &mut u64) -> u32 {
    const LENGTH: [u64; 16] = [1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5];
    const SHIFT: [u32; 16] = [25, 18, 25, 11, 25, 18, 25, 4, 25, 18, 25, 11, 25, 18, 25, 0];
    // SAFETY: inside the image's exception data (the caller checked).
    let low = usize::from(unsafe { *(*at as *const u8) } & 0x0F);
    let n = LENGTH[low];
    let value = u32_at(*at + n - 4) >> SHIFT[low];
    *at += n;
    value
}

/// The places FH4 resumes a function at after a catch, from its function
/// info at `func_info` (`DecompFuncInfo`, `TryBlockMap4`, `HandlerMap4` in
/// `ehdata4_export.h`). `function` is the function's start (continuations
/// are relative to it unless marked as RVAs). None if the data does not
/// read as FH4. A catch with no continuation in the tables returns it from
/// its funclet with a `lea`, which the scan takes as a taken address.
fn fh4_continuations(
    base: u64,
    func_info: u64,
    function: u64,
    in_image: &dyn Fn(u64) -> bool,
) -> Option<Vec<u64>> {
    let mut at = func_info;
    // SAFETY: the header byte, inside the image.
    let header = unsafe { *(at as *const u8) };
    at += 1;
    if header & 0x80 != 0 {
        return None;
    }
    if header & 0x04 != 0 {
        fh4_unsigned(&mut at);
    }
    if header & 0x08 != 0 {
        at += 4;
    }
    if header & 0x10 == 0 {
        return Some(Vec::new());
    }
    let try_map = base + u64::from(u32_at(at));
    if !in_image(try_map) {
        return None;
    }
    let mut t = try_map;
    let blocks = fh4_unsigned(&mut t);
    if blocks > 4096 {
        return None;
    }
    let mut out = Vec::new();
    for _ in 0..blocks {
        fh4_unsigned(&mut t);
        fh4_unsigned(&mut t);
        fh4_unsigned(&mut t);
        let handlers = base + u64::from(u32_at(t));
        t += 4;
        if !in_image(handlers) {
            return None;
        }
        let mut h = handlers;
        let count = fh4_unsigned(&mut h);
        if count > 4096 {
            return None;
        }
        for _ in 0..count {
            // SAFETY: the handler entry's header byte, inside the image.
            let hh = unsafe { *(h as *const u8) };
            h += 1;
            if hh & 0x01 != 0 {
                fh4_unsigned(&mut h);
            }
            if hh & 0x02 != 0 {
                h += 4;
            }
            if hh & 0x04 != 0 {
                fh4_unsigned(&mut h);
            }
            h += 4;
            let (is_rva, conts) = (hh & 0x08 != 0, (hh >> 4) & 3);
            if conts == 3 {
                return None;
            }
            for _ in 0..conts {
                let c = if is_rva {
                    let rva = u32_at(h);
                    h += 4;
                    base + u64::from(rva)
                } else {
                    function + u64::from(fh4_unsigned(&mut h))
                };
                if !in_image(c) {
                    return None;
                }
                out.push(c);
            }
        }
    }
    Some(out)
}

/// Reads the language handler of the function whose UNWIND_INFO is at
/// `info` (following chained entries to the primary one), adding the
/// addresses it can resume at to `targets`. `function` is its range.
fn read_handler(base: u64, mut info: u64, function: (u64, u64), targets: &mut Vec<u64>) -> Handler {
    // The function whose UNWIND_INFO holds the handler: the primary one of
    // a chain. FH4's continuations are relative to its start.
    let mut primary = function.0;
    for _ in 0..8 {
        // SAFETY: UNWIND_INFO in the loaded image.
        let (flags, codes) = unsafe {
            (
                *(info as *const u8) >> 3,
                u64::from(*((info + 2) as *const u8)),
            )
        };
        let after_codes = info + 4 + codes.next_multiple_of(2) * 2;
        if flags & 4 != 0 {
            // Chained: a RUNTIME_FUNCTION after the codes.
            primary = base + u64::from(u32_at(after_codes));
            info = base + u64::from(u32_at(after_codes + 8));
            continue;
        }
        if flags & 1 == 0 {
            return Handler::None;
        }
        // The handler's RVA, then its data.
        let handler = base + u64::from(u32_at(after_codes));
        let data = after_codes + 4;
        let func_info = base + u64::from(u32_at(data));
        let size = u64::from(u32_at(base + u64::from(u32_at(base + 0x3C)) + 24 + 56));
        let in_image = |a: u64| a >= base && a < base + size;
        // Who the handler is: a name, code recognised, or what it calls.
        match identify(base, handler, 2) {
            Some(Named::Gs) => return Handler::GsCheck,
            Some(Named::Seh) => {
                // Named, so the data is a scope table: count, then records
                // of (begin, end, handler or filter, jump target) RVAs. A
                // jump target is where an `__except` block resumes.
                let count = u64::from(u32_at(data));
                if count >= 4096 || !in_image(data + 4 + count * 16) {
                    return Handler::Unknown;
                }
                for k in 0..count {
                    let jump = u32_at(data + 4 + k * 16 + 12);
                    if jump != 0 && in_image(base + u64::from(jump)) {
                        targets.push(base + u64::from(jump));
                    }
                }
                return Handler::CSpecific;
            }
            Some(Named::Fh4) if in_image(func_info) => {
                return match fh4_continuations(base, func_info, primary, &in_image) {
                    Some(conts) => {
                        if std::env::var_os("VICTORY_BARRIER_DEBUG_FH4").is_some()
                            && !conts.is_empty()
                        {
                            eprintln!(
                                "fh4 function +{:#x}: continuations {:x?}",
                                primary - base,
                                conts.iter().map(|c| c - base).collect::<Vec<_>>()
                            );
                        }
                        targets.extend(conts);
                        Handler::CxxFh4
                    }
                    None => Handler::Unknown,
                };
            }
            Some(Named::Fh3)
                if in_image(func_info)
                    && (0x1993_0520..=0x1993_0522).contains(&u32_at(func_info)) =>
            {
                return Handler::CxxFh3;
            }
            _ => {}
        }
        if in_image(func_info) && (0x1993_0520..=0x1993_0522).contains(&u32_at(func_info)) {
            return Handler::CxxFh3;
        }
        let count = u64::from(u32_at(data));
        let within = |rva: u32| {
            let at = base + u64::from(rva);
            at >= function.0 && at < function.1
        };
        UNKNOWN_HANDLER.with(|h| h.set(base + u64::from(u32_at(after_codes))));
        if (1..1024).contains(&count)
            && (0..count).all(|k| {
                within(u32_at(data + 4 + k * 16))
                    && u32_at(data + 4 + k * 16 + 4) >= u32_at(data + 4 + k * 16)
            })
        {
            for k in 0..count {
                let jump = u32_at(data + 4 + k * 16 + 12);
                if jump != 0 {
                    targets.push(base + u64::from(jump));
                }
            }
            return Handler::CSpecific;
        }
        return Handler::Unknown;
    }
    Handler::Unknown
}

impl Text {
    /// Executable memory that is not an image: no targets, no padding.
    pub fn anonymous(start: u64, end: u64) -> Self {
        Self {
            base: start,
            sections: vec![(start, end)],
            targets: Vec::new(),
            functions: Vec::new(),
            closed: Vec::new(),
            open_why: Vec::new(),
            handlers: Vec::new(),
            external: Vec::new(),
            table_edges: Vec::new(),
            unknown_handlers: Default::default(),
            gaps: Default::default(),
            unread: Vec::new(),
            size: end - start,
            stats: TargetStats::default(),
            image: false,
        }
    }

    /// Reads the image mapped at `base`.
    ///
    /// # Safety
    ///
    /// `base` is a loaded PE32+ image.
    pub unsafe fn read(base: u64) -> Option<Self> {
        if u16_at(base) != 0x5A4D {
            return None;
        }
        let nt = base + u64::from(u32_at(base + 0x3C));
        if u32_at(nt) != 0x4550 {
            return None;
        }
        let sections_count = u64::from(u16_at(nt + 6));
        let optional = nt + 24;
        let optional_size = u64::from(u16_at(nt + 20));
        let image_size = u64::from(u32_at(optional + 56));
        let directory = |i: u64| {
            (
                u64::from(u32_at(optional + 112 + i * 8)),
                u64::from(u32_at(optional + 112 + i * 8 + 4)),
            )
        };
        let mut text = Self {
            base,
            sections: Vec::new(),
            targets: Vec::new(),
            functions: Vec::new(),
            closed: Vec::new(),
            open_why: Vec::new(),
            handlers: Vec::new(),
            external: Vec::new(),
            table_edges: Vec::new(),
            unknown_handlers: Default::default(),
            unread: Vec::new(),
            size: image_size,
            stats: TargetStats::default(),
            image: true,
            gaps: Default::default(),
        };
        let first = optional + optional_size;
        for i in 0..sections_count {
            let s = first + i * 40;
            let (size, rva, flags) = (
                u64::from(u32_at(s + 8)),
                u64::from(u32_at(s + 12)),
                u32_at(s + 36),
            );
            if flags & IMAGE_SCN_MEM_EXECUTE != 0 {
                text.sections.push((base + rva, base + rva + size));
            }
        }
        // .pdata
        let (pdata, pdata_size) = directory(3);
        for k in 0..pdata_size / 12 {
            let e = base + pdata + k * 12;
            text.functions.push((
                base + u64::from(u32_at(e)),
                base + u64::from(u32_at(e + 4)),
                base + u64::from(u32_at(e + 8)),
            ));
        }
        text.functions.sort_unstable();
        text.stats.functions = text.functions.len();
        // A function whose exception handler can resume it in the middle
        // (where the scan cannot read where) is not closed.
        let mut targets: Vec<u64> = text.functions.iter().map(|f| f.0).collect();
        text.handlers = text
            .functions
            .iter()
            .map(|f| {
                let h = read_handler(base, f.2, (f.0, f.1), &mut targets);
                if h == Handler::Unknown {
                    *text
                        .unknown_handlers
                        .entry(UNKNOWN_HANDLER.with(|c| c.get()))
                        .or_default() += 1;
                }
                h
            })
            .collect();
        // Everything so far is entered from outside the flow; so are
        // relocation pointers and taken addresses, added below.
        let mut external = targets.clone();
        let mut edges: Vec<(u64, u64)> = Vec::new();
        text.closed = text
            .handlers
            .iter()
            .map(|h| *h != Handler::Unknown)
            .collect();
        text.open_why = vec![0; text.functions.len()];
        // .reloc: absolute pointers into code.
        let (reloc, reloc_size) = directory(5);
        let mut at = base + reloc;
        while reloc_size != 0 && at + 8 <= base + reloc + reloc_size {
            let (page, block) = (u64::from(u32_at(at)), u64::from(u32_at(at + 4)));
            if block < 8 {
                break;
            }
            for k in 0..(block - 8) / 2 {
                let entry = u16_at(at + 8 + k * 2);
                // IMAGE_REL_BASED_DIR64
                if entry >> 12 == 10 {
                    let where_ = base + page + u64::from(entry & 0xFFF);
                    if where_ + 8 <= base + image_size {
                        let value = u64_at(where_);
                        if text.contains(value) {
                            targets.push(value);
                            external.push(value);
                            text.stats.relocations += 1;
                        }
                    }
                }
            }
            at += block;
        }
        // Indirect jumps the first pass could not read: (function, the
        // instructions before, the jump).
        let mut pending: Vec<(usize, Vec<Instruction>, Instruction)> = Vec::new();
        // Functions another layer's patch jumps out of.
        let mut foreign: Vec<usize> = Vec::new();
        for &(start, end) in &text.sections.clone() {
            // SAFETY: an executable section of the loaded image.
            let bytes =
                unsafe { std::slice::from_raw_parts(start as *const u8, (end - start) as usize) };
            let mut decoder = Decoder::with_ip(64, bytes, start, DecoderOptions::NONE);
            let mut window: Vec<Instruction> = Vec::with_capacity(16);
            let mut fi = 0usize;
            let _ = &pending;
            // RIP-relative addresses each register was last given with `lea`
            // in this function (in layout order, not flow order: a table's
            // entries are checked to land in the function, and reading too
            // many only adds targets, which is the safe side).
            let mut values: [Option<u64>; 16] = [None; 16];
            let mut factory = iced_x86::InstructionInfoFactory::new();
            let mut current = usize::MAX;
            for instr in &mut decoder {
                while fi < text.functions.len() && text.functions[fi].1 <= instr.ip() {
                    fi += 1;
                }
                if fi != current {
                    current = fi;
                    values = [None; 16];
                }
                let inside = text.functions.get(fi).is_some_and(|f| f.0 <= instr.ip());
                match instr.flow_control() {
                    FlowControl::ConditionalBranch
                    | FlowControl::UnconditionalBranch
                    | FlowControl::Call
                        if instr.op_kind(0) == OpKind::NearBranch64 =>
                    {
                        targets.push(instr.near_branch_target());
                        text.stats.direct += 1;
                        // Out of the image: another layer's patch (VIctory
                        // patches some of the title's stores and reads to
                        // jump to stubs of its own), whose stub comes back
                        // after the bytes it replaced, which the jump may
                        // not fill: what follows it is entered from outside,
                        // and the rest of the function may not decode as it
                        // runs. Nothing of it moves.
                        if !text.in_image(instr.near_branch_target()) {
                            targets.push(instr.next_ip());
                            external.push(instr.next_ip());
                            if inside {
                                foreign.push(fi);
                            }
                        }
                    }
                    FlowControl::IndirectBranch => {
                        let table = text.jump_table(&window, &values, &instr, false);
                        let found = table.len();
                        if found > 0 {
                            text.stats.jump_tables += 1;
                            text.stats.table_entries += found;
                            edges.extend(table.iter().map(|&t| (instr.ip(), t)));
                            targets.extend(table);
                        }
                        // A jump through a pointer in data (a tail call
                        // through the import table) leaves the function; a
                        // jump through a register whose table was not read
                        // may come back into it anywhere.
                        // So does an indirect jump with REX.W: the Windows
                        // x64 ABI marks an indirect jump that ends an epilog
                        // (a tail call through a pointer or vtable) that
                        // way, and compilers leave it off jumps within a
                        // function, which the unwinder would take for an
                        // epilog otherwise.
                        let tail_call = (instr.op_kind(0) == OpKind::Memory
                            && instr.is_ip_rel_memory_operand())
                            || rex_w(&instr);
                        if found == 0 && !tail_call && inside {
                            pending.push((fi, window.clone(), instr));
                        }
                    }
                    _ => {}
                }
                if instr.is_ip_rel_memory_operand() && text.contains(instr.ip_rel_memory_address())
                {
                    targets.push(instr.ip_rel_memory_address());
                    external.push(instr.ip_rel_memory_address());
                    text.stats.address_taken += 1;
                }
                if instr.code() == Code::Mov_r64_imm64 && text.contains(instr.immediate64()) {
                    targets.push(instr.immediate64());
                    external.push(instr.immediate64());
                    text.stats.address_taken += 1;
                }
                // What this instruction does to the tracked values.
                if instr.code() == Code::Lea_r64_m && instr.is_ip_rel_memory_operand() {
                    values[instr.op0_register().number()] = Some(instr.ip_rel_memory_address());
                } else {
                    for u in factory.info(&instr).used_registers() {
                        let r = u.register().full_register();
                        if r.is_gpr64()
                            && !matches!(
                                u.access(),
                                iced_x86::OpAccess::Read
                                    | iced_x86::OpAccess::CondRead
                                    | iced_x86::OpAccess::NoMemAccess
                            )
                        {
                            values[r.number()] = None;
                        }
                    }
                    if instr.flow_control() == FlowControl::Call
                        || instr.flow_control() == FlowControl::IndirectCall
                    {
                        // Volatile registers do not survive a call.
                        for n in [0usize, 1, 2, 8, 9, 10, 11] {
                            values[n] = None;
                        }
                    }
                }
                if window.len() == 16 {
                    window.remove(0);
                }
                window.push(instr);
            }
        }
        external.sort_unstable();
        external.dedup();
        text.external = external;
        // Read what the first pass could not, by following control flow
        // back from each jump. Optimistic first: a jump not read yet adds no
        // edges, so its targets look unreachable. Each table read adds
        // edges; repeat until nothing new is read. Then check every reading
        // against the final flow, dropping any that no longer holds (which
        // removes its edges), until the readings agree with the flow they
        // were read in. A function keeps any jump left unread: not closed.
        let no_values = [None; 16];
        let base_edges = edges.clone();
        let mut read: Vec<Option<Vec<u64>>> = vec![None; pending.len()];
        let rebuild = |text: &mut Text, read: &[Option<Vec<u64>>]| {
            let mut e = base_edges.clone();
            for ((_, _, jmp), t) in pending.iter().zip(read) {
                if let Some(t) = t {
                    e.extend(t.iter().map(|&x| (jmp.ip(), x)));
                }
            }
            text.table_edges = e;
        };
        rebuild(&mut text, &read);
        loop {
            let mut changed = false;
            for (k, (_, window, jmp)) in pending.iter().enumerate() {
                if read[k].is_none() {
                    let t = text.jump_table(window, &no_values, jmp, true);
                    if !t.is_empty() {
                        targets.extend(t.iter().copied());
                        read[k] = Some(t);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
            rebuild(&mut text, &read);
        }
        for _ in 0..16 {
            let mut changed = false;
            for (k, (_, window, jmp)) in pending.iter().enumerate() {
                if read[k].is_some() {
                    let t = text.jump_table(window, &no_values, jmp, true);
                    let now = if t.is_empty() { None } else { Some(t) };
                    if now != read[k] {
                        if let Some(t) = &now {
                            targets.extend(t.iter().copied());
                        }
                        read[k] = now;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
            rebuild(&mut text, &read);
        }
        // Every target ever read stays a target: more targets only make
        // relocation more cautious.
        targets.sort_unstable();
        targets.dedup();
        text.targets = targets;
        for ((fi, window, jmp), t) in pending.iter().zip(&read) {
            match t {
                Some(t) => {
                    text.stats.jump_tables += 1;
                    text.stats.table_entries += t.len();
                }
                None => {
                    text.unread.push((window.clone(), *jmp));
                    if text.closed[*fi] {
                        text.closed[*fi] = false;
                        text.open_why[*fi] = jmp.ip();
                    }
                }
            }
        }
        for fi in foreign {
            text.closed[fi] = false;
            text.open_why[fi] = FOREIGN;
        }
        Some(text)
    }

    /// Reads the jump table an indirect `jmp reg` uses, if `window` (the
    /// instructions before it) shows one. Returns the entries found.
    fn jump_table(
        &self,
        window: &[Instruction],
        values: &[Option<u64>; 16],
        jmp: &Instruction,
        reach: bool,
    ) -> Vec<u64> {
        let mut targets = Vec::new();
        if jmp.op_kind(0) != OpKind::Register || jmp.mnemonic() != Mnemonic::Jmp {
            return targets;
        }
        let r = jmp.op0_register().full_register();
        // add r, base
        let Some(add_at) = window.iter().rposition(|i| {
            i.mnemonic() == Mnemonic::Add
                && i.op_kind(0) == OpKind::Register
                && i.op0_register().full_register() == r
                && i.op_kind(1) == OpKind::Register
        }) else {
            return targets;
        };
        let base_reg = window[add_at].op1_register().full_register();
        // The base's value: a RIP-relative lea, nearby or earlier in the
        // function (MSVC loads the image base once at the top).
        let near = window[..add_at]
            .iter()
            .rev()
            .find(|i| {
                i.code() == Code::Lea_r64_m
                    && i.op0_register() == base_reg
                    && i.is_ip_rel_memory_operand()
            })
            .map(|i| i.ip_rel_memory_address());
        let bases: Vec<u64> = match near.or(if base_reg.is_gpr64() {
            values[base_reg.number()]
        } else {
            None
        }) {
            Some(b) => vec![b],
            // Not in layout order: follow the function's control flow back
            // from the jump (reaching definitions of the base register).
            None if !reach => return targets,
            None => match self
                .function_of(jmp.ip())
                .and_then(|f| reaching_leas(self, f, window[add_at].ip(), base_reg))
            {
                Some(found) => found,
                None => return targets,
            },
        };
        for base in bases {
            self.read_table(window, add_at, base, base_reg, r, jmp, &mut targets);
        }
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    /// Reads one jump table given its base's value.
    #[allow(clippy::too_many_arguments)]
    fn read_table(
        &self,
        window: &[Instruction],
        add_at: usize,
        base: u64,
        base_reg: iced_x86::Register,
        r: iced_x86::Register,
        jmp: &Instruction,
        targets: &mut Vec<u64>,
    ) -> usize {
        // The load of r: [b + i*4 + disp], b holding the base.
        let Some(load) = window[..add_at].iter().rev().find(|i| {
            matches!(i.mnemonic(), Mnemonic::Movsxd | Mnemonic::Mov)
                && i.op_kind(0) == OpKind::Register
                && i.op0_register().full_register() == r
                && i.op_kind(1) == OpKind::Memory
                && i.memory_index_scale() == 4
                && i.memory_base().full_register() == base_reg
        }) else {
            return 0;
        };
        let table = base.wrapping_add(load.memory_displacement64());
        let signed = load.mnemonic() == Mnemonic::Movsxd;
        let index = load.memory_index().full_register();
        // The bound: cmp index, n then ja (n + 1 entries) or jae (n).
        let mut bound = None;
        for (k, i) in window.iter().enumerate().rev() {
            if i.mnemonic() == Mnemonic::Cmp
                && i.op_kind(0) == OpKind::Register
                && i.op0_register().full_register() == index
            {
                if let Some(next) = window.get(k + 1) {
                    let n = i.immediate(1);
                    bound = match next.mnemonic() {
                        Mnemonic::Ja => Some(n + 1),
                        Mnemonic::Jae => Some(n),
                        _ => None,
                    };
                }
                break;
            }
        }
        let function = self.function_of(jmp.ip());
        let mut found = 0;
        for k in 0..bound.unwrap_or(1024).min(4096) {
            let at = table + k * 4;
            if !self.readable(at, 4) {
                break;
            }
            let entry = u32_at(at);
            let target = if signed {
                base.wrapping_add(entry as i32 as i64 as u64)
            } else {
                base.wrapping_add(u64::from(entry))
            };
            let fits = match (bound, function) {
                (Some(_), _) => self.contains(target),
                (None, Some((s, e))) => target >= s && target < e,
                (None, None) => false,
            };
            if !fits {
                break;
            }
            targets.push(target);
            found += 1;
        }
        found
    }

    fn readable(&self, at: u64, bytes: u64) -> bool {
        self.image && at >= self.base && at + bytes <= self.base + self.size
    }

    pub fn contains(&self, ip: u64) -> bool {
        self.sections.iter().any(|&(s, e)| ip >= s && ip < e)
    }

    pub fn section_end(&self, ip: u64) -> u64 {
        self.sections
            .iter()
            .find(|&&(s, e)| ip >= s && ip < e)
            .map_or(ip, |s| s.1)
    }

    fn function_of(&self, ip: u64) -> Option<(u64, u64)> {
        self.runtime_function(ip).map(|f| (f.0, f.1))
    }

    /// The `.pdata` entry (begin, end, unwind data) of the function at `ip`.
    pub fn runtime_function(&self, ip: u64) -> Option<(u64, u64, u64)> {
        let i = self.functions.partition_point(|f| f.0 <= ip);
        let f = *self.functions.get(i.checked_sub(1)?)?;
        (ip < f.1).then_some(f)
    }

    /// Whether relocating code at `ip` is safe by what the scan saw: `ip`
    /// is in a function from `.pdata` whose every indirect jump was read
    /// (a jump table) or leaves it, without an exception handler that could
    /// resume it in the middle. Then the only ways into its middle are
    /// branches the scan collected. Left out still: a thread context saved
    /// and restored from outside (a debugger, a rewinding watchdog).
    pub fn relocation_proven(&self, ip: u64) -> bool {
        let i = self.functions.partition_point(|f| f.0 <= ip);
        i.checked_sub(1)
            .is_some_and(|i| ip < self.functions[i].1 && self.closed[i])
    }

    /// A leaf function around `ip` that has no `.pdata` entry (MSVC gives
    /// none to functions that never touch the stack): from the last known
    /// branch target at or before `ip` that follows `int3` padding (so the
    /// decode starts in step with the code) to the `ret` or jump that
    /// padding follows. None unless the decode reaches `ip` exactly and the
    /// range has no indirect jump but tail calls: then the same reasoning as
    /// for a closed function applies.
    pub fn leaf_piece(&self, ip: u64) -> Option<(u64, u64)> {
        if !self.image || self.function_of(ip).is_some() {
            return None;
        }
        let &(section_start, section_end) =
            self.sections.iter().find(|&&(s, e)| ip >= s && ip < e)?;
        let i = self.functions.partition_point(|f| f.0 <= ip);
        let lo = i
            .checked_sub(1)
            .map_or(section_start, |i| self.functions[i].1)
            .max(section_start);
        let hi = self
            .functions
            .get(i)
            .map_or(section_end, |f| f.0)
            .min(section_end);
        let byte = |at: u64| unsafe { *(at as *const u8) };
        let k = self.targets.partition_point(|&t| t <= ip);
        let entry = self.targets[..k]
            .iter()
            .rev()
            .take_while(|&&t| t >= lo)
            .find(|&&t| t > lo && byte(t - 1) == 0xCC)
            .copied()?;
        let mut at = entry;
        let mut reached = false;
        while at < hi {
            // SAFETY: inside an executable section.
            let bytes =
                unsafe { std::slice::from_raw_parts(at as *const u8, (hi - at).min(15) as usize) };
            let i = Decoder::with_ip(64, bytes, at, DecoderOptions::NONE).decode();
            if i.is_invalid() || i.flow_control() == FlowControl::Interrupt {
                return None;
            }
            reached |= i.ip() == ip;
            if i.flow_control() == FlowControl::IndirectBranch
                && !((i.op_kind(0) == OpKind::Memory && i.is_ip_rel_memory_operand()) || rex_w(&i))
            {
                return None;
            }
            at = i.next_ip();
            let ends = matches!(
                i.flow_control(),
                FlowControl::Return
                    | FlowControl::UnconditionalBranch
                    | FlowControl::IndirectBranch
            );
            if ends && (at >= hi || byte(at) == 0xCC) {
                return reached.then_some((entry, at));
            }
        }
        None
    }

    /// `leaf_piece`, when moving code inside it is as safe as in a closed
    /// function: nothing in the whole stretch without `.pdata` around it
    /// (between the functions on either side) can jump to an address the
    /// scan did not see. A piece is inferred from padding, so it may be part
    /// of a larger leaf whose `switch` lies before an `int3` inside it; the
    /// stretch is decoded from its start, past runs of `int3`, and an
    /// indirect jump that is not a tail call, or bytes that do not decode,
    /// anywhere in it refuses every piece in it.
    pub fn leaf_proven(&self, ip: u64) -> Option<(u64, u64)> {
        let piece = self.leaf_piece(ip)?;
        let &(section_start, section_end) =
            self.sections.iter().find(|&&(s, e)| ip >= s && ip < e)?;
        let i = self.functions.partition_point(|f| f.0 <= ip);
        let lo = i
            .checked_sub(1)
            .map_or(section_start, |i| self.functions[i].1)
            .max(section_start);
        let hi = self
            .functions
            .get(i)
            .map_or(section_end, |f| f.0)
            .min(section_end);
        let mut gaps = self.gaps.lock().unwrap();
        let clean = *gaps
            .entry((lo, hi))
            .or_insert_with(|| self.stretch_clean(lo, hi));
        clean.then_some(piece)
    }

    fn stretch_clean(&self, lo: u64, hi: u64) -> bool {
        let mut at = lo;
        while at < hi {
            // SAFETY: inside an executable section.
            if unsafe { *(at as *const u8) } == 0xCC {
                at += 1;
                continue;
            }
            // SAFETY: as above.
            let bytes =
                unsafe { std::slice::from_raw_parts(at as *const u8, (hi - at).min(15) as usize) };
            let i = Decoder::with_ip(64, bytes, at, DecoderOptions::NONE).decode();
            if i.is_invalid() {
                return false;
            }
            if i.flow_control() == FlowControl::IndirectBranch
                && !((i.op_kind(0) == OpKind::Memory && i.is_ip_rel_memory_operand()) || rex_w(&i))
            {
                return false;
            }
            // Another layer's patch (see `read`).
            if matches!(
                i.flow_control(),
                FlowControl::UnconditionalBranch | FlowControl::Call
            ) && i.op_kind(0) == OpKind::NearBranch64
                && !self.in_image(i.near_branch_target())
            {
                return false;
            }
            at = i.next_ip();
        }
        true
    }

    /// Whether `at` is inside the image (not only its code).
    pub fn in_image(&self, at: u64) -> bool {
        !self.image || (at >= self.base && at - self.base < self.size)
    }

    /// Short reason relocation at `ip` is not proven, for counting.
    pub fn unproven_kind(&self, ip: u64) -> &'static str {
        if !self.image {
            return "generated code";
        }
        let i = self.functions.partition_point(|f| f.0 <= ip);
        match i.checked_sub(1).filter(|&i| ip < self.functions[i].1) {
            None => "no .pdata (leaf or assembly)",
            Some(i) if self.closed[i] => "proven",
            Some(i) if self.open_why[i] == 0 => "exception handler not understood",
            Some(i) if self.open_why[i] == FOREIGN => "another layer's patch jumps out of it",
            Some(_) => "indirect jump not read",
        }
    }

    /// Why relocation at `ip` is not proven, for diagnostics.
    pub fn why_not_proven(&self, ip: u64) -> String {
        if !self.image {
            return "not an image (generated code)".into();
        }
        let i = self.functions.partition_point(|f| f.0 <= ip);
        match i.checked_sub(1).filter(|&i| ip < self.functions[i].1) {
            None => "no .pdata entry (hand-written assembly?)".into(),
            Some(i) if self.closed[i] => "proven".into(),
            Some(i) if self.open_why[i] == 0 => format!(
                "its exception handler ({:?}) could resume it anywhere",
                self.handlers[i]
            ),
            Some(i) if self.open_why[i] == FOREIGN => {
                "another layer's patch jumps out of it".into()
            }
            Some(i) => format!(
                "unread indirect jump at +{:#x} in the function",
                self.open_why[i] - self.functions[i].0
            ),
        }
    }

    /// Whether `ip` is entered from outside the flow the scan follows.
    fn is_external(&self, ip: u64) -> bool {
        self.external.binary_search(&ip).is_ok()
    }

    /// Whether the scan knows of a branch to `ip`.
    pub fn is_target(&self, ip: u64) -> bool {
        self.targets.binary_search(&ip).is_ok()
    }

    /// Whether something branches to an address in `(from, to)`.
    pub fn targets_between(&self, from: u64, to: u64) -> bool {
        let i = self.targets.partition_point(|&t| t <= from);
        self.targets.get(i).is_some_and(|&t| t < to)
    }

    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Five bytes of `int3` padding within a `jmp rel8` of `from` (the
    /// address after the site's two-byte jump): outside every function,
    /// not a target, not taken by another island.
    pub fn island_near(&self, from: u64, taken: &std::collections::BTreeSet<u64>) -> Option<u64> {
        if !self.image {
            return None;
        }
        let (lo, hi) = (from.saturating_sub(128), from + 127 - 4);
        let mut best: Option<u64> = None;
        let mut at = lo;
        while at <= hi {
            if !self.contains(at) || !self.contains(at + 4) {
                at += 1;
                continue;
            }
            // SAFETY: inside an executable section.
            let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, 5) };
            let free = bytes.iter().all(|&b| b == 0xCC)
                && self.function_of(at).is_none()
                && self.function_of(at + 4).is_none()
                && !self.targets_between(at.wrapping_sub(1), at + 5)
                && taken.range(at.saturating_sub(4)..at + 5).next().is_none();
            if free && best.is_none_or(|b| at.abs_diff(from) < b.abs_diff(from)) {
                best = Some(at);
            }
            at += 1;
        }
        best
    }
}
