//! Which faulting instructions a barrier can stand in for, and how.

use iced_x86::{
    Code, Instruction, InstructionInfoFactory, MemorySize, Mnemonic, OpAccess, OpKind, Register,
};

/// Why an instruction is left to the guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Refusal {
    /// The fault is not in code the patcher may change (the host's own,
    /// or a trampoline: a relocated store that was not given a barrier).
    NotPatchable,
    Undecodable,
    /// No memory operand that the instruction writes.
    NotAStore,
    /// Gather/scatter addressing: one operand, many addresses.
    Vsib,
    /// fs: its base is not known to the barrier.
    Segment,
    /// 32-bit addressing.
    AddressSize,
    /// Uses the stack pointer, which the barrier's pushes move (`push`,
    /// `pop [mem]`, `call [mem]`, and stores through `rsp`).
    Stack,
    /// Fewer than four free general registers.
    NoScratch,
    /// A store whose size is not known, or over 4 KiB.
    Size,
    /// Relocation was needed and refused: the bytes after the store are a
    /// branch target, or not five bytes to the next control transfer.
    Span,
    /// Another thread is stopped inside the bytes to overwrite, not at an
    /// instruction start. Retried at the next fault.
    ThreadInSpan,
    /// The trampoline is out of reach or the encoder refused it.
    Encode,
    /// No room for trampolines within reach.
    Full,
    /// Only an `int3` could reach the trampoline, and `int3` sites are not
    /// asked for (`Config::int3`): a trap at every execution can cost more
    /// than the guard's fault, which comes once a block per refresh.
    Int3Off,
}

/// How the barrier computes the address the title meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    /// `lea` of the instruction's memory operand.
    Operand,
    /// `lea` of the operand plus the `gs` base (the TEB, read at `gs:[0x30]`).
    Gs,
    /// RIP-relative: known when patching.
    Absolute(u64),
    /// In a register the instruction reads implicitly (`clzero` uses rax,
    /// `maskmovdqu` rdi), rounded down to `1 << align` bytes.
    Implicit { reg: Register, align: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct Inline {
    pub size: u32,
    pub scratch: [Register; 4],
    pub address: Address,
    /// `bts`/`btr`/`btc` with a register bit offset: the word written is at
    /// `address + (offset >> log2(bits)) * bytes`, which can be anywhere
    /// within ±256 MiB, not at the operand.
    pub bit_offset: Option<(Register, u32)>,
}

/// How a barrier does the store.
#[derive(Clone, Copy, Debug)]
pub enum Store {
    /// Re-encoded with its memory operand on the barrier's registers.
    Inline(Inline),
    /// `movs`/`stos`, with or without `rep`: done by a host function.
    String {
        kind: StringKind,
        element: u8,
        rep: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StringKind {
    Movs = 0,
    Stos = 1,
}

/// Registers the barrier can borrow, most preferred first (the volatile
/// ones compilers hold the fewest long-lived values in).
const SCRATCH: [Register; 15] = [
    Register::R11,
    Register::R10,
    Register::R9,
    Register::R8,
    Register::RAX,
    Register::RCX,
    Register::RDX,
    Register::RBX,
    Register::RSI,
    Register::RDI,
    Register::R12,
    Register::R13,
    Register::R14,
    Register::R15,
    // Last: often the frame pointer. Pushed and popped like the others, and
    // described in the trampoline's unwind data, so borrowing it is safe; it
    // is what lets a high-byte store with two address registers
    // (`add [rdx+rdi+13h], ah`) find four legacy scratch registers.
    Register::RBP,
];

fn writes(access: OpAccess) -> bool {
    matches!(
        access,
        OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite
    )
}

fn bit_string_operand_bytes(code: Code) -> Option<u32> {
    use Code::*;
    match code {
        Bts_rm16_r16 | Btr_rm16_r16 | Btc_rm16_r16 => Some(2),
        Bts_rm32_r32 | Btr_rm32_r32 | Btc_rm32_r32 => Some(4),
        Bts_rm64_r64 | Btr_rm64_r64 | Btc_rm64_r64 => Some(8),
        _ => None,
    }
}

pub fn classify(
    instr: &Instruction,
    factory: &mut InstructionInfoFactory,
    xsave_bytes: u32,
) -> Result<Store, Refusal> {
    if instr.is_invalid() {
        return Err(Refusal::Undecodable);
    }
    let string = match instr.code() {
        Code::Movsb_m8_m8 => Some((StringKind::Movs, 1)),
        Code::Movsw_m16_m16 => Some((StringKind::Movs, 2)),
        Code::Movsd_m32_m32 => Some((StringKind::Movs, 4)),
        Code::Movsq_m64_m64 => Some((StringKind::Movs, 8)),
        Code::Stosb_m8_AL => Some((StringKind::Stos, 1)),
        Code::Stosw_m16_AX => Some((StringKind::Stos, 2)),
        Code::Stosd_m32_EAX => Some((StringKind::Stos, 4)),
        Code::Stosq_m64_RAX => Some((StringKind::Stos, 8)),
        _ => None,
    };
    if let Some((kind, element)) = string {
        if instr.op_kind(0) != OpKind::MemoryESRDI {
            return Err(Refusal::AddressSize);
        }
        if instr.has_repne_prefix() {
            return Err(Refusal::NotAStore);
        }
        if matches!(instr.segment_prefix(), Register::FS | Register::GS) {
            return Err(Refusal::Segment);
        }
        return Ok(Store::String {
            kind,
            element,
            rep: instr.has_rep_prefix(),
        });
    }
    if instr.is_string_instruction() {
        return Err(Refusal::NotAStore);
    }
    let info = factory.info(instr);
    if instr.code() == Code::Clzeroq {
        // Zeroes the 64-byte line at rax; the decoder lists no memory use.
        let used: Vec<Register> = info
            .used_registers()
            .iter()
            .map(|r| r.register().full_register())
            .collect();
        let mut free = SCRATCH
            .iter()
            .copied()
            .filter(|r| !used.contains(r) && *r != Register::RAX);
        let mut scratch = [Register::None; 4];
        for s in &mut scratch {
            *s = free.next().ok_or(Refusal::NoScratch)?;
        }
        return Ok(Store::Inline(Inline {
            size: 64,
            scratch,
            address: Address::Implicit {
                reg: Register::RAX,
                align: 6,
            },
            bit_offset: None,
        }));
    }
    let used: Vec<Register> = info
        .used_registers()
        .iter()
        .map(|r| r.register().full_register())
        .collect();
    let written: Vec<_> = info
        .used_memory()
        .iter()
        .filter(|m| writes(m.access()))
        .copied()
        .collect();
    let reads_other_memory = info.used_memory().iter().any(|m| !writes(m.access()));
    if written.is_empty() {
        return Err(Refusal::NotAStore);
    }
    if instr.is_vsib() {
        return Err(Refusal::Vsib);
    }
    if instr.is_stack_instruction() || written.len() != 1 || used.contains(&Register::RSP) {
        return Err(Refusal::Stack);
    }
    if reads_other_memory {
        // A second, read operand (`push [mem]` is caught above; nothing else
        // the barrier knows reads one memory operand and writes another).
        return Err(Refusal::NotAStore);
    }
    let explicit = (0..instr.op_count())
        .filter(|&i| instr.op_kind(i) == OpKind::Memory)
        .count();
    let address = if explicit == 1 {
        match instr.segment_prefix() {
            Register::FS => return Err(Refusal::Segment),
            _ if instr.memory_base().is_gpr32() || instr.memory_index().is_gpr32() => {
                return Err(Refusal::AddressSize);
            }
            Register::GS => Address::Gs,
            _ if instr.is_ip_rel_memory_operand() => {
                Address::Absolute(instr.memory_displacement64())
            }
            // No base, no index: an absolute address (moffs, or [disp32]).
            _ if instr.memory_base() == Register::None
                && instr.memory_index() == Register::None =>
            {
                Address::Absolute(instr.memory_displacement64())
            }
            _ => Address::Operand,
        }
    } else {
        match (instr.mnemonic(), instr.op_kind(0)) {
            (Mnemonic::Maskmovdqu | Mnemonic::Vmaskmovdqu, OpKind::MemorySegRDI)
                if instr.segment_prefix() == Register::None =>
            {
                Address::Implicit {
                    reg: Register::RDI,
                    align: 0,
                }
            }
            (Mnemonic::Clzero, _) if matches!(instr.code(), Code::Clzeroq) => Address::Implicit {
                reg: Register::RAX,
                align: 6,
            },
            _ => return Err(Refusal::NotAStore),
        }
    };
    let size = match written[0].memory_size() {
        MemorySize::Xsave | MemorySize::Xsave64 => xsave_bytes as usize,
        _ if matches!(instr.mnemonic(), Mnemonic::Clzero) => 64,
        m => m.size(),
    };
    if size == 0 || size > 4096 {
        return Err(Refusal::Size);
    }
    let bit_offset = match bit_string_operand_bytes(instr.code()) {
        // The register as the instruction names it (dx, edx, rdx): the
        // barrier sign-extends it from that width.
        Some(bytes) if instr.op_kind(0) == OpKind::Memory => Some((instr.op1_register(), bytes)),
        _ => None,
    };
    let (base, index) = (
        instr.memory_base().full_register(),
        instr.memory_index().full_register(),
    );
    // ah, bh, ch, dh cannot be encoded with a REX prefix, which r8-r15 in
    // the re-encoded store's address would need.
    let high_byte = (0..instr.op_count()).any(|k| {
        instr.op_kind(k) == OpKind::Register
            && matches!(
                instr.op_register(k),
                Register::AH | Register::BH | Register::CH | Register::DH
            )
    });
    let mut scratch = [Register::None; 4];
    let mut free = SCRATCH.iter().copied().filter(|r| {
        !used.contains(r) && *r != base && *r != index && !(high_byte && r.number() >= 8)
    });
    for s in &mut scratch {
        *s = free.next().ok_or(Refusal::NoScratch)?;
    }
    Ok(Store::Inline(Inline {
        size: size as u32,
        scratch,
        address,
        bit_offset,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions};

    fn classify_bytes(bytes: &[u8]) -> Result<Store, Refusal> {
        let instr = Decoder::with_ip(64, bytes, 0x1000, DecoderOptions::NONE).decode();
        classify(&instr, &mut InstructionInfoFactory::new(), 2048)
    }

    fn inline(bytes: &[u8]) -> Inline {
        match classify_bytes(bytes) {
            Ok(Store::Inline(i)) => i,
            other => panic!("{bytes:02x?}: {other:?}"),
        }
    }

    #[test]
    fn plain_and_vector_stores_are_inline() {
        // mov [rax+rcx*8+0x10], rdx
        let i = inline(&[0x48, 0x89, 0x54, 0xC8, 0x10]);
        assert_eq!(i.size, 8);
        assert!(
            !i.scratch.contains(&Register::RAX)
                && !i.scratch.contains(&Register::RCX)
                && !i.scratch.contains(&Register::RDX)
        );
        // vmovdqu [rcx], ymm0
        assert_eq!(inline(&[0xC5, 0xFE, 0x7F, 0x01]).size, 32);
        // lock xadd [rcx], eax
        assert_eq!(inline(&[0xF0, 0x0F, 0xC1, 0x01]).size, 4);
        // movntdq [rax], xmm0
        assert_eq!(inline(&[0x66, 0x0F, 0xE7, 0x00]).size, 16);
        // fxsave [rax]
        assert_eq!(inline(&[0x0F, 0xAE, 0x00]).size, 512);
    }

    #[test]
    fn special_addresses() {
        // mov gs:[rax], ecx
        assert_eq!(inline(&[0x65, 0x89, 0x08]).address, Address::Gs);
        // mov [rip+0x10], eax at 0x1000 (6 bytes): 0x1016
        assert_eq!(
            inline(&[0x89, 0x05, 0x10, 0, 0, 0]).address,
            Address::Absolute(0x1016)
        );
        // ds: is a no-op in 64-bit code
        assert_eq!(inline(&[0x3E, 0x89, 0x08]).address, Address::Operand);
        // bts [rcx], rdx
        assert_eq!(
            inline(&[0x48, 0x0F, 0xAB, 0x11]).bit_offset,
            Some((Register::RDX, 8))
        );
        // maskmovdqu xmm0, xmm1
        assert_eq!(
            inline(&[0x66, 0x0F, 0xF7, 0xC1]).address,
            Address::Implicit {
                reg: Register::RDI,
                align: 0
            }
        );
        // clzero
        let c = inline(&[0x0F, 0x01, 0xFC]);
        assert_eq!(
            (c.address, c.size),
            (
                Address::Implicit {
                    reg: Register::RAX,
                    align: 6
                },
                64
            )
        );
    }

    #[test]
    fn high_byte_stores_find_legacy_scratch() {
        // add [rdx+rdi+13h], ah: rax, rdx, rdi in use; rcx, rbx, rsi, rbp left.
        let i = inline(&[0x00, 0x64, 0x3A, 0x13]);
        assert!(i.scratch.iter().all(|r| r.number() < 8), "{:?}", i.scratch);
        assert!(i.scratch.contains(&Register::RBP));
    }

    #[test]
    fn string_stores_go_to_the_helper() {
        assert!(matches!(
            classify_bytes(&[0xF3, 0xA4]),
            Ok(Store::String {
                kind: StringKind::Movs,
                element: 1,
                rep: true
            })
        ));
        assert!(matches!(
            classify_bytes(&[0xF3, 0x48, 0xAB]),
            Ok(Store::String {
                kind: StringKind::Stos,
                element: 8,
                rep: true
            })
        ));
    }

    #[test]
    fn refusals() {
        // mov rax, [rcx]: a load
        assert_eq!(
            classify_bytes(&[0x48, 0x8B, 0x01]).unwrap_err(),
            Refusal::NotAStore
        );
        // mov [rsp+8], rax
        assert_eq!(
            classify_bytes(&[0x48, 0x89, 0x44, 0x24, 0x08]).unwrap_err(),
            Refusal::Stack
        );
        // pop qword [rax]
        assert_eq!(classify_bytes(&[0x8F, 0x00]).unwrap_err(), Refusal::Stack);
        // mov fs:[rax], ecx
        assert_eq!(
            classify_bytes(&[0x64, 0x89, 0x08]).unwrap_err(),
            Refusal::Segment
        );
    }
}
