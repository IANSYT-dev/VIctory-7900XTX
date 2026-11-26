//! Trampolines: the code a patched store site goes to.
//!
//! An inline trampoline is the store itself with the barrier around it (see
//! `layout`), any instructions the site's jump covered (each store among
//! them with a barrier of its own), then a jump back. A string trampoline
//! saves the thread's whole state, calls `string::store` and restores it.
//! Both are encoded with iced's block encoder at their final address, which
//! re-targets relocated branches and RIP-relative operands.
//!
//! Every trampoline first moves `rsp` down 128 bytes, so its pushes do not
//! land on data the title keeps just below its stack pointer. Windows
//! promises nothing there, but in the `red_zone` case its exception
//! dispatch (a guard fault) left the 64 bytes below rsp alone, so the
//! barrier should too.

use crate::classify::{Address, Inline, Refusal, Store};
use crate::layout::{BLOCK_SHR, COARSE_SHR, INDEX_SHL, MID_SHR, PART_SHR, SLOT_SHIFT, Tables};
use crate::string::{self, StringSite};
use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, FlowControl, Instruction, InstructionBlock,
    MemoryOperand, Register,
};

pub const RED_ZONE: i64 = 128;

/// Encoded trampoline code, where each relocated instruction landed, and
/// the instructions with their offsets (for unwind data).
pub struct Encoded {
    pub code: Vec<u8>,
    /// For each instruction of the span after the store: (its old address,
    /// its new one).
    pub moved: Vec<(u64, u64)>,
    pub instructions: Vec<Instruction>,
    pub offsets: Vec<u32>,
}

fn op<T>(r: Result<T, iced_x86::IcedError>) -> Result<T, Refusal> {
    r.map_err(|_| Refusal::Encode)
}

fn mov_imm(list: &mut Vec<Instruction>, r: Register, v: i32) -> Result<(), Refusal> {
    list.push(op(Instruction::with2(Code::Mov_rm64_imm32, r, v))?);
    Ok(())
}

fn rsp_by(list: &mut Vec<Instruction>, by: i64) -> Result<(), Refusal> {
    list.push(op(Instruction::with2(
        Code::Lea_r64_m,
        Register::RSP,
        MemoryOperand::with_base_displ(Register::RSP, by),
    ))?);
    Ok(())
}

/// `out = (a << 28) >> shr`, with `count` clobbered.
fn index(
    list: &mut Vec<Instruction>,
    out: Register,
    a: Register,
    count: Register,
    shr: u32,
) -> Result<(), Refusal> {
    mov_imm(list, count, INDEX_SHL as i32)?;
    list.push(op(Instruction::with3(
        Code::VEX_Shlx_r64_rm64_r64,
        out,
        a,
        count,
    ))?);
    mov_imm(list, count, shr as i32)?;
    list.push(op(Instruction::with3(
        Code::VEX_Shrx_r64_rm64_r64,
        out,
        out,
        count,
    ))?);
    Ok(())
}

/// `out = a >> 36`, with `count` clobbered.
fn slot(
    list: &mut Vec<Instruction>,
    out: Register,
    a: Register,
    count: Register,
) -> Result<(), Refusal> {
    mov_imm(list, count, SLOT_SHIFT as i32)?;
    list.push(op(Instruction::with3(
        Code::VEX_Shrx_r64_rm64_r64,
        out,
        a,
        count,
    ))?);
    Ok(())
}

/// The store `instr` with its barrier.
fn inline_store(
    list: &mut Vec<Instruction>,
    instr: &Instruction,
    s: &Inline,
    tables: Tables,
) -> Result<(), Refusal> {
    let [a, b, c, d] = s.scratch;
    rsp_by(list, -RED_ZONE)?;
    for r in [a, b, c, d] {
        list.push(op(Instruction::with1(Code::Push_r64, r))?);
    }
    // a = the address the title meant.
    let operand = MemoryOperand::new(
        instr.memory_base(),
        instr.memory_index(),
        instr.memory_index_scale(),
        instr.memory_displacement64() as i64,
        if instr.memory_displacement64() == 0 {
            0
        } else {
            8
        },
        false,
        Register::None,
    );
    match s.address {
        Address::Operand => list.push(op(Instruction::with2(Code::Lea_r64_m, a, operand))?),
        Address::Gs => {
            list.push(op(Instruction::with2(Code::Lea_r64_m, a, operand))?);
            let teb = MemoryOperand::new(
                Register::None,
                Register::None,
                1,
                0x30,
                4,
                false,
                Register::GS,
            );
            list.push(op(Instruction::with2(Code::Mov_r64_rm64, c, teb))?);
            list.push(op(Instruction::with2(
                Code::Lea_r64_m,
                a,
                MemoryOperand::with_base_index(a, c),
            ))?);
        }
        Address::Absolute(at) => list.push(op(Instruction::with2(Code::Mov_r64_imm64, a, at))?),
        Address::Implicit { reg, align } => {
            list.push(op(Instruction::with2(Code::Mov_r64_rm64, a, reg))?);
            if align != 0 {
                mov_imm(list, d, align as i32)?;
                list.push(op(Instruction::with3(
                    Code::VEX_Shrx_r64_rm64_r64,
                    a,
                    a,
                    d,
                ))?);
                list.push(op(Instruction::with3(
                    Code::VEX_Shlx_r64_rm64_r64,
                    a,
                    a,
                    d,
                ))?);
            }
        }
    }
    // b = the block's delta.
    slot(list, c, a, d)?;
    list.push(op(Instruction::with2(
        Code::Mov_r64_imm64,
        b,
        tables.delta as u64,
    ))?);
    list.push(op(Instruction::with2(
        Code::Mov_r64_rm64,
        b,
        MemoryOperand::with_base_index_scale(b, c, 8),
    ))?);
    index(list, d, a, c, BLOCK_SHR)?;
    list.push(op(Instruction::with2(
        Code::Mov_r64_rm64,
        b,
        MemoryOperand::with_base_index_scale(b, d, 8),
    ))?);
    // The store, at a + b.
    match s.address {
        // `clzero` as eight plain stores to the line (`a` is aligned): the
        // instruction is AMD's alone and raises #UD on Intel, where a layer
        // emulates it at every pass. No register or flag changes either way.
        Address::Implicit { .. } if instr.code() == Code::Clzeroq => {
            for k in 0..8 {
                let line = MemoryOperand::new(a, b, 1, k * 8, 1, false, Register::None);
                list.push(op(Instruction::with2(Code::Mov_rm64_imm32, line, 0u32))?);
            }
        }
        Address::Implicit { reg, .. } => {
            list.push(op(Instruction::with2(Code::Mov_r64_rm64, d, reg))?);
            list.push(op(Instruction::with2(
                Code::Lea_r64_m,
                reg,
                MemoryOperand::with_base_index(reg, b),
            ))?);
            list.push(*instr);
            list.push(op(Instruction::with2(Code::Mov_r64_rm64, reg, d))?);
        }
        // `mov [moffs], acc` has no form with registers in the address:
        // the same store as `mov [a + b], acc`.
        _ if matches!(
            instr.code(),
            Code::Mov_moffs8_AL
                | Code::Mov_moffs16_AX
                | Code::Mov_moffs32_EAX
                | Code::Mov_moffs64_RAX
        ) =>
        {
            let (code, acc) = match instr.code() {
                Code::Mov_moffs8_AL => (Code::Mov_rm8_r8, Register::AL),
                Code::Mov_moffs16_AX => (Code::Mov_rm16_r16, Register::AX),
                Code::Mov_moffs32_EAX => (Code::Mov_rm32_r32, Register::EAX),
                _ => (Code::Mov_rm64_r64, Register::RAX),
            };
            list.push(op(Instruction::with2(
                code,
                MemoryOperand::with_base_index(a, b),
                acc,
            ))?);
        }
        _ => {
            let mut moved = *instr;
            moved.set_memory_base(a);
            moved.set_memory_index(b);
            moved.set_memory_index_scale(1);
            moved.set_memory_displacement64(0);
            moved.set_memory_displ_size(0);
            moved.set_segment_prefix(Register::None);
            list.push(moved);
        }
    }
    // A bit-string store writes the word its bit offset selects.
    if let Some((reg, bytes)) = s.bit_offset {
        match bytes {
            2 => list.push(op(Instruction::with2(Code::Movsx_r64_rm16, c, reg))?),
            4 => list.push(op(Instruction::with2(Code::Movsxd_r64_rm32, c, reg))?),
            _ => list.push(op(Instruction::with2(Code::Mov_r64_rm64, c, reg))?),
        }
        mov_imm(list, d, (bytes * 8).trailing_zeros() as i32)?;
        list.push(op(Instruction::with3(
            Code::VEX_Sarx_r64_rm64_r64,
            c,
            c,
            d,
        ))?);
        list.push(op(Instruction::with2(
            Code::Lea_r64_m,
            a,
            MemoryOperand::with_base_index_scale(a, c, bytes),
        ))?);
    }
    // The marks go by where the store went: our view's address of the page,
    // the same whichever address the title wrote it through.
    list.push(op(Instruction::with2(
        Code::Lea_r64_m,
        a,
        MemoryOperand::with_base_index(a, b),
    ))?);
    // Mark the first and last byte in `table`'s map at `shr`: a store of up
    // to 4 KiB covers at most two units of each level. `a` is left at the
    // first byte.
    let mark = |list: &mut Vec<Instruction>, table: *const u64, shr: u32| -> Result<(), Refusal> {
        slot(list, c, a, d)?;
        list.push(op(Instruction::with2(
            Code::Mov_r64_imm64,
            b,
            table as u64,
        ))?);
        list.push(op(Instruction::with2(
            Code::Mov_r64_rm64,
            b,
            MemoryOperand::with_base_index_scale(b, c, 8),
        ))?);
        index(list, d, a, c, shr)?;
        list.push(op(Instruction::with2(
            Code::Mov_rm8_imm8,
            MemoryOperand::with_base_index(b, d),
            1u32,
        ))?);
        if s.size > 1 {
            let last = i64::from(s.size) - 1;
            list.push(op(Instruction::with2(
                Code::Lea_r64_m,
                a,
                MemoryOperand::with_base_displ(a, last),
            ))?);
            index(list, d, a, c, shr)?;
            list.push(op(Instruction::with2(
                Code::Mov_rm8_imm8,
                MemoryOperand::with_base_index(b, d),
                1u32,
            ))?);
            list.push(op(Instruction::with2(
                Code::Lea_r64_m,
                a,
                MemoryOperand::with_base_displ(a, -last),
            ))?);
        }
        Ok(())
    };
    // The finest first: who takes them takes the coarsest first.
    mark(list, tables.part, PART_SHR)?;
    mark(list, tables.fine, BLOCK_SHR)?;
    mark(list, tables.mid, MID_SHR)?;
    mark(list, tables.coarse, COARSE_SHR)?;
    for r in [d, c, b, a] {
        list.push(op(Instruction::with1(Code::Pop_r64, r))?);
    }
    rsp_by(list, RED_ZONE)
}

/// Encodes the trampoline for `span` at `ip`. `stores[k]` says how to do
/// `span[k]` if it stores (the first always does): a store the jump covers
/// gets its barrier too, else it would fault here, in the trampoline, where
/// nothing can patch it.
pub fn trampoline(
    ip: u64,
    span: &[Instruction],
    stores: &[Option<Store>],
    tables: Tables,
    site: *const StringSite,
) -> Result<Encoded, Refusal> {
    let mut list = Vec::new();
    // Where each span instruction's code starts in `list`.
    let mut starts = Vec::with_capacity(span.len());
    for (instr, store) in span.iter().zip(stores) {
        starts.push(list.len());
        match store {
            Some(Store::Inline(s)) => inline_store(&mut list, instr, s, tables)?,
            Some(Store::String { .. }) => string_call(&mut list, site)?,
            None => list.push(*instr),
        }
    }
    // Back to the site: `jmp [rip+0]` and the target, which Windows does
    // not take for an epilog (`unwind`). Encoded twice: the second time
    // with the target's real address.
    let last = span.last().unwrap();
    let back = !matches!(
        last.flow_control(),
        FlowControl::UnconditionalBranch | FlowControl::Return
    );
    if back {
        list.push(op(Instruction::with1(
            Code::Jmp_rm64,
            MemoryOperand::with_base_displ(Register::RIP, ip as i64),
        ))?);
        list.push(Instruction::with_declare_qword_1(last.next_ip()));
    }
    let encode = |list: &[Instruction]| {
        op(BlockEncoder::encode(
            64,
            InstructionBlock::new(list, ip),
            BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        ))
    };
    let mut encoded = encode(&list)?;
    if back {
        let target = ip + u64::from(*encoded.new_instruction_offsets.last().unwrap());
        let n = list.len();
        list[n - 2] = op(Instruction::with1(
            Code::Jmp_rm64,
            MemoryOperand::with_base_displ(Register::RIP, target as i64),
        ))?;
        let again = encode(&list)?;
        if again.new_instruction_offsets != encoded.new_instruction_offsets {
            return Err(Refusal::Encode);
        }
        encoded = again;
    }
    let moved = span[1..]
        .iter()
        .zip(&starts[1..])
        .map(|(i, &k)| (i.ip(), ip + u64::from(encoded.new_instruction_offsets[k])))
        .collect();
    Ok(Encoded {
        code: encoded.code_buffer,
        moved,
        instructions: list,
        offsets: encoded.new_instruction_offsets,
    })
}

/// Saves everything, calls `string::store(&saved, site)`, restores.
fn string_call(list: &mut Vec<Instruction>, site: *const StringSite) -> Result<(), Refusal> {
    use Register::*;
    rsp_by(list, -RED_ZONE)?;
    list.push(Instruction::with(Code::Pushfq));
    // The title may have set the direction flag (`std; rep movsb`); host
    // code must run with it clear. popfq puts the title's back.
    list.push(Instruction::with(Code::Cld));
    for r in string::SAVED_PUSH_ORDER {
        list.push(op(Instruction::with1(Code::Push_r64, r))?);
    }
    // rbx keeps the saved state's address across the call.
    list.push(op(Instruction::with2(Code::Mov_r64_rm64, RBX, RSP))?);
    let area = string::xsave_area_bytes();
    list.push(op(Instruction::with2(
        Code::Sub_rm64_imm32,
        RSP,
        area as i32,
    ))?);
    list.push(op(Instruction::with2(Code::And_rm64_imm8, RSP, -64i32))?);
    // XRSTOR faults on a header with stray bits: clear it first.
    for k in 0..8 {
        list.push(op(Instruction::with2(
            Code::Mov_rm64_imm32,
            MemoryOperand::with_base_displ(RSP, 512 + 8 * k),
            0i32,
        ))?);
    }
    list.push(op(Instruction::with2(Code::Mov_r32_imm32, EAX, u32::MAX))?);
    list.push(op(Instruction::with2(Code::Mov_r32_imm32, EDX, u32::MAX))?);
    list.push(op(Instruction::with1(
        Code::Xsave64_mem,
        MemoryOperand::with_base(RSP),
    ))?);
    list.push(op(Instruction::with2(Code::Sub_rm64_imm8, RSP, 32i32))?);
    list.push(op(Instruction::with2(Code::Mov_r64_rm64, RCX, RBX))?);
    list.push(op(Instruction::with2(
        Code::Mov_r64_imm64,
        RDX,
        site as u64,
    ))?);
    list.push(op(Instruction::with2(
        Code::Mov_r64_imm64,
        RAX,
        string::store as *const () as u64,
    ))?);
    list.push(op(Instruction::with1(Code::Call_rm64, RAX))?);
    list.push(op(Instruction::with2(Code::Add_rm64_imm8, RSP, 32i32))?);
    list.push(op(Instruction::with2(Code::Mov_r32_imm32, EAX, u32::MAX))?);
    list.push(op(Instruction::with2(Code::Mov_r32_imm32, EDX, u32::MAX))?);
    list.push(op(Instruction::with1(
        Code::Xrstor64_mem,
        MemoryOperand::with_base(RSP),
    ))?);
    list.push(op(Instruction::with2(Code::Mov_r64_rm64, RSP, RBX))?);
    for r in string::SAVED_PUSH_ORDER.iter().rev() {
        list.push(op(Instruction::with1(Code::Pop_r64, *r))?);
    }
    list.push(Instruction::with(Code::Popfq));
    rsp_by(list, RED_ZONE)
}

/// `jmp rel32` from `from` to `to`, or None if out of reach.
pub fn jump32(from: u64, to: u64) -> Option<[u8; 5]> {
    let rel = to.wrapping_sub(from + 5) as i64;
    let rel = i32::try_from(rel).ok()?;
    let mut b = [0xE9, 0, 0, 0, 0];
    b[1..].copy_from_slice(&rel.to_le_bytes());
    Some(b)
}

/// `jmp rel8` from `from` to `to`, or None if out of reach.
pub fn jump8(from: u64, to: u64) -> Option<[u8; 2]> {
    let rel = to.wrapping_sub(from + 2) as i64;
    Some([0xEB, i8::try_from(rel).ok()? as u8])
}
