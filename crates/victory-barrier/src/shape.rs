//! Instruction sequences rendered without anything specific to the code
//! they came from, so a survey of an image nobody can share can still say
//! what the scan fails on.
//!
//! Registers are renamed in order of first use (`r0`, `r1`, ... with their
//! width: `r0d` is a 32-bit use; `v0x`/`v0y`/`v0z` for vector registers);
//! rsp and rip keep their names, since they mean something structural.
//! Immediates and displacements up to 255 stay (table bounds, scales,
//! small offsets); larger ones become `imm`/`disp`. RIP-relative operands
//! become `[rip+x]`; branches show only their direction and whether they
//! leave the function. A `*` marks an instruction something branches to,
//! `>` the store the shape is about.

use crate::text::Text;
use iced_x86::{Instruction, MemorySize, OpKind, Register};
use std::collections::HashMap;
use std::fmt::Write;

fn width(r: Register) -> &'static str {
    if r.is_gpr64() {
        "q"
    } else if r.is_gpr32() {
        "d"
    } else if r.is_gpr16() {
        "w"
    } else if r.is_gpr8() {
        "b"
    } else if r.is_xmm() {
        "x"
    } else if r.is_ymm() {
        "y"
    } else if r.is_zmm() {
        "z"
    } else if r.is_k() {
        "k"
    } else {
        "?"
    }
}

struct Names(HashMap<Register, usize>, HashMap<Register, usize>);

impl Names {
    fn reg(&mut self, r: Register) -> String {
        match r {
            Register::None => String::new(),
            Register::RSP | Register::ESP => "rsp".into(),
            Register::RIP => "rip".into(),
            Register::ES
            | Register::CS
            | Register::SS
            | Register::DS
            | Register::FS
            | Register::GS => format!("{r:?}").to_lowercase(),
            _ => {
                let vector = r.is_vector_register() || r.is_k();
                let full = r.full_register();
                let map = if vector { &mut self.1 } else { &mut self.0 };
                let n = map.len();
                let k = *map.entry(full).or_insert(n);
                format!("{}{k}{}", if vector { "v" } else { "r" }, width(r))
            }
        }
    }
}

fn number(v: u64, word: &str) -> String {
    if v <= 255 {
        format!("{v:#x}")
    } else {
        word.into()
    }
}

/// Renders `list` (in order) as one anonymised shape.
pub fn render(list: &[Instruction], text: &Text, store: Option<u64>) -> String {
    let mut names = Names(HashMap::new(), HashMap::new());
    let mut out = String::new();
    for i in list {
        // `>` the store, `*` a branch target, both for a store that is one.
        let target = if text.is_target(i.ip()) { '*' } else { ' ' };
        let mark = if Some(i.ip()) == store { '>' } else { ' ' };
        let _ = write!(out, "{mark}{target} {:?}", i.mnemonic());
        if i.has_lock_prefix() {
            out.push_str(" (lock)");
        }
        if i.has_rep_prefix() {
            out.push_str(" (rep)");
        }
        for k in 0..i.op_count() {
            out.push_str(if k == 0 { " " } else { ", " });
            match i.op_kind(k) {
                OpKind::Register => out.push_str(&names.reg(i.op_register(k))),
                OpKind::Memory => {
                    if i.is_ip_rel_memory_operand() {
                        out.push_str("[rip+x]");
                    } else {
                        let mut parts = Vec::new();
                        if i.memory_base() != Register::None {
                            parts.push(names.reg(i.memory_base()));
                        }
                        if i.memory_index() != Register::None {
                            parts.push(format!(
                                "{}*{}",
                                names.reg(i.memory_index()),
                                i.memory_index_scale()
                            ));
                        }
                        if i.memory_displacement64() != 0 {
                            let d = i.memory_displacement64() as i64;
                            parts.push(if d < 0 {
                                format!("-{}", number(d.unsigned_abs(), "disp"))
                            } else {
                                number(d as u64, "disp")
                            });
                        }
                        let size = match i.memory_size() {
                            MemorySize::Unknown => String::new(),
                            m => format!("{}:", m.size()),
                        };
                        let _ = write!(out, "{size}[{}]", parts.join("+"));
                    }
                }
                OpKind::NearBranch64 => {
                    let t = i.near_branch_target();
                    let dir = if t <= i.ip() { "back" } else { "fwd" };
                    let inside = text
                        .runtime_function(i.ip())
                        .is_some_and(|f| t >= f.0 && t < f.1);
                    let _ = write!(out, "{dir}{}", if inside { "" } else { "(out)" });
                }
                OpKind::Immediate8
                | OpKind::Immediate16
                | OpKind::Immediate32
                | OpKind::Immediate64
                | OpKind::Immediate8to16
                | OpKind::Immediate8to32
                | OpKind::Immediate8to64
                | OpKind::Immediate32to64 => {
                    out.push_str(&number(i.immediate(k), "imm"));
                }
                other => {
                    let _ = write!(out, "{other:?}");
                }
            }
        }
        out.push('\n');
    }
    out
}
