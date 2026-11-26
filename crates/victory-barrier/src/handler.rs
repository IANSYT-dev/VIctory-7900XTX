//! The breakpoint handler for `int3` sites ([`crate::Strategy::Int3`]).
//!
//! The page faults that find stores to patch are `victory-gpu`'s (its watch
//! asks [`crate::Patcher::on_write_fault`]); this handler only sends a thread
//! that reached one of our `int3`s to the site's trampoline. It is installed
//! first, so the title's own vectored handlers see no breakpoint of ours.

use crate::Patcher;
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::EXCEPTION_BREAKPOINT;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_POINTERS,
};

const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

static PATCHER: OnceLock<&'static Patcher> = OnceLock::new();

unsafe extern "system" fn on_exception(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes a valid record and context.
    let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    let Some(patcher) = PATCHER.get() else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    if record.ExceptionCode != EXCEPTION_BREAKPOINT {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    match patcher.breakpoint(context.Rip) {
        Some(trampoline) => {
            context.Rip = trampoline;
            EXCEPTION_CONTINUE_EXECUTION
        }
        // A breakpoint with no int3 where it says. Seen when another thread
        // set this thread's context (a watchdog rewinding it) while it was
        // trapping on one of our int3s: Windows delivers the breakpoint with
        // the new registers, and backs RIP up one byte as for any breakpoint.
        // The thread was meant to go on at RIP + 1.
        None if patcher.has_breakpoints() && phantom(context.Rip) => {
            context.Rip += 1;
            EXCEPTION_CONTINUE_EXECUTION
        }
        None => EXCEPTION_CONTINUE_SEARCH,
    }
}

/// Whether a breakpoint reported at `rip` has no `int3` (`CC`, or the
/// `CD 03` form) to account for it.
fn phantom(rip: u64) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let readable = unsafe { VirtualQuery((rip - 1) as *const _, &mut m, size_of_val(&m)) } != 0
        && m.State == MEM_COMMIT
        && (m.BaseAddress as u64 + m.RegionSize as u64) > rip;
    if !readable {
        return false;
    }
    // SAFETY: committed code bytes, checked above.
    let (before, at) = unsafe { (*((rip - 1) as *const u8), *(rip as *const u8)) };
    at != 0xCC && before != 0xCD
}

/// Installs the handler for `patcher`'s `int3` sites; once per process.
pub fn install(patcher: &'static Patcher) -> Result<(), String> {
    if PATCHER.set(patcher).is_err() {
        return Err("the breakpoint handler is installed already".into());
    }
    // SAFETY: a handler that lives as long as the process.
    let handle = unsafe { AddVectoredExceptionHandler(1, Some(on_exception)) };
    if handle.is_null() {
        Err("AddVectoredExceptionHandler failed".into())
    } else {
        Ok(())
    }
}
