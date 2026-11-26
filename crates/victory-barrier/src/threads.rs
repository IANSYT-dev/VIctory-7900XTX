//! Stopping the process's other threads while a site's bytes change.
//!
//! Writing a jump over instructions another core may be executing is only
//! safe if no thread is part-way through them. The patcher suspends every
//! other thread, looks where each one stopped and moves any that stopped on
//! an instruction it relocates to that instruction's copy, then writes and
//! lets them go. Patching happens once per site, so the cost (a few hundred
//! microseconds) does not recur.
//!
//! Between suspending and resuming nothing here allocates: a suspended
//! thread may hold the heap's lock.
//!
//! Threads are listed with `NtGetNextThread`, which walks this process's
//! threads only. A Toolhelp snapshot lists every thread of the system and
//! took 25-33 ms per patch here, which made patching, not storing, the cost
//! of the first frames.

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, GetThreadContext, SetThreadContext,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThreadId, GetThreadId, ResumeThread, SuspendThread,
    THREAD_GET_CONTEXT, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SET_CONTEXT,
    THREAD_SUSPEND_RESUME,
};

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    fn NtGetNextThread(
        process: HANDLE,
        thread: HANDLE,
        access: u32,
        attributes: u32,
        flags: u32,
        next: *mut HANDLE,
    ) -> i32;
}

/// GetThreadContext needs 16-byte alignment, which windows-sys's CONTEXT
/// does not declare.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Context(pub CONTEXT);

pub struct Frozen {
    threads: Vec<(HANDLE, Context, bool)>,
}

impl Frozen {
    /// Suspends every other thread of the process.
    pub fn others() -> Self {
        let me = unsafe { GetCurrentThreadId() };
        let access = THREAD_SUSPEND_RESUME
            | THREAD_GET_CONTEXT
            | THREAD_SET_CONTEXT
            | THREAD_QUERY_LIMITED_INFORMATION;
        let mut handles = Vec::new();
        let mut at: HANDLE = std::ptr::null_mut();
        // SAFETY: walks this process's threads; each handle is kept or closed.
        unsafe {
            loop {
                let mut next: HANDLE = std::ptr::null_mut();
                let status = NtGetNextThread(GetCurrentProcess(), at, access, 0, 0, &mut next);
                if !at.is_null() && handles.last() != Some(&at) {
                    CloseHandle(at);
                }
                if status < 0 {
                    break;
                }
                at = next;
                if GetThreadId(next) != me {
                    handles.push(next);
                }
            }
        }
        // SAFETY: zeroed CONTEXTs are plain data.
        let blank = Context(unsafe { std::mem::zeroed() });
        let mut threads: Vec<(HANDLE, Context, bool)> =
            handles.into_iter().map(|h| (h, blank, false)).collect();
        for (h, ctx, ok) in &mut threads {
            // SAFETY: an open thread handle. GetThreadContext returns once the
            // thread has really stopped.
            unsafe {
                if SuspendThread(*h) != u32::MAX {
                    ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64;
                    *ok = GetThreadContext(*h, &mut ctx.0) != 0;
                    if !*ok {
                        ResumeThread(*h);
                    }
                }
            }
        }
        threads.retain(|t| t.2);
        Self { threads }
    }

    /// Moves the stopped threads `to` names: `to(rip)` is `None` to leave a
    /// thread, `Some(Some(new))` to move it to `new`, `Some(None)` for one
    /// that cannot be moved. How many were moved, or `None` (and none moved)
    /// if one could not be. Allocates nothing: the others are stopped.
    pub fn redirect(&mut self, to: impl Fn(u64) -> Option<Option<u64>>) -> Option<u64> {
        if self.threads.iter().any(|t| to(t.1.0.Rip) == Some(None)) {
            return None;
        }
        let mut moved = 0;
        for (h, ctx, _) in &mut self.threads {
            if let Some(Some(new)) = to(ctx.0.Rip) {
                let was = ctx.0.Rip;
                ctx.0.Rip = new;
                ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64;
                // SAFETY: a suspended thread and its own control context.
                if unsafe { SetThreadContext(*h, &ctx.0) } == 0 {
                    ctx.0.Rip = was;
                    return None;
                }
                moved += 1;
            }
        }
        Some(moved)
    }
}

impl Drop for Frozen {
    fn drop(&mut self) {
        for (h, _, _) in &self.threads {
            // SAFETY: suspended by `others`.
            unsafe {
                ResumeThread(*h);
                CloseHandle(*h);
            }
        }
    }
}
