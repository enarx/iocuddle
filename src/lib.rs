// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(clippy::all)]

use core::convert::TryInto;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr::null;

use std::io::{Error, Result};
use std::os::raw::{c_int, c_uint, c_ulong, c_void};
use std::os::unix::io::AsRawFd;

use zerocopy::{FromBytes, Immutable, IntoBytes};

extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
}

// Platform-specific ioctl encoding constants.
//
// Most architectures use the asm-generic defaults, but several override
// the direction bits and size field width. These values are sourced from
// the kernel's `arch/*/include/uapi/asm/ioctl.h` headers.
//
// See: https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/ioctl.h

/// OSF/1-derived platforms: powerpc, mips, sparc, (alpha — no Rust target)
#[cfg(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "sparc",
    target_arch = "sparc64"
))]
mod platform {
    use std::os::raw::c_ulong;
    pub const SIZEBITS: c_ulong = 13;
    pub const NONE: c_ulong = 1;
    pub const READ: c_ulong = 2;
    pub const WRITE: c_ulong = 4;
}

/// Standard (asm-generic): x86, x86_64, arm, aarch64, riscv, s390x, etc.
///
/// Note: PA-RISC (parisc) has swapped READ/WRITE values but no Rust target
/// exists for it, so it falls into this default. If a Rust parisc target is
/// ever added, it will need its own cfg block (READ=1, WRITE=2).
#[cfg(not(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "sparc",
    target_arch = "sparc64"
)))]
mod platform {
    use std::os::raw::c_ulong;
    pub const SIZEBITS: c_ulong = 14;
    pub const NONE: c_ulong = 0;
    pub const READ: c_ulong = 2;
    pub const WRITE: c_ulong = 1;
}

/// A marker for the read direction
pub struct Read(());

/// A marker for the write direction
pub struct Write(());

/// A marker for the write/read direction
pub struct WriteRead(());

/// A collection of related `ioctl`s
///
/// In the Linux kernel macros, this is called the `ioctl` `type`. We have
/// chosen a distinct name to disambiguate from the `ioctl` argument type.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Group(u8);

impl Group {
    /// Create a new group for related `ioctl`s from its allocated number
    pub const fn new(value: u8) -> Self {
        Self(value)
    }

    // This function implements the _IOC() macro found in the kernel tree at:
    // `include/uapi/asm-generic/ioctl.h`.
    const unsafe fn make<D, T>(self, nr: u8, dir: c_ulong, size: usize) -> Ioctl<D, T> {
        const NR_BITS: usize = 8;
        const TYPE_BITS: usize = 8;
        const SIZE_MASK: c_ulong = (1 << platform::SIZEBITS) - 1;

        let mut req = dir;

        req <<= platform::SIZEBITS;
        req |= size as c_ulong & SIZE_MASK;

        req <<= TYPE_BITS;
        req |= self.0 as c_ulong;

        req <<= NR_BITS;
        req |= nr as c_ulong;

        Ioctl::classic(req)
    }

    /// Define a new `ioctl` with an associated `type`
    ///
    /// This corresponds to the `_IO(type, nr)` macro.
    ///
    /// The `nr` argument is the allocated integer which uniquely
    /// identifies this `ioctl` within the `Group`.
    ///
    /// # Safety
    ///
    /// For safety details, see [Ioctl::classic].
    ///
    /// It is important to note that this function can produce any style of
    /// ioctl. It is in many ways similar to [Ioctl::classic], but with
    /// namespacing.
    pub const unsafe fn none<D, T>(self, nr: u8) -> Ioctl<D, T> {
        self.make(nr, platform::NONE, 0)
    }

    /// Define a new `Read` `ioctl` with an associated `type`
    ///
    /// This corresponds to the `_IOR(type, nr, size)` macro.
    ///
    /// The `nr` argument is the allocated integer which uniquely
    /// identifies this `ioctl` within the `Group`.
    ///
    /// # Safety
    ///
    /// For safety details, see [Ioctl::classic].
    pub const unsafe fn read<'a, T>(self, nr: u8) -> Ioctl<Read, &'a T> {
        self.make(nr, platform::READ, size_of::<T>())
    }

    /// Define a new `Write` `ioctl` with an associated `type`
    ///
    /// This corresponds to the `_IOW(type, nr, size)` macro.
    ///
    /// The `nr` argument is the allocated integer which uniquely
    /// identifies this `ioctl` within the `Group`.
    ///
    /// # Safety
    ///
    /// For safety details, see [Ioctl::classic].
    pub const unsafe fn write<'a, T>(self, nr: u8) -> Ioctl<Write, &'a T> {
        self.make(nr, platform::WRITE, size_of::<T>())
    }

    /// Define a new `WriteRead` `ioctl` with an associated `type`
    ///
    /// This corresponds to the `_IOWR(type, nr, size)` macro.
    ///
    /// The `nr` argument is the allocated integer which uniquely
    /// identifies this `ioctl` within the `Group`.
    ///
    /// # Safety
    ///
    /// For safety details, see [Ioctl::classic].
    pub const unsafe fn write_read<'a, T>(self, nr: u8) -> Ioctl<WriteRead, &'a T> {
        self.make(nr, platform::READ | platform::WRITE, size_of::<T>())
    }
}

/// A defined `ioctl` along with its associated `direction` and `type`
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Ioctl<D, T>(c_ulong, PhantomData<(D, T)>);

impl<D, T> Ioctl<D, T> {
    /// Define a new `ioctl` with an associated `direction` and `type`
    ///
    /// The `request` argument is the allocated integer which uniquely
    /// identifies this `ioctl`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because defining an `ioctl` with an incorrect
    /// `request`, `direction` or argument `type` can later result in memory
    /// corruption. You are responsible to ensure these values are correct.
    ///
    /// Further, you are responsible to ensure that the argument `type` itself
    /// provides appropriate safe wrappers around its raw contents. For some
    /// `type`s none are required. For others, particularly `type`s that pass
    /// pointers to the kernel as `u64`, you need to ensure that things like
    /// lifetimes are correct.
    pub const unsafe fn classic(request: c_ulong) -> Self {
        Self(request, PhantomData)
    }

    /// Lie about the ioctl direction or type
    ///
    /// This function should be avoided unless necessary.
    ///
    /// Sometimes kernel developers make mistakes and use the wrong macros
    /// or types during their ioctl definitions. However, once merged these
    /// form part of the userspace API and won't be broken. Therefore, we
    /// need a way to use the `request` number with the incorrect type. This
    /// function allows this.
    ///
    /// Whenever using this function, it would be wise to include a comment
    /// with a link to the kernel's ioctl definition and explaining why the
    /// definition is incorrect.
    ///
    /// # Safety
    ///
    /// For safety details, see [Ioctl::classic].
    ///
    /// Additionally, one should note that this function discards your normal
    /// protections. So you need to make sure that you have it correct.
    pub const unsafe fn lie<E, U>(self) -> Ioctl<E, U> {
        Ioctl(self.0, PhantomData)
    }
}

impl Ioctl<Read, c_void> {
    /// Issue an `ioctl` to read a file descriptor's metadata as `c_uint`.
    ///
    /// No argument is supplied to the internal `ioctl()` call. The raw
    /// (positive) return value from the internal `ioctl()` call is returned
    /// on success.
    pub fn ioctl(self, fd: &impl AsRawFd) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error())
    }
}

impl<T: FromBytes> Ioctl<Read, &T> {
    /// Issue an `ioctl` to read a file descriptor's metadata as type `T`.
    ///
    /// A zeroed instance of type `T` is passed as the first argument to the
    /// internal `ioctl()` call. Upon success, returns the raw (positive)
    /// return value and the instance of `T`.
    ///
    /// `T: FromBytes` lets us start from a zeroed, already-valid `T` rather
    /// than uninitialized memory: if the underlying `ioctl()` fails to fully
    /// initialize every byte on some success path, the result is stale
    /// zeros in the untouched fields, not undefined behavior.
    pub fn ioctl(self, fd: &impl AsRawFd) -> Result<(c_uint, T)> {
        let mut out = T::new_zeroed();

        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, &mut out as *mut T, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error()).map(|x| (x, out))
    }
}

impl Ioctl<Write, c_void> {
    /// Issue an `ioctl` to modify a file descriptor
    ///
    /// No argument is provided.
    ///
    /// On success, returns the (positive) return value.
    pub fn ioctl(self, fd: &mut impl AsRawFd) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error())
    }
}

impl Ioctl<Write, c_int> {
    /// Issue an `ioctl` to modify a file descriptor
    ///
    /// A C-integer argument is provided.
    ///
    /// On success, returns the (positive) return value.
    pub fn ioctl(self, fd: &mut impl AsRawFd, data: c_int) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, data, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error())
    }
}

impl<T: IntoBytes + Immutable> Ioctl<Write, &T> {
    /// Issue an `ioctl` to modify a file descriptor
    ///
    /// A reference to an immutable instance of `T` is provided as the argument.
    ///
    /// On success, returns the (positive) return value.
    ///
    /// `T: IntoBytes + Immutable` guarantees no uninitialized padding is
    /// exposed to the kernel and that no interior mutability could race the
    /// read.
    pub fn ioctl(self, fd: &mut impl AsRawFd, data: &T) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, data as *const _, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error())
    }
}

impl<T: FromBytes + IntoBytes + Immutable> Ioctl<WriteRead, &T> {
    /// Issue an `ioctl` to modify a file descriptor and read its metadata
    ///
    /// A reference to a mutable instance of `T` is provided as the argument.
    ///
    /// On success, returns the (positive) return value.
    ///
    /// `T: FromBytes + IntoBytes + Immutable`: the kernel may overwrite any
    /// bytes of `T` during this call, and `data` remains a valid `&mut T`
    /// that safe code can read afterward, so every bit pattern the kernel
    /// could leave behind must be a legal `T`.
    pub fn ioctl(self, fd: &mut impl AsRawFd, data: &mut T) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, data as *mut _, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // These expected values assume the standard (asm-generic) platform
    // encoding, which is correct for x86_64 CI runners.

    const KVMIO: Group = Group::new(0xAE);

    #[test]
    fn req() {
        const KVM_CREATE_VM: Ioctl<Read, c_void> = unsafe { KVMIO.none(0x01) };

        assert_eq!(KVM_CREATE_VM.0, 0xae01);

        if let Ok(file) = std::fs::File::open("/dev/kvm") {
            let fd: c_uint = KVM_CREATE_VM.ioctl(&file).unwrap();
            assert!(fd > 0);
        }
    }

    #[test]
    fn req_r() {
        const KVM_X86_GET_MCE_CAP_SUPPORTED: Ioctl<Read, &u64> = unsafe { KVMIO.read(0x9d) };

        assert_eq!(KVM_X86_GET_MCE_CAP_SUPPORTED.0, 0x8008_ae9d);
    }

    // `/dev/kvm`-gated tests above (`req`) only actually run on CI hosts with
    // KVM enabled, which most hosted runners don't have — the `if let Ok`
    // guard makes them silently skip rather than fail, but that also means
    // they usually provide zero real coverage of the ioctl() call path.
    // `FIONREAD` (`_IOR('f', 127, int)` on Linux) works on any regular file,
    // no special device/hardware/permissions required, so this exercises
    // `Ioctl<Read, &T>::ioctl()`'s `T::new_zeroed()` path (this impl no
    // longer uses `MaybeUninit`/`assume_init`) against a real ioctl() call
    // that's guaranteed to run everywhere this test suite does.
    #[test]
    fn req_r_runs_everywhere_via_fionread() {
        const FIONREAD: Ioctl<Read, &c_int> = unsafe { Ioctl::classic(0x541B) };

        let mut path = std::env::temp_dir();
        path.push(format!("iocuddle-test-fionread-{}", std::process::id()));
        std::fs::write(&path, b"hello").expect("write temp file");

        let file = std::fs::File::open(&path).expect("open temp file");
        let result = FIONREAD.ioctl(&file);
        let _ = std::fs::remove_file(&path);

        let (ret, available) = result.unwrap();
        assert_eq!(ret, 0);
        assert_eq!(available, 5);
    }

    #[test]
    fn req_w() {
        const KVM_X86_SETUP_MCE: Ioctl<Write, &u64> = unsafe { KVMIO.write(0x9c) };

        assert_eq!(KVM_X86_SETUP_MCE.0, 0x4008_ae9c);
    }

    // Same rationale as `req_r_runs_everywhere_via_fionread`: `FIONBIO`
    // (`_IOW('f', 126, int)` on Linux) sets O_NONBLOCK through any file
    // descriptor and needs no special device, so it exercises
    // `Ioctl<Write, &T>::ioctl()` for real everywhere this suite runs.
    #[test]
    fn req_w_runs_everywhere_via_fionbio() {
        const FIONBIO: Ioctl<Write, &c_int> = unsafe { Ioctl::classic(0x5421) };

        let mut path = std::env::temp_dir();
        path.push(format!("iocuddle-test-fionbio-{}", std::process::id()));
        std::fs::write(&path, b"hello").expect("write temp file");

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("open temp file");
        let result = FIONBIO.ioctl(&mut file, &1);
        let _ = std::fs::remove_file(&path);

        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn req_wr() {
        const KVM_PPC_ALLOCATE_HTAB: Ioctl<WriteRead, &u32> = unsafe { KVMIO.write_read(0xa7) };

        assert_eq!(KVM_PPC_ALLOCATE_HTAB.0, 0xc004_aea7);
    }

    // FIONREAD/FIONBIO above are legacy fixed-value ioctls, not composed via
    // the kernel's _IOC macros (that's exactly why they need `Ioctl::classic`
    // rather than `Group::read`/`Group::write`) — so despite exercising a
    // real ioctl() call, they say nothing about whether `Group`'s own _IOC
    // encoding logic is correct against a real syscall. Every other real
    // call in this module (`req`/`req_r`/`req_w`/`req_wr` via `Group::none`/
    // `read`/`write`/`write_read`) is `/dev/kvm`-gated and silently skips
    // when unavailable, which is most CI hosts.
    //
    // `/dev/ptmx` closes that gap: it's needed for basic terminal support
    // (spawning a subprocess with a pty, `ssh`, etc.), so it's present and
    // world-accessible on essentially any real Linux host, unlike `/dev/kvm`
    // which needs virtualization support. It exposes two genuinely
    // _IOC-composed ioctls with no side effects worth avoiding:
    // `TIOCGPTN = _IOR('T', 0x30, unsigned int)` and
    // `TIOCSPTLCK = _IOW('T', 0x31, int)`. These hard-fail rather than
    // silently skip if `/dev/ptmx` is missing — a Linux host without it is
    // missing basic terminal support, which is worth surfacing as a real
    // test failure, not hiding behind an `if let Ok`.
    const PTY: Group = Group::new(0x54); // 'T', matches <asm-generic/ioctls.h>

    // Linux's O_NOCTTY, hardcoded to avoid a libc dependency for one flag —
    // same "documented raw magic value" style this crate already uses for
    // ioctl numbers. Keeps opening /dev/ptmx from accidentally attaching it
    // as this process's controlling terminal.
    const O_NOCTTY: i32 = 0o400;

    fn open_ptmx() -> std::fs::File {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(O_NOCTTY)
            .open("/dev/ptmx")
            .expect("/dev/ptmx should exist and be openable on any real Linux host")
    }

    #[test]
    fn group_read_runs_everywhere_via_tiocgptn() {
        const TIOCGPTN: Ioctl<Read, &u32> = unsafe { PTY.read(0x30) };
        assert_eq!(TIOCGPTN.0, 0x8004_5430);

        let file = open_ptmx();
        let (ret, _ptn) = TIOCGPTN.ioctl(&file).unwrap();
        assert_eq!(ret, 0);
    }

    #[test]
    fn group_write_runs_everywhere_via_tiocsptlck() {
        const TIOCSPTLCK: Ioctl<Write, &c_int> = unsafe { PTY.write(0x31) };
        assert_eq!(TIOCSPTLCK.0, 0x4004_5431);

        let mut file = open_ptmx();
        // 0 unlocks the pty pair; harmless on a pty nothing else is using.
        let ret = TIOCSPTLCK.ioctl(&mut file, &0).unwrap();
        assert_eq!(ret, 0);
    }
}
