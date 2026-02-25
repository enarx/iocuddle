// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(clippy::all)]

use core::convert::TryInto;
use core::marker::PhantomData;
use core::mem::{size_of, MaybeUninit};
use core::ptr::null;

use std::io::{Error, Result};
use std::os::raw::{c_int, c_uint, c_ulong, c_void};
use std::os::unix::io::AsRawFd;

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

/// PA-RISC: same bit widths as standard but READ and WRITE are swapped
#[cfg(target_arch = "parisc")]
mod platform {
    use std::os::raw::c_ulong;
    pub const SIZEBITS: c_ulong = 14;
    pub const NONE: c_ulong = 0;
    pub const READ: c_ulong = 1;
    pub const WRITE: c_ulong = 2;
}

/// Standard (asm-generic): x86, x86_64, arm, aarch64, riscv, s390x, etc.
#[cfg(not(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "sparc",
    target_arch = "sparc64",
    target_arch = "parisc"
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

impl<T> Ioctl<Read, &T> {
    /// Issue an `ioctl` to read a file descriptor's metadata as type `T`.
    ///
    /// A zeroed instance of type `T` is passed as the first argument to the
    /// internal `ioctl()` call. Upon success, returns the raw (positive)
    /// return value and the instance of `T`.
    pub fn ioctl(self, fd: &impl AsRawFd) -> Result<(c_uint, T)> {
        let mut out: MaybeUninit<T> = MaybeUninit::uninit();

        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, out.as_mut_ptr(), null::<c_void>()) };

        r.try_into()
            .map_err(|_| Error::last_os_error())
            .map(|x| (x, unsafe { out.assume_init() }))
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

impl<T> Ioctl<Write, &T> {
    /// Issue an `ioctl` to modify a file descriptor
    ///
    /// A reference to an immutable instance of `T` is provided as the argument.
    ///
    /// On success, returns the (positive) return value.
    pub fn ioctl(self, fd: &mut impl AsRawFd, data: &T) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_raw_fd(), self.0, data as *const _, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error())
    }
}

impl<T> Ioctl<WriteRead, &T> {
    /// Issue an `ioctl` to modify a file descriptor and read its metadata
    ///
    /// A reference to a mutable instance of `T` is provided as the argument.
    ///
    /// On success, returns the (positive) return value.
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

        if let Ok(mut file) = std::fs::File::open("/dev/kvm") {
            let fd: c_uint = KVM_CREATE_VM.ioctl(&mut file).unwrap();
            assert!(fd > 0);
        }
    }

    #[test]
    fn req_r() {
        const KVM_X86_GET_MCE_CAP_SUPPORTED: Ioctl<Read, &u64> = unsafe { KVMIO.read(0x9d) };

        assert_eq!(KVM_X86_GET_MCE_CAP_SUPPORTED.0, 0x8008_ae9d);
    }

    #[test]
    fn req_w() {
        const KVM_X86_SETUP_MCE: Ioctl<Write, &u64> = unsafe { KVMIO.write(0x9c) };

        assert_eq!(KVM_X86_SETUP_MCE.0, 0x4008_ae9c);
    }

    #[test]
    fn req_wr() {
        const KVM_PPC_ALLOCATE_HTAB: Ioctl<WriteRead, &u32> = unsafe { KVMIO.write_read(0xa7) };

        assert_eq!(KVM_PPC_ALLOCATE_HTAB.0, 0xc004_aea7);
    }
}
