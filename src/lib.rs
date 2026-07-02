// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]
//!
//! ## Requirements on `T`
//!
//! [`Group::read`], [`Group::write`], [`Group::write_read`], [`Ioctl::classic`],
//! and [`Ioctl::lie`] are `unsafe fn`s: they let you assert that a given
//! `(nr, direction, size)` triple, and the argument type `T` paired with it,
//! matches the kernel's real definition of that `ioctl`. Once that assertion
//! has been made, the resulting `Ioctl<D, T>::ioctl()` call is safe to invoke
//! — but only because it is checked against, and constrained by, two
//! independent halves of a contract that `T` must uphold. Both halves are
//! necessary; neither is sufficient on its own.
//!
//! **1. Representation (enforced by the compiler).** Depending on the
//! direction, the kernel may write arbitrary bytes into `T`'s memory that
//! safe Rust code will subsequently read back as a `T` value. This is only
//! sound if every bit pattern the kernel could leave behind is a legal value
//! of `T` — i.e. no enums, `bool`s, references, or other types with invalid
//! bit patterns, and no padding bytes that would otherwise be exposed to the
//! kernel uninitialized. The direction-specific `ioctl()` impls require
//! [`zerocopy`]'s [`FromBytes`]/[`IntoBytes`]/[`Immutable`] traits precisely
//! where the kernel reads from or writes into `T`, so this half of the
//! contract is checked at compile time.
//!
//! **2. Construction (a convention this crate cannot check).** Representation
//! safety alone does not stop *misuse*: a `T` with public fields could still
//! be constructed with a value that's representation-valid but semantically
//! wrong for the ioctl (an unterminated name string, an inconsistent length
//! field), or — most importantly — a raw pointer-shaped field could be set to
//! an address with no real allocation behind it. The convention is that `T`'s
//! fields are private to the module that defines the `ioctl`, with the only
//! public constructors/mutators being ones that cannot produce an invalid
//! `T`. When a field is itself the address of a second buffer (common in
//! ioctls like `SG_IO`'s `sg_io_hdr`), use [`Ptr`]/[`PtrMut`] for that field
//! rather than a bare integer: their constructors can only be called with a
//! genuine, currently-live Rust borrow of the target, and that borrow's
//! lifetime is threaded through `T` into the `&T`/`&mut T` you pass to
//! `.ioctl()`, so ordinary borrow-checking forbids touching the pointee
//! anywhere else for as long as the kernel might be reading or writing
//! through it.
//!
//! A minimal example combining both halves — a struct wrapping a
//! pointer-shaped field, usable with `Ioctl<Read, &T>`, `Ioctl<Write, &T>`,
//! and `Ioctl<WriteRead, &T>` alike:
//!
//! ```
//! use iocuddle::Ptr;
//! use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};
//!
//! #[repr(C)]
//! #[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
//! struct MyArg<'a> {
//!     buf: Ptr<'a, [u8; 16]>,
//! }
//! ```
//!
//! [`Ptr`]/[`PtrMut`] implement [`FromBytes`], so a struct embedding one
//! isn't confined to the `Write` direction (an earlier version of this
//! crate confined them exactly that way, which made them useless for
//! `WriteRead` ioctls like x86's `KVM_MEMORY_ENCRYPT_OP`/`kvm_sev_cmd`,
//! which reads a pointer field while only writing back into other fields
//! of the same struct). See [`Ptr`]'s docs for why this is sound and the
//! invariant it depends on.
//!
//! One real sharp edge: `zerocopy`'s derive macros prove a `#[repr(C)]`
//! struct has no undefined padding bytes by reasoning about adjacent fields'
//! sizes and alignments, and that reasoning is conservative for a foreign
//! generic field type like `Ptr<'a, U>` — pairing a `Ptr`/`PtrMut` field
//! with a plain sibling field in the same struct may require the derive
//! macro's padding proof to fall back to demanding the sibling also be
//! [`zerocopy::Unaligned`], or an explicit padding field. Reordering fields
//! is *not* a real option for a struct that must byte-for-byte mirror a
//! fixed kernel layout (like `SG_IO`'s `sg_io_hdr`) — an explicit padding
//! field (or wrapping the plain field in a byte-oriented `Unaligned` type)
//! is the only adjustment available when the kernel's own field order isn't
//! friendly to this proof.
//!
//! Note what this contract does *not* cover: an ioctl whose kernel side
//! retains and uses a pointer *after* the call returns (persistent buffer
//! registration, as with some `io_uring` or KVM memory-region ioctls) needs
//! an RAII registration/deregistration object, not a pointer wrapper — no
//! synchronous borrow can correctly express "valid until explicitly
//! revoked."
#![deny(missing_docs)]
#![deny(clippy::all)]

use core::convert::TryInto;
use core::marker::PhantomData;
use core::mem::size_of;
use core::ptr::null;

use std::io::{Error, Result};
use std::os::raw::{c_int, c_uint, c_ulong, c_void};
use std::os::unix::io::{AsFd, AsRawFd};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

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

/// An embedded pointer field that the kernel only *reads* through.
///
/// Some `ioctl`s embed the address of a second buffer as a raw integer
/// field (e.g. SCSI `SG_IO`'s `sg_io_hdr.dxferp`). A bare `u64` field
/// carries no information about what it points to or how long that pointee
/// needs to remain valid — see the "Requirements on `T`" section of the
/// crate docs. `Ptr` closes that gap: it can only be constructed from a
/// live `&'a U`, and threading `'a` through the struct that embeds it (and
/// from there into the `&T`/`&mut T` passed to `Ioctl::ioctl`) makes ordinary
/// borrow-checking forbid mutating or dropping the pointee for as long as
/// the kernel might be reading through the address.
///
/// `#[repr(transparent)]` over a bare `u64`: the [`PhantomData`] tail is
/// zero-sized, so this has the exact layout the kernel ABI expects for a
/// pointer-as-integer field.
///
/// Implements [`zerocopy::FromBytes`], so a struct embedding this field can
/// still be used with `Ioctl<Read, &T>`/`Ioctl<WriteRead, &T>`, not just
/// `Write` — this matters in practice: x86's `KVM_MEMORY_ENCRYPT_OP`
/// (`struct kvm_sev_cmd`), for instance, is `WriteRead` and reads a pointer
/// field while only writing back into *other* fields, and confining such a
/// `T` to `Write` would make `Ptr` unusable for it. This is sound only
/// because `PhantomData<&'a U>`
/// occupies zero bytes and is never dereferenced by this type — an
/// arbitrary kernel-written `u64` in `addr` produces an inert value, not a
/// dangling reference anyone can act on. **This is a load-bearing
/// invariant, not an implementation detail**: neither `Ptr` nor [`PtrMut`]
/// may ever gain a safe method that dereferences `addr` (e.g. a `get(&self)
/// -> &U`) without redoing this soundness argument from scratch — that
/// would turn a bytes-reconstructed instance into a genuinely dangerous
/// one.
///
/// The borrow lasts exactly as long as this value does. `Ptr` only borrows
/// its target shared (unlike [`PtrMut`]'s exclusive borrow), so reading the
/// original binding is still fine, but mutating it while a `Ptr` derived
/// from it is still alive is a borrow-check error:
///
/// ```compile_fail
/// use iocuddle::Ptr;
///
/// let mut val: u32 = 42;
/// let p = Ptr::new(&val);
/// val = 1; // fails: `val` is still immutably borrowed by `p`
/// let _ = p;
/// ```
///
/// Not covered: an ioctl whose kernel side keeps using the address *after*
/// this call returns (persistent buffer registration) needs an RAII
/// registration/deregistration object, not this type — a borrow that ends
/// when `.ioctl()` returns is the wrong shape for that case.
#[repr(transparent)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Ptr<'a, U> {
    addr: u64,
    _marker: PhantomData<&'a U>,
}

impl<'a, U: IntoBytes + Immutable> Ptr<'a, U> {
    /// Build a pointer-as-integer field from a live reference.
    ///
    /// Not `const fn`: a pointer-to-integer cast of a genuine reference is
    /// rejected even at `const fn` *definition* time ("pointers cannot be
    /// cast to integers during const eval"), because the address doesn't
    /// exist until the referent is placed in memory at runtime — this is a
    /// hard restriction on the cast itself, not something that only bites
    /// a caller who happens to invoke it from a const context. `Ptr` values
    /// are always built just-in-time before a call, never as part of a
    /// `const` `Ioctl` declaration, so this costs nothing in practice.
    pub fn new(r: &'a U) -> Self {
        Self {
            addr: r as *const U as u64,
            _marker: PhantomData,
        }
    }
}

/// An embedded pointer field that the kernel *writes* through, and the
/// caller reads back afterward (e.g. `SG_IO`'s sense-buffer pointer).
///
/// Same layout and lifetime-carrying discipline as [`Ptr`], but for the
/// output direction: constructed from `&'a mut U`, so no other Rust code
/// can read or write the pointee while this value — and the `ioctl` call it
/// participates in — is alive. Unlike [`Ptr`], this type does not implement
/// `Copy`/`Clone`: duplicating a token that represents exclusive access to
/// the kernel's write target is a footgun worth refusing to compile, even
/// where it wouldn't immediately violate the borrow on the pointee.
///
/// `PtrMut::new` requires `U: FromBytes` (in addition to [`Ptr`]'s bounds)
/// because the kernel may leave an arbitrary bit pattern in `*U` that the
/// caller will read back as a `U` once the borrow ends. `PtrMut` itself
/// implements [`zerocopy::FromBytes`] regardless of `U` — see [`Ptr`]'s docs
/// for why that's sound (no accessor ever dereferences `addr`) and why it
/// matters (it's what makes `Ptr`/`PtrMut` usable in `Read`/`WriteRead`
/// ioctls, not just `Write`).
///
/// The borrow lasts exactly as long as this value does. Touching the
/// original binding while a `PtrMut` derived from it is still alive is a
/// borrow-check error:
///
/// ```compile_fail
/// use iocuddle::PtrMut;
///
/// let mut val: u32 = 42;
/// let p = PtrMut::new(&mut val);
/// val = 1; // fails: `val` is still exclusively borrowed by `p`
/// let _ = p;
/// ```
///
/// Not `Clone`, so duplicating the token is a compile error, not just a
/// clippy lint:
///
/// ```compile_fail
/// use iocuddle::PtrMut;
///
/// let mut val: u32 = 42;
/// let p = PtrMut::new(&mut val);
/// let _p2 = p.clone(); // fails: no method named `clone`
/// ```
///
/// And not `Copy`, so using it twice is a use-after-move error:
///
/// ```compile_fail
/// use iocuddle::PtrMut;
///
/// let mut val: u32 = 42;
/// let p = PtrMut::new(&mut val);
/// let _moved = p;
/// let _used_again = p; // fails: use of moved value
/// ```
///
/// `Send`/`Sync` are derived structurally through the `PhantomData<&'a mut
/// U>` marker, mirroring `&'a mut U`'s own rules exactly: `PtrMut<U>: Send`
/// requires `U: Send`, regardless of `U: Sync`. `std::sync::MutexGuard` is
/// `Sync` (if its contents are) but deliberately never `Send` (unlocking a
/// mutex from a different thread than the one that locked it is unsound),
/// so it's a type that actually distinguishes the correct rule from a
/// plausible-looking wrong one:
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<iocuddle::PtrMut<'_, std::sync::MutexGuard<'_, u32>>>();
/// // fails: MutexGuard is not Send, so neither is PtrMut wrapping one
/// ```
#[repr(transparent)]
#[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct PtrMut<'a, U> {
    addr: u64,
    _marker: PhantomData<&'a mut U>,
}

impl<'a, U: FromBytes + IntoBytes + Immutable> PtrMut<'a, U> {
    /// Build a pointer-as-integer field from a live mutable reference.
    ///
    /// See [`Ptr::new`] for why this isn't `const fn`.
    pub fn new(r: &'a mut U) -> Self {
        Self {
            addr: r as *mut U as u64,
            _marker: PhantomData,
        }
    }
}

impl Ioctl<Read, c_void> {
    /// Issue an `ioctl` to read a file descriptor's metadata as `c_uint`.
    ///
    /// No argument is supplied to the internal `ioctl()` call. The raw
    /// (positive) return value from the internal `ioctl()` call is returned
    /// on success.
    pub fn ioctl(self, fd: impl AsFd) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_fd().as_raw_fd(), self.0, null::<c_void>()) };

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
    pub fn ioctl(self, fd: impl AsFd) -> Result<(c_uint, T)> {
        let mut out = T::new_zeroed();

        let r =
            unsafe { ioctl(fd.as_fd().as_raw_fd(), self.0, &mut out as *mut T, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error()).map(|x| (x, out))
    }
}

impl Ioctl<Write, c_void> {
    /// Issue an `ioctl` to modify a file descriptor
    ///
    /// No argument is provided.
    ///
    /// On success, returns the (positive) return value.
    pub fn ioctl(self, fd: impl AsFd) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_fd().as_raw_fd(), self.0, null::<c_void>()) };

        r.try_into().map_err(|_| Error::last_os_error())
    }
}

impl Ioctl<Write, c_int> {
    /// Issue an `ioctl` to modify a file descriptor
    ///
    /// A C-integer argument is provided.
    ///
    /// On success, returns the (positive) return value.
    pub fn ioctl(self, fd: impl AsFd, data: c_int) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_fd().as_raw_fd(), self.0, data, null::<c_void>()) };

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
    pub fn ioctl(self, fd: impl AsFd, data: &T) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_fd().as_raw_fd(), self.0, data as *const _, null::<c_void>()) };

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
    pub fn ioctl(self, fd: impl AsFd, data: &mut T) -> Result<c_uint> {
        let r = unsafe { ioctl(fd.as_fd().as_raw_fd(), self.0, data as *mut _, null::<c_void>()) };

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

    // Every other test in this module passes `&file` — which only proves
    // this migration didn't regress the single most common calling
    // pattern, since `&File` already satisfied the old `&impl AsRawFd`
    // bound too. `impl AsFd`'s actual payoff is accepting things that
    // never satisfied `AsRawFd` at all, like an owned `BorrowedFd` passed
    // by value: `BorrowedFd` implements `AsFd` but not `AsRawFd` on
    // stable, so this would fail to compile against the pre-migration
    // signature.
    #[test]
    fn ioctl_accepts_borrowed_fd_directly() {
        const FIONREAD: Ioctl<Read, &c_int> = unsafe { Ioctl::classic(0x541B) };

        let mut path = std::env::temp_dir();
        path.push(format!("iocuddle-test-fionread-asfd-{}", std::process::id()));
        std::fs::write(&path, b"hello").expect("write temp file");

        let file = std::fs::File::open(&path).expect("open temp file");
        let result = FIONREAD.ioctl(file.as_fd());
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

        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("open temp file");
        let result = FIONBIO.ioctl(&file, &1);
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

        let file = open_ptmx();
        // 0 unlocks the pty pair; harmless on a pty nothing else is using.
        let ret = TIOCSPTLCK.ioctl(&file, &0).unwrap();
        assert_eq!(ret, 0);
    }

    #[test]
    fn ptr_addr_matches_reference() {
        let val: u32 = 42;
        let expected = &val as *const u32 as u64;
        let p = Ptr::new(&val);
        assert_eq!(p.addr, expected);
    }

    #[test]
    #[allow(clippy::clone_on_copy)] // deliberately exercising Clone, not just Copy
    fn ptr_is_copy_and_clone() {
        let val: u32 = 42;
        let p = Ptr::new(&val);
        let p2 = p; // Copy: `p` must still be usable afterward.
        let p3 = p2.clone();
        assert_eq!(p.addr, p2.addr);
        assert_eq!(p2.addr, p3.addr);
    }

    #[test]
    fn ptrmut_addr_matches_reference() {
        let mut val: u32 = 42;
        let expected = &mut val as *mut u32 as u64;
        let p = PtrMut::new(&mut val);
        assert_eq!(p.addr, expected);
    }

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn ptr_send_sync_mirrors_shared_reference() {
        // u32: Sync, so &u32 (and therefore Ptr<'_, u32>) is both Send and Sync.
        assert_send::<Ptr<'_, u32>>();
        assert_sync::<Ptr<'_, u32>>();

        // std::sync::MutexGuard is Sync but *not* Send — an asymmetric case
        // that actually distinguishes "Ptr<U>: Send iff U: Sync" (correct)
        // from a hypothetical "Ptr<U>: Send iff U: Send" (wrong): only the
        // correct rule lets this line compile, since a lone `u32` target
        // can't tell the two rules apart (u32 is both Send and Sync).
        assert_send::<Ptr<'_, std::sync::MutexGuard<'_, u32>>>();
        assert_sync::<Ptr<'_, std::sync::MutexGuard<'_, u32>>>();
    }

    #[test]
    fn ptrmut_send_sync_mirrors_exclusive_reference() {
        // u32: Send + Sync, so &mut u32 (and therefore PtrMut<'_, u32>) is
        // both Send and Sync.
        assert_send::<PtrMut<'_, u32>>();
        assert_sync::<PtrMut<'_, u32>>();

        // Same asymmetric target as above, for the Sync half: MutexGuard is
        // Sync regardless of its own Send-ness, so this only compiles under
        // the correct "PtrMut<U>: Sync iff U: Sync" rule.
        assert_sync::<PtrMut<'_, std::sync::MutexGuard<'_, u32>>>();
    }

    // Locks in the fix: a struct embedding Ptr/PtrMut must satisfy every
    // direction's bound, not just Write's — otherwise WriteRead ioctls that
    // read a pointer field while only writing back into other fields (e.g.
    // x86's KVM_MEMORY_ENCRYPT_OP / struct kvm_sev_cmd) would have no
    // usable safe wrapper at all. Struct names denote which field type
    // they embed, not which single direction is being asserted — every
    // assertion below is applied to both.
    #[test]
    fn ptr_and_ptrmut_in_struct_support_every_direction() {
        #[repr(C)]
        #[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
        struct PtrArg<'a> {
            buf: Ptr<'a, [u8; 4]>,
        }

        #[repr(C)]
        #[derive(FromBytes, IntoBytes, Immutable, KnownLayout)]
        struct PtrMutArg<'a> {
            out: PtrMut<'a, [u8; 4]>,
        }

        fn assert_read<T: FromBytes>() {}
        fn assert_write<T: IntoBytes + Immutable>() {}
        fn assert_write_read<T: FromBytes + IntoBytes + Immutable>() {}

        assert_read::<PtrArg<'_>>();
        assert_write::<PtrArg<'_>>();
        assert_write_read::<PtrArg<'_>>();

        assert_read::<PtrMutArg<'_>>();
        assert_write::<PtrMutArg<'_>>();
        assert_write_read::<PtrMutArg<'_>>();
    }
}
