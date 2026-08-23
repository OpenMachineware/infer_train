# core/tensor.mojo
#
# The runtime tensor data structure.
#
# Design decisions:
#   * `dtype` and `rank` are comptime (square-bracket) parameters so the
#     compiler can specialize every kernel on element type and arity.
#   * `shape`/`strides` are *runtime values* but stored inline in a
#     `StaticTuple[Int, rank]`.  Mojo 1.0's `List` is move-only (it is
#     `Copyable` but not `ImplicitlyCopyable`), so holding shape in a `List`
#     would make every tensor move-only and forbid the tuple returns the
#     operator interface needs.  `StaticTuple` keeps `Tensor` trivially
#     copyable, which is the natural fit for a comptime-rank tensor.
#   * Storage is a `Pointer[Scalar[dtype], MutUntrackedOrigin]`; we never
#     track individual allocations here - the `MemoryPool` owns the bulk
#     buffer and frees it wholesale between requests.
#   * `grad` and `quantization_info` are recursive references, so Mojo 1.0
#     forbids storing them inline.  They are held behind an owning-free
#     `Optional[Pointer[...]]` (None means "absent").
#
# Mojo 1.0 notes reflected here: `Option` -> `Optional`, `UnsafePointer` ->
# `Pointer[T, MutUntrackedOrigin]`, `inout` -> `mut`, `enum` -> struct-based.

from .device import Device
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.utils.static_tuple import StaticTuple


def _numel_of[rank: Int](shape: StaticTuple[Int, rank]) -> Int:
    var total = 1
    for i in range(rank):
        total *= shape[i]
    return total


def _compute_strides[rank: Int](
    shape: StaticTuple[Int, rank]
) -> StaticTuple[Int, rank]:
    """Row-major strides; `strides[i] = strides[i+1] * shape[i+1]`."""
    var strides = StaticTuple[Int, rank](fill=1)
    var i = rank - 2
    while i >= 0:
        strides[i] = strides[i + 1] * shape[i + 1]
        i -= 1
    return strides


struct Tensor[dtype: DType, rank: Int](Copyable, Movable, ImplicitlyCopyable):
    var _shape: StaticTuple[Int, Self.rank]
    var _strides: StaticTuple[Int, Self.rank]
    var _numel: Int
    var _data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _device: Device
    var requires_grad: Bool
    var _grad: Optional[
        Pointer[Tensor[Self.dtype, Self.rank], MutUntrackedOrigin]
    ]
    var _quant: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    var _opt_state: Optional[Pointer[UInt8, MutUntrackedOrigin]]

    def __init__(
        out self,
        shape: StaticTuple[Int, Self.rank],
        device: Device = Device.CPU,
    ):
        """Allocate contiguous, uninitialized storage for `shape`."""
        self._shape = shape
        self._strides = _compute_strides(shape)
        self._numel = _numel_of(shape)
        var alloc_count = self._numel
        if alloc_count < 1:
            alloc_count = 1
        self._data = unsafe_alloc[Scalar[Self.dtype]](alloc_count)
        self._device = device
        self.requires_grad = False
        self._grad = None
        self._quant = None
        self._opt_state = None

    def __init__(
        out self,
        shape: StaticTuple[Int, Self.rank],
        data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        device: Device = Device.CPU,
    ):
        """Build a view over caller-owned storage (used by `reshape`).

        No allocation happens here: the tensor borrows `data`.  This is the
        zero-copy path that lets a reshaped tensor alias its source buffer.
        """
        self._shape = shape
        self._strides = _compute_strides(shape)
        self._numel = _numel_of(shape)
        self._data = data
        self._device = device
        self.requires_grad = False
        self._grad = None
        self._quant = None
        self._opt_state = None

    # -- metadata accessors -------------------------------------------------

    def shape(self) -> StaticTuple[Int, Self.rank]:
        return self._shape

    def strides(self) -> StaticTuple[Int, Self.rank]:
        return self._strides

    def numel(self) -> Int:
        return self._numel

    def data(self) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        return self._data

    def device(self) -> Device:
        return self._device

    # -- element access -----------------------------------------------------

    def index(self, *offsets: Int) -> Int:
        """Map a multi-dimensional index to a linear memory offset.

        Uses the variadic (`*offsets: Int`) form so callers may pass any
        number of coordinates, one per axis.
        """
        var result = 0
        var axis = 0
        for offset in offsets:
            result += offset * self._strides[axis]
            axis += 1
        return result

    def get(self, offset: Int) -> Scalar[Self.dtype]:
        return self._data.unsafe_load[width=1](offset=offset)

    def set(self, offset: Int, value: Scalar[Self.dtype]):
        self._data.unsafe_store(offset, value)

    # -- views --------------------------------------------------------------

    def reshape[new_rank: Int](
        self, new_shape: StaticTuple[Int, new_rank]
    ) -> Tensor[Self.dtype, new_rank]:
        """Return a zero-copy view with a new rank and shape.

        `new_rank` is a comptime parameter so the resulting tensor type is
        fully static; the caller's shape is only checked by construction.
        """
        var view = Tensor[Self.dtype, new_rank](
            new_shape, self._data, self._device
        )
        view.requires_grad = self.requires_grad
        view._quant = self._quant
        return view

    # -- device -------------------------------------------------------------

    def is_cpu(self) -> Bool:
        return self._device.is_cpu()

    def is_gpu(self) -> Bool:
        return self._device.is_gpu()

    def to_device(self, device: Device) -> Tensor[Self.dtype, Self.rank]:
        """Return a tensor whose storage lives on `device`.

        M1 note: pure Mojo 1.0 ships no host-side GPU runtime (`std.gpu`
        only exposes device-side intrinsics; the `max.gpu.host.DeviceContext`
        API is a separate, uninstalled package).  We therefore always copy in
        host memory and only re-tag the tensor with the requested device.
        Wiring a real Metal transfer in later is a drop-in replacement for
        the body of this method.
        """
        var result = Tensor[Self.dtype, Self.rank](self._shape, device)
        for i in range(self._numel):
            result._data.unsafe_store(
                i, self._data.unsafe_load[width=1](offset=i)
            )
        return result

    # -- autograd bookkeeping ----------------------------------------------

    def grad(
        self,
    ) -> Optional[
        Pointer[Tensor[Self.dtype, Self.rank], MutUntrackedOrigin]
    ]:
        return self._grad

    def set_grad(
        mut self,
        grad: Pointer[Tensor[Self.dtype, Self.rank], MutUntrackedOrigin],
    ):
        self._grad = grad

    def zero_grad(mut self):
        """Detach any recorded gradient (sets the pointer back to None)."""
        self._grad = None

    # -- optimizer bookkeeping (M6) -----------------------------------------

    def opt_state(self) -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
        """Opaque handle to the per-parameter optimizer state (see
        train_optimizer.mojo)."""
        return self._opt_state

    def set_opt_state(mut self, state: Pointer[UInt8, MutUntrackedOrigin]):
        self._opt_state = state

    # -- quantization bookkeeping ------------------------------------------

    def quantization_info(self) -> Optional[Pointer[UInt8, MutUntrackedOrigin]]:
        """Opaque handle to a `QuantizationInfo` (see quantization.mojo)."""
        return self._quant

    def set_quantization_info(
        mut self, info: Pointer[UInt8, MutUntrackedOrigin]
    ):
        self._quant = info


def tensor_zeros[dtype: DType, rank: Int](
    shape: StaticTuple[Int, rank], device: Device = Device.CPU
) -> Tensor[dtype, rank]:
    """Allocate a zero-initialized tensor of the given shape."""
    var out = Tensor[dtype, rank](shape, device)
    var zero = Scalar[dtype](0)
    for i in range(out.numel()):
        out.set(i, zero)
    return out


def tensor_copy[dtype: DType, rank: Int](
    src: Tensor[dtype, rank]
) -> Tensor[dtype, rank]:
    """Deep-copy a tensor's payload (used by device transfers)."""
    var out = Tensor[dtype, rank](src.shape(), src.device())
    for i in range(src.numel()):
        out.set(i, src.get(i))
    return out


def copy_to_device[dtype: DType, rank: Int](
    src: Tensor[dtype, rank], device: Device
) -> Tensor[dtype, rank]:
    """Copy `src` into a tensor tagged with the target `device`.

    Lives here (not in device.mojo) because it needs the full `Tensor` type;
    device.mojo is a leaf module that `tensor.mojo` imports.
    """
    return src.to_device(device)


def copy_to_host[dtype: DType, rank: Int](
    src: Tensor[dtype, rank]
) -> Tensor[dtype, rank]:
    """Copy a (possibly device-resident) tensor back to host memory."""
    return src.to_device(Device.CPU)
