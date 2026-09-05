# core/ops/base/op_interface.mojo
#
# The uniform operator interface plus the type-erased tensor (`AnyTensor`)
# used to cross the registry boundary.
#
# Mojo 1.0 has no runtime `Any` and `Tensor` is comptime-parameterized, so a
# registry that stores one function per op name cannot reference `Tensor`
# directly.  `AnyTensor` carries the same metadata at runtime (dtype, rank,
# shape, data pointer, device) and is converted back into a static
# `Tensor[dtype, rank]` inside each dispatch function via `from_any`.

from ...device import Device
from ...tensor import Tensor
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.utils.static_tuple import StaticTuple

# Maximum erased rank supported by `AnyTensor`.  Rank-1 scales, rank-2
# activations/weights, and rank-3/4 attention tensors all fit.
comptime ANYTENSOR_MAX_RANK = 8


struct AnyTensor(Copyable, ImplicitlyCopyable, Movable):
    """Type-erased tensor handle used by the operator registry.

    The shape is stored in a fixed `StaticTuple` so `AnyTensor` stays
    trivially copyable (a `List` would make it move-only and break the
    registry's list/dict plumbing).
    """

    var dtype: DType
    var rank: Int
    var shape: StaticTuple[Int, ANYTENSOR_MAX_RANK]
    var numel: Int
    var device: Device
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var requires_grad: Bool

    def __init__(
        out self,
        dtype: DType,
        rank: Int,
        shape: StaticTuple[Int, ANYTENSOR_MAX_RANK],
        numel: Int,
        device: Device,
        data: Pointer[UInt8, MutUntrackedOrigin],
    ):
        self.dtype = dtype
        self.rank = rank
        self.shape = shape
        self.numel = numel
        self.device = device
        self.data = data
        self.requires_grad = False


def to_any[dtype: DType, rank: Int](tensor: Tensor[dtype, rank]) -> AnyTensor:
    """Wrap a static tensor as a type-erased `AnyTensor` (zero copy)."""
    var shape = StaticTuple[Int, ANYTENSOR_MAX_RANK](fill=0)
    var static_shape = tensor.shape()
    for i in range(rank):
        shape[i] = static_shape[i]
    return AnyTensor(
        dtype,
        rank,
        shape,
        tensor.numel(),
        tensor.device(),
        tensor.data().unsafe_bitcast[UInt8](),
    )


def _static_shape[
    rank: Int
](shape: StaticTuple[Int, ANYTENSOR_MAX_RANK]) -> StaticTuple[Int, rank]:
    var result = StaticTuple[Int, rank](fill=0)
    for i in range(rank):
        result[i] = shape[i]
    return result


def from_any[
    dtype: DType, rank: Int
](any_tensor: AnyTensor) -> Tensor[dtype, rank]:
    """Rebuild a static tensor *view* over the same storage (zero copy)."""
    var shape = _static_shape[rank](any_tensor.shape)
    var data = any_tensor.data.unsafe_bitcast[Scalar[dtype]]()
    var tensor = Tensor[dtype, rank](shape, data, any_tensor.device)
    tensor.requires_grad = any_tensor.requires_grad
    return tensor


# ---------------------------------------------------------------------------
# Function type aliases
#
# Mojo 1.0 note: a `comptime` alias of a function type is treated as a trait
# and cannot be used as a struct field type, so the concrete function types
# are spelled out inline below.  The intended aliases are:
#   ForwardFn          = def(List[AnyTensor]) -> List[AnyTensor]
#   ForwardWithSavedFn = def(List[AnyTensor]) -> Tuple[List[AnyTensor],
#                                                      List[AnyTensor]]
#   BackwardFn         = def(List[AnyTensor], List[AnyTensor]) ->
#                                                       List[AnyTensor]
# ---------------------------------------------------------------------------


struct OpInfo(Copyable, ImplicitlyCopyable, Movable):
    """Metadata and function pointers for one (name, device) op impl."""

    var name: String
    var forward: def(List[AnyTensor]) thin -> List[AnyTensor]
    var forward_with_saved: def(List[AnyTensor]) thin -> Tuple[
        List[AnyTensor], List[AnyTensor]
    ]
    var backward: def(List[AnyTensor], List[AnyTensor]) thin -> List[AnyTensor]
    var device: Device
    var priority: Int

    def __init__(
        out self,
        name: String,
        forward: def(List[AnyTensor]) thin -> List[AnyTensor],
        forward_with_saved: def(List[AnyTensor]) thin -> Tuple[
            List[AnyTensor], List[AnyTensor]
        ],
        backward: def(List[AnyTensor], List[AnyTensor]) thin -> List[AnyTensor],
        device: Device,
        priority: Int = 0,
    ):
        self.name = name
        self.forward = forward
        self.forward_with_saved = forward_with_saved
        self.backward = backward
        self.device = device
        self.priority = priority
