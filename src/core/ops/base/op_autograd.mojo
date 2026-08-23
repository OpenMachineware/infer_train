# core/ops/base/op_autograd.mojo
#
# M6: the type-erased forward_with_saved / backward dispatchers for every
# registered operator.  These are the functions the `OpRegistry` stores as
# `OpInfo.forward_with_saved` / `OpInfo.backward`; each one rebuilds the
# typed tensors from `AnyTensor` lists, calls the typed kernel from the
# per-op module, and packs the results back into `List[AnyTensor]`.
#
# Conventions:
#   * `saved` lists are flat `List[AnyTensor]`; each op documents its layout.
#   * backward returns ONE gradient per forward input, in input order; a
#     zero-numel `AnyTensor` (`no_grad_any()`) marks "no gradient" (e.g. the
#     integer token ids of an embedding lookup).
#   * GPU entries delegate to the CPU kernels until the Metal backend lands
#     (same policy as the M1-M5 forward dispatch).
#
# Shared helpers also live here: the no-grad sentinel, ones_like for the
# loss seed gradient, and the erased gradient accumulator used by the
# interpreter's `run_with_grad`.

from ...device import Device
from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from .op_interface import AnyTensor, from_any, to_any
from ..cpu.matmul_cpu import (
    matmul_cpu_dynamic,
    matmul_cpu_backward,
    matmul_weight_cpu,
    matmul_weight_cpu_backward,
    matmul_weight_3_threaded,
)
from ..cpu.add_cpu import (
    add_cpu_dynamic,
    add_cpu_backward,
    add_row_cpu,
    add_row_cpu_backward,
)
from ..cpu.rms_norm_cpu import (
    rms_norm_cpu_dynamic,
    rms_norm_cpu_backward,
    rms_norm_weight_cpu_forward_with_saved,
    rms_norm_weight_cpu_backward,
)
from ..cpu.softmax_cpu import (
    softmax_cpu_dynamic,
    softmax_cpu_backward,
)
from ..cpu.rope_cpu import rope_cpu_dynamic, rope_cpu_backward
from ..cpu.swiglu_cpu import (
    swiglu_cpu_dynamic,
    swiglu_cpu_backward,
)
from ..cpu.embedding_cpu import embedding_cpu_dynamic, embedding_cpu_backward
from ..loss.cross_entropy import cross_entropy_forward, cross_entropy_backward
from ..attention.mha import _mha_seq_forward_typed, _mha_seq_backward_typed
from std.memory.alloc import unsafe_alloc
from std.utils.static_tuple import StaticTuple
from std.math import exp, sqrt


# -- shared helpers -----------------------------------------------------------
#


def no_grad_any() -> AnyTensor:
    """A zero-numel AnyTensor used as the "no gradient" sentinel."""
    var buf = unsafe_alloc[UInt8](1)
    var shape = StaticTuple[Int, 8](fill=0)
    return AnyTensor(
        DType.float32, 1, shape, 0, Device.CPU, buf
    )


def _is_no_grad(t: AnyTensor) -> Bool:
    return t.numel == 0


def ones_like_any(t: AnyTensor) -> AnyTensor:
    """A tensor of ones with `t`'s dtype and shape (loss seed gradient)."""
    if t.dtype == DType.float32 and t.rank == 1:
        var out = tensor_zeros[DType.float32, 1](
            _shape_n[1](t.shape)
        )
        for i in range(out.numel()):
            out.set(i, Scalar[DType.float32](Float32(1.0)))
        return to_any[DType.float32, 1](out)
    if t.dtype == DType.float16 and t.rank == 1:
        var out = tensor_zeros[DType.float16, 1](
            _shape_n[1](t.shape)
        )
        for i in range(out.numel()):
            out.set(i, Scalar[DType.float16](Float32(1.0)))
        return to_any[DType.float16, 1](out)
    if t.dtype == DType.float32 and t.rank == 2:
        var out = tensor_zeros[DType.float32, 2](
            _shape_n[2](t.shape)
        )
        for i in range(out.numel()):
            out.set(i, Scalar[DType.float32](Float32(1.0)))
        return to_any[DType.float32, 2](out)
    if t.dtype == DType.float16 and t.rank == 2:
        var out = tensor_zeros[DType.float16, 2](
            _shape_n[2](t.shape)
        )
        for i in range(out.numel()):
            out.set(i, Scalar[DType.float16](Float32(1.0)))
        return to_any[DType.float16, 2](out)
    unimplemented("ones_like_any: unsupported dtype/rank")
    return no_grad_any()


def _shape_n[rank: Int](
    shape: StaticTuple[Int, 8]
) -> StaticTuple[Int, rank]:
    var result = StaticTuple[Int, rank](fill=0)
    for i in range(rank):
        result[i] = shape[i]
    return result


def accumulate_any(mut dest: AnyTensor, delta: AnyTensor):
    """dest += delta in place (same dtype/shape/rank required)."""
    if dest.dtype == DType.float32 and dest.rank == 1:
        var d = from_any[DType.float32, 1](dest)
        var g = from_any[DType.float32, 1](delta)
        for i in range(d.numel()):
            d.set(
                i, Scalar[DType.float32](Float32(d.get(i)) + Float32(g.get(i)))
            )
        return
    if dest.dtype == DType.float16 and dest.rank == 1:
        var d = from_any[DType.float16, 1](dest)
        var g = from_any[DType.float16, 1](delta)
        for i in range(d.numel()):
            d.set(
                i, Scalar[DType.float16](Float32(d.get(i)) + Float32(g.get(i)))
            )
        return
    if dest.dtype == DType.float32 and dest.rank == 2:
        var d = from_any[DType.float32, 2](dest)
        var g = from_any[DType.float32, 2](delta)
        for i in range(d.numel()):
            d.set(
                i, Scalar[DType.float32](Float32(d.get(i)) + Float32(g.get(i)))
            )
        return
    if dest.dtype == DType.float16 and dest.rank == 2:
        var d = from_any[DType.float16, 2](dest)
        var g = from_any[DType.float16, 2](delta)
        for i in range(d.numel()):
            d.set(
                i, Scalar[DType.float16](Float32(d.get(i)) + Float32(g.get(i)))
            )
        return
    if dest.dtype == DType.float32 and dest.rank == 3:
        var d = from_any[DType.float32, 3](dest)
        var g = from_any[DType.float32, 3](delta)
        for i in range(d.numel()):
            d.set(
                i, Scalar[DType.float32](Float32(d.get(i)) + Float32(g.get(i)))
            )
        return
    if dest.dtype == DType.float16 and dest.rank == 3:
        var d = from_any[DType.float16, 3](dest)
        var g = from_any[DType.float16, 3](delta)
        for i in range(d.numel()):
            d.set(
                i, Scalar[DType.float16](Float32(d.get(i)) + Float32(g.get(i)))
            )
        return
    unimplemented("accumulate_any: unsupported dtype/rank")


def _rebuild2[dtype: DType](
    saved: List[AnyTensor]
) -> List[Tensor[dtype, 2]]:
    """Rebuild a typed rank-2 saved list from erased AnyTensors."""
    var result = List[Tensor[dtype, 2]]()
    result.reserve(len(saved))
    for t in saved:
        result.append(from_any[dtype, 2](t))
    return result^


# -- matmul -------------------------------------------------------------------
#


def matmul_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var a = from_any[DType.float32, 2](inputs[0])
        var b = from_any[DType.float32, 2](inputs[1])
        var out = matmul_cpu_dynamic[DType.float32](a, b)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](a))
        saved.append(to_any[DType.float32, 2](b))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var a = from_any[DType.float16, 2](inputs[0])
        var b = from_any[DType.float16, 2](inputs[1])
        var out = matmul_cpu_dynamic[DType.float16](a, b)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](a))
        saved.append(to_any[DType.float16, 2](b))
        return (outputs^, saved^)
    unimplemented("matmul_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def matmul_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var grads = matmul_cpu_backward[DType.float32](
            g, _rebuild2[DType.float32](saved)
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        results.append(to_any[DType.float32, 2](grads[1]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var grads = matmul_cpu_backward[DType.float16](
            g, _rebuild2[DType.float16](saved)
        )
        results.append(to_any[DType.float16, 2](grads[0]))
        results.append(to_any[DType.float16, 2](grads[1]))
        return results^
    unimplemented("matmul_bwd_cpu: unsupported dtype")
    return results^


def matmul_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return matmul_fws_cpu(inputs)


def matmul_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return matmul_bwd_cpu(grad_outputs, saved)


# -- lm_head (weight-major linear) --------------------------------------------


def lm_head_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var w = from_any[DType.float32, 2](inputs[1])
        var out = matmul_weight_cpu[DType.float32](x, w)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](x))
        saved.append(to_any[DType.float32, 2](w))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var w = from_any[DType.float16, 2](inputs[1])
        var out = matmul_weight_cpu[DType.float16](x, w)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](x))
        saved.append(to_any[DType.float16, 2](w))
        return (outputs^, saved^)
    unimplemented("lm_head_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def lm_head_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var grads = matmul_weight_cpu_backward[DType.float32](
            g, _rebuild2[DType.float32](saved)
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        results.append(to_any[DType.float32, 2](grads[1]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var grads = matmul_weight_cpu_backward[DType.float16](
            g, _rebuild2[DType.float16](saved)
        )
        results.append(to_any[DType.float16, 2](grads[0]))
        results.append(to_any[DType.float16, 2](grads[1]))
        return results^
    unimplemented("lm_head_bwd_cpu: unsupported dtype")
    return results^


def lm_head_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return lm_head_fws_cpu(inputs)


def lm_head_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return lm_head_bwd_cpu(grad_outputs, saved)


# -- add ----------------------------------------------------------------------


def add_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var a = from_any[DType.float32, 2](inputs[0])
        var b = from_any[DType.float32, 2](inputs[1])
        var out = add_cpu_dynamic[DType.float32](a, b)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](a))
        saved.append(to_any[DType.float32, 2](b))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var a = from_any[DType.float16, 2](inputs[0])
        var b = from_any[DType.float16, 2](inputs[1])
        var out = add_cpu_dynamic[DType.float16](a, b)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](a))
        saved.append(to_any[DType.float16, 2](b))
        return (outputs^, saved^)
    unimplemented("add_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def add_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var grads = add_cpu_backward[DType.float32, 0, 0](
            g, _rebuild2[DType.float32](saved)
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        results.append(to_any[DType.float32, 2](grads[1]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var grads = add_cpu_backward[DType.float16, 0, 0](
            g, _rebuild2[DType.float16](saved)
        )
        results.append(to_any[DType.float16, 2](grads[0]))
        results.append(to_any[DType.float16, 2](grads[1]))
        return results^
    unimplemented("add_bwd_cpu: unsupported dtype")
    return results^


def add_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return add_fws_cpu(inputs)


def add_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return add_bwd_cpu(grad_outputs, saved)


# -- add_bias -----------------------------------------------------------------


def add_bias_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var bias = from_any[DType.float32, 1](inputs[1])
        var out = add_row_cpu[DType.float32](x, bias)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](x))
        saved.append(to_any[DType.float32, 1](bias))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var bias = from_any[DType.float16, 1](inputs[1])
        var out = add_row_cpu[DType.float16](x, bias)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](x))
        saved.append(to_any[DType.float16, 1](bias))
        return (outputs^, saved^)
    unimplemented("add_bias_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def add_bias_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var bias = from_any[DType.float32, 1](saved[1])
        var grads = add_row_cpu_backward[DType.float32](g, bias)
        results.append(to_any[DType.float32, 2](grads[0]))
        results.append(to_any[DType.float32, 1](grads[1]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var bias = from_any[DType.float16, 1](saved[1])
        var grads = add_row_cpu_backward[DType.float16](g, bias)
        results.append(to_any[DType.float16, 2](grads[0]))
        results.append(to_any[DType.float16, 1](grads[1]))
        return results^
    unimplemented("add_bias_bwd_cpu: unsupported dtype")
    return results^


def add_bias_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return add_bias_fws_cpu(inputs)


def add_bias_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return add_bias_bwd_cpu(grad_outputs, saved)


# -- rms_norm -----------------------------------------------------------------


def rms_norm_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var out = rms_norm_cpu_dynamic[DType.float32](x)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](x))
        saved.append(to_any[DType.float32, 2](out))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var out = rms_norm_cpu_dynamic[DType.float16](x)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](x))
        saved.append(to_any[DType.float16, 2](out))
        return (outputs^, saved^)
    unimplemented("rms_norm_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def rms_norm_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var grads = rms_norm_cpu_backward[DType.float32, 0](
            g, _rebuild2[DType.float32](saved)
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var grads = rms_norm_cpu_backward[DType.float16, 0](
            g, _rebuild2[DType.float16](saved)
        )
        results.append(to_any[DType.float16, 2](grads[0]))
        return results^
    unimplemented("rms_norm_bwd_cpu: unsupported dtype")
    return results^


def rms_norm_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return rms_norm_fws_cpu(inputs)


def rms_norm_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return rms_norm_bwd_cpu(grad_outputs, saved)


# -- rms_norm_weight ----------------------------------------------------------
# saved = [x, z, w] where z = x / r (the unweighted normalized tensor).


def rms_norm_weight_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var w = from_any[DType.float32, 1](inputs[1])
        var fws = rms_norm_weight_cpu_forward_with_saved[DType.float32](x, w)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](fws[0]))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](fws[1][0]))
        saved.append(to_any[DType.float32, 2](fws[1][1]))
        saved.append(to_any[DType.float32, 1](w))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var w = from_any[DType.float16, 1](inputs[1])
        var fws = rms_norm_weight_cpu_forward_with_saved[DType.float16](x, w)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](fws[0]))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](fws[1][0]))
        saved.append(to_any[DType.float16, 2](fws[1][1]))
        saved.append(to_any[DType.float16, 1](w))
        return (outputs^, saved^)
    unimplemented("rms_norm_weight_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def rms_norm_weight_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var w = from_any[DType.float32, 1](saved[2])
        var xz = _rebuild2[DType.float32](_first_two(saved))
        var grads = rms_norm_weight_cpu_backward[DType.float32](g, xz, w)
        results.append(to_any[DType.float32, 2](grads[0]))
        results.append(to_any[DType.float32, 1](grads[1]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var w = from_any[DType.float16, 1](saved[2])
        var xz = _rebuild2[DType.float16](_first_two(saved))
        var grads = rms_norm_weight_cpu_backward[DType.float16](g, xz, w)
        results.append(to_any[DType.float16, 2](grads[0]))
        results.append(to_any[DType.float16, 1](grads[1]))
        return results^
    unimplemented("rms_norm_weight_bwd_cpu: unsupported dtype")
    return results^


def _first_two(saved: List[AnyTensor]) -> List[AnyTensor]:
    var result = List[AnyTensor]()
    result.append(saved[0])
    result.append(saved[1])
    return result^


def rms_norm_weight_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return rms_norm_weight_fws_cpu(inputs)


def rms_norm_weight_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return rms_norm_weight_bwd_cpu(grad_outputs, saved)


# -- softmax ------------------------------------------------------------------


def softmax_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var out = softmax_cpu_dynamic[DType.float32](x)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](out))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var out = softmax_cpu_dynamic[DType.float16](x)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](out))
        return (outputs^, saved^)
    unimplemented("softmax_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def softmax_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var grads = softmax_cpu_backward[DType.float32, 0](
            g, _rebuild2[DType.float32](saved)
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var grads = softmax_cpu_backward[DType.float16, 0](
            g, _rebuild2[DType.float16](saved)
        )
        results.append(to_any[DType.float16, 2](grads[0]))
        return results^
    unimplemented("softmax_bwd_cpu: unsupported dtype")
    return results^


def softmax_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return softmax_fws_cpu(inputs)


def softmax_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return softmax_bwd_cpu(grad_outputs, saved)


# -- rope ---------------------------------------------------------------------
# saved = [x, pos(f32 [1])]


def rope_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 3](inputs[0])
        var pos_t = from_any[DType.float32, 1](inputs[1])
        var out = rope_cpu_dynamic[DType.float32](
            x, Int(Float32(pos_t.get(0)))
        )
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 3](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(inputs[0])
        saved.append(inputs[1])
        return (outputs^, saved^)
    if dtype == DType.float16:
        var x = from_any[DType.float16, 3](inputs[0])
        var pos_t = from_any[DType.float32, 1](inputs[1])
        var out = rope_cpu_dynamic[DType.float16](
            x, Int(Float32(pos_t.get(0)))
        )
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 3](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(inputs[0])
        saved.append(inputs[1])
        return (outputs^, saved^)
    unimplemented("rope_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def rope_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    var pos_t = from_any[DType.float32, 1](saved[1])
    var start_pos = Int(Float32(pos_t.get(0)))
    if dtype == DType.float32:
        var g = from_any[DType.float32, 3](grad_outputs[0])
        var x = from_any[DType.float32, 3](saved[0])
        var grad = rope_cpu_backward[DType.float32, 0, 0](
            g, x, start_pos
        )
        results.append(to_any[DType.float32, 3](grad))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 3](grad_outputs[0])
        var x = from_any[DType.float16, 3](saved[0])
        var grad = rope_cpu_backward[DType.float16, 0, 0](
            g, x, start_pos
        )
        results.append(to_any[DType.float16, 3](grad))
        return results^
    unimplemented("rope_bwd_cpu: unsupported dtype")
    return results^


def rope_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return rope_fws_cpu(inputs)


def rope_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return rope_bwd_cpu(grad_outputs, saved)


# -- swiglu -------------------------------------------------------------------


def swiglu_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var gate = from_any[DType.float32, 2](inputs[0])
        var up = from_any[DType.float32, 2](inputs[1])
        var out = swiglu_cpu_dynamic[DType.float32](gate, up)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](gate))
        saved.append(to_any[DType.float32, 2](up))
        return (outputs^, saved^)
    if dtype == DType.float16:
        var gate = from_any[DType.float16, 2](inputs[0])
        var up = from_any[DType.float16, 2](inputs[1])
        var out = swiglu_cpu_dynamic[DType.float16](gate, up)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](gate))
        saved.append(to_any[DType.float16, 2](up))
        return (outputs^, saved^)
    unimplemented("swiglu_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def swiglu_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var grads = swiglu_cpu_backward[DType.float32, 0, 0](
            g, _rebuild2[DType.float32](saved)
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        results.append(to_any[DType.float32, 2](grads[1]))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var grads = swiglu_cpu_backward[DType.float16, 0, 0](
            g, _rebuild2[DType.float16](saved)
        )
        results.append(to_any[DType.float16, 2](grads[0]))
        results.append(to_any[DType.float16, 2](grads[1]))
        return results^
    unimplemented("swiglu_bwd_cpu: unsupported dtype")
    return results^


def swiglu_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return swiglu_fws_cpu(inputs)


def swiglu_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return swiglu_bwd_cpu(grad_outputs, saved)


# -- embedding ----------------------------------------------------------------
# saved = [tokens(i32), table]


def embedding_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[1].dtype
    var tokens = from_any[DType.int32, 1](inputs[0])
    if dtype == DType.float32:
        var table = from_any[DType.float32, 2](inputs[1])
        var out = embedding_cpu_dynamic[DType.float32](tokens, table)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(inputs[0])
        saved.append(inputs[1])
        return (outputs^, saved^)
    if dtype == DType.float16:
        var table = from_any[DType.float16, 2](inputs[1])
        var out = embedding_cpu_dynamic[DType.float16](tokens, table)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(inputs[0])
        saved.append(inputs[1])
        return (outputs^, saved^)
    unimplemented("embedding_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def embedding_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    var tokens = from_any[DType.int32, 1](saved[0])
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        var table = from_any[DType.float32, 2](saved[1])
        var grad = embedding_cpu_backward[DType.float32, 0, 0](
            g, tokens, table
        )
        results.append(no_grad_any())
        results.append(to_any[DType.float32, 2](grad))
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        var table = from_any[DType.float16, 2](saved[1])
        var grad = embedding_cpu_backward[DType.float16, 0, 0](
            g, tokens, table
        )
        results.append(no_grad_any())
        results.append(to_any[DType.float16, 2](grad))
        return results^
    unimplemented("embedding_bwd_cpu: unsupported dtype")
    return results^


def embedding_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return embedding_fws_cpu(inputs)


def embedding_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return embedding_bwd_cpu(grad_outputs, saved)


# -- swiglu_ffn ---------------------------------------------------------------
# saved = [x, g, u, h, gw, uw, dw]


def swiglu_ffn_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](inputs[0])
        var gw = from_any[DType.float32, 2](inputs[1])
        var uw = from_any[DType.float32, 2](inputs[2])
        var dw = from_any[DType.float32, 2](inputs[3])
        var g = matmul_weight_cpu[DType.float32](x, gw)
        var u = matmul_weight_cpu[DType.float32](x, uw)
        var h = swiglu_cpu_dynamic[DType.float32](g, u)
        var out = matmul_weight_cpu[DType.float32](h, dw)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float32, 2](x))
        saved.append(to_any[DType.float32, 2](g))
        saved.append(to_any[DType.float32, 2](u))
        saved.append(to_any[DType.float32, 2](h))
        saved.append(inputs[1])
        saved.append(inputs[2])
        saved.append(inputs[3])
        return (outputs^, saved^)
    if dtype == DType.float16:
        var x = from_any[DType.float16, 2](inputs[0])
        var gw = from_any[DType.float16, 2](inputs[1])
        var uw = from_any[DType.float16, 2](inputs[2])
        var dw = from_any[DType.float16, 2](inputs[3])
        var g = matmul_weight_cpu[DType.float16](x, gw)
        var u = matmul_weight_cpu[DType.float16](x, uw)
        var h = swiglu_cpu_dynamic[DType.float16](g, u)
        var out = matmul_weight_cpu[DType.float16](h, dw)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 2](out))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(to_any[DType.float16, 2](x))
        saved.append(to_any[DType.float16, 2](g))
        saved.append(to_any[DType.float16, 2](u))
        saved.append(to_any[DType.float16, 2](h))
        saved.append(inputs[1])
        saved.append(inputs[2])
        saved.append(inputs[3])
        return (outputs^, saved^)
    unimplemented("swiglu_ffn_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def _swiglu_ffn_bwd_typed[dtype: DType](
    grad_out: Tensor[dtype, 2],
    x: Tensor[dtype, 2],
    g: Tensor[dtype, 2],
    u: Tensor[dtype, 2],
    h: Tensor[dtype, 2],
    gw: Tensor[dtype, 2],
    uw: Tensor[dtype, 2],
    dw: Tensor[dtype, 2],
) -> List[AnyTensor]:
    var results = List[AnyTensor]()
    results.reserve(8)
    # down projection: grad_h = grad_out @ dw; grad_dw = grad_out^T @ h
    var down_saved = List[Tensor[dtype, 2]]()
    down_saved.append(h)
    down_saved.append(dw)
    var down_grads = matmul_weight_cpu_backward[dtype](grad_out, down_saved)
    var grad_h = down_grads[0]
    var grad_dw = down_grads[1]
    # swiglu: grad_g, grad_u
    var sw_saved = List[Tensor[dtype, 2]]()
    sw_saved.append(g)
    sw_saved.append(u)
    var sw_grads = swiglu_cpu_backward[dtype, 0, 0](grad_h, sw_saved)
    var grad_g = sw_grads[0]
    var grad_u = sw_grads[1]
    # gate/up projections
    var gate_saved = List[Tensor[dtype, 2]]()
    gate_saved.append(x)
    gate_saved.append(gw)
    var gate_grads = matmul_weight_cpu_backward[dtype](grad_g, gate_saved)
    var up_saved = List[Tensor[dtype, 2]]()
    up_saved.append(x)
    up_saved.append(uw)
    var up_grads = matmul_weight_cpu_backward[dtype](grad_u, up_saved)
    # grad_x accumulates
    var grad_x = tensor_zeros[dtype, 2](x.shape())
    for i in range(x.numel()):
        grad_x.set(
            i,
            Scalar[dtype](
                Float32(gate_grads[0].get(i)) + Float32(up_grads[0].get(i))
            ),
        )
    results.append(to_any[dtype, 2](grad_x))
    results.append(to_any[dtype, 2](gate_grads[1]))
    results.append(to_any[dtype, 2](up_grads[1]))
    results.append(to_any[dtype, 2](grad_dw))
    return results^


def swiglu_ffn_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    if dtype == DType.float32:
        var g = from_any[DType.float32, 2](grad_outputs[0])
        return _swiglu_ffn_bwd_typed[DType.float32](
            g,
            from_any[DType.float32, 2](saved[0]),
            from_any[DType.float32, 2](saved[1]),
            from_any[DType.float32, 2](saved[2]),
            from_any[DType.float32, 2](saved[3]),
            from_any[DType.float32, 2](saved[4]),
            from_any[DType.float32, 2](saved[5]),
            from_any[DType.float32, 2](saved[6]),
        )
    if dtype == DType.float16:
        var g = from_any[DType.float16, 2](grad_outputs[0])
        return _swiglu_ffn_bwd_typed[DType.float16](
            g,
            from_any[DType.float16, 2](saved[0]),
            from_any[DType.float16, 2](saved[1]),
            from_any[DType.float16, 2](saved[2]),
            from_any[DType.float16, 2](saved[3]),
            from_any[DType.float16, 2](saved[4]),
            from_any[DType.float16, 2](saved[5]),
            from_any[DType.float16, 2](saved[6]),
        )
    unimplemented("swiglu_ffn_bwd_cpu: unsupported dtype")
    return List[AnyTensor]()


def swiglu_ffn_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return swiglu_ffn_fws_cpu(inputs)


def swiglu_ffn_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return swiglu_ffn_bwd_cpu(grad_outputs, saved)


# -- mha_seq ------------------------------------------------------------------
# inputs: [x, wq, wk, wv, wo, bq, bk, bv, cfg(i32 [n_heads, n_kv_heads,
# head_dim]), pos_theta(f32 [start_pos, theta])]
# saved: [x, q_rot, k_rot, v3, p, o_flat, pos_theta, wq, wk, wv, wo, bq, bk, bv]


def mha_seq_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    if dtype != DType.float32 and dtype != DType.float16:
        unimplemented("mha_seq_fws_cpu: unsupported dtype")
    var cfg = from_any[DType.int32, 1](inputs[8])
    var n_heads = Int(cfg.get(0))
    var n_kv_heads = Int(cfg.get(1))
    var head_dim = Int(cfg.get(2))
    var pos_t = from_any[DType.float32, 1](inputs[9])
    var start_pos = Int(Float32(pos_t.get(0)))
    var theta = Float32(pos_t.get(1))
    if dtype == DType.float32:
        var res = _mha_seq_forward_typed[DType.float32](
            from_any[DType.float32, 2](inputs[0]),
            from_any[DType.float32, 2](inputs[1]),
            from_any[DType.float32, 2](inputs[2]),
            from_any[DType.float32, 2](inputs[3]),
            from_any[DType.float32, 2](inputs[4]),
            from_any[DType.float32, 1](inputs[5]),
            from_any[DType.float32, 1](inputs[6]),
            from_any[DType.float32, 1](inputs[7]),
            start_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            theta,
        )
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 2](res[0]))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(inputs[0])
        saved.append(to_any[DType.float32, 3](res[1]))
        saved.append(to_any[DType.float32, 3](res[2]))
        saved.append(to_any[DType.float32, 3](res[3]))
        saved.append(to_any[DType.float32, 3](res[4]))
        saved.append(to_any[DType.float32, 2](res[5]))
        saved.append(inputs[9])
        for i in range(1, 8):
            saved.append(inputs[i])
        return (outputs^, saved^)
    var res = _mha_seq_forward_typed[DType.float16](
        from_any[DType.float16, 2](inputs[0]),
        from_any[DType.float16, 2](inputs[1]),
        from_any[DType.float16, 2](inputs[2]),
        from_any[DType.float16, 2](inputs[3]),
        from_any[DType.float16, 2](inputs[4]),
        from_any[DType.float16, 1](inputs[5]),
        from_any[DType.float16, 1](inputs[6]),
        from_any[DType.float16, 1](inputs[7]),
        start_pos,
        n_heads,
        n_kv_heads,
        head_dim,
        theta,
    )
    var outputs = List[AnyTensor]()
    outputs.reserve(8)
    outputs.append(to_any[DType.float16, 2](res[0]))
    var saved = List[AnyTensor]()
    saved.reserve(16)
    saved.append(inputs[0])
    saved.append(to_any[DType.float16, 3](res[1]))
    saved.append(to_any[DType.float16, 3](res[2]))
    saved.append(to_any[DType.float16, 3](res[3]))
    saved.append(to_any[DType.float16, 3](res[4]))
    saved.append(to_any[DType.float16, 2](res[5]))
    saved.append(inputs[9])
    for i in range(1, 8):
        saved.append(inputs[i])
    return (outputs^, saved^)


def mha_seq_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    var pos_t = from_any[DType.float32, 1](saved[6])
    var start_pos = Int(Float32(pos_t.get(0)))
    var theta = Float32(pos_t.get(1))
    var q_rot = saved[1]
    var n_heads = q_rot.shape[0]
    var head_dim = q_rot.shape[2]
    var k_rot = saved[2]
    var n_kv_heads = k_rot.shape[0]
    if dtype == DType.float32:
        var grads = _mha_seq_backward_typed[DType.float32](
            from_any[DType.float32, 2](grad_outputs[0]),
            from_any[DType.float32, 2](saved[0]),
            from_any[DType.float32, 2](saved[7]),
            from_any[DType.float32, 2](saved[8]),
            from_any[DType.float32, 2](saved[9]),
            from_any[DType.float32, 2](saved[10]),
            from_any[DType.float32, 1](saved[11]),
            from_any[DType.float32, 1](saved[12]),
            from_any[DType.float32, 1](saved[13]),
            from_any[DType.float32, 3](saved[1]),
            from_any[DType.float32, 3](saved[2]),
            from_any[DType.float32, 3](saved[3]),
            from_any[DType.float32, 3](saved[4]),
            from_any[DType.float32, 2](saved[5]),
            start_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            theta,
        )
        results.append(to_any[DType.float32, 2](grads[0]))
        results.append(to_any[DType.float32, 2](grads[1]))
        results.append(to_any[DType.float32, 2](grads[2]))
        results.append(to_any[DType.float32, 2](grads[3]))
        results.append(to_any[DType.float32, 2](grads[4]))
        results.append(to_any[DType.float32, 1](grads[5]))
        results.append(to_any[DType.float32, 1](grads[6]))
        results.append(to_any[DType.float32, 1](grads[7]))
        results.append(no_grad_any())  # cfg
        results.append(no_grad_any())  # pos_theta
        return results^
    if dtype == DType.float16:
        var grads = _mha_seq_backward_typed[DType.float16](
            from_any[DType.float16, 2](grad_outputs[0]),
            from_any[DType.float16, 2](saved[0]),
            from_any[DType.float16, 2](saved[7]),
            from_any[DType.float16, 2](saved[8]),
            from_any[DType.float16, 2](saved[9]),
            from_any[DType.float16, 2](saved[10]),
            from_any[DType.float16, 1](saved[11]),
            from_any[DType.float16, 1](saved[12]),
            from_any[DType.float16, 1](saved[13]),
            from_any[DType.float16, 3](saved[1]),
            from_any[DType.float16, 3](saved[2]),
            from_any[DType.float16, 3](saved[3]),
            from_any[DType.float16, 3](saved[4]),
            from_any[DType.float16, 2](saved[5]),
            start_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            theta,
        )
        results.append(to_any[DType.float16, 2](grads[0]))
        results.append(to_any[DType.float16, 2](grads[1]))
        results.append(to_any[DType.float16, 2](grads[2]))
        results.append(to_any[DType.float16, 2](grads[3]))
        results.append(to_any[DType.float16, 2](grads[4]))
        results.append(to_any[DType.float16, 1](grads[5]))
        results.append(to_any[DType.float16, 1](grads[6]))
        results.append(to_any[DType.float16, 1](grads[7]))
        results.append(no_grad_any())
        results.append(no_grad_any())
        return results^
    unimplemented("mha_seq_bwd_cpu: unsupported dtype")
    return results^


def mha_seq_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return mha_seq_fws_cpu(inputs)


def mha_seq_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return mha_seq_bwd_cpu(grad_outputs, saved)


# -- cross_entropy ------------------------------------------------------------
# inputs: [logits(B,V), targets(B) i32]; output: scalar loss as rank-1 [1].


def cross_entropy_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    var targets = from_any[DType.int32, 1](inputs[1])
    if dtype == DType.float32:
        var logits = from_any[DType.float32, 2](inputs[0])
        var loss = cross_entropy_forward[DType.float32](logits, targets)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float32, 1](loss))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(inputs[0])
        saved.append(inputs[1])
        return (outputs^, saved^)
    if dtype == DType.float16:
        var logits = from_any[DType.float16, 2](inputs[0])
        var loss = cross_entropy_forward[DType.float16](logits, targets)
        var outputs = List[AnyTensor]()
        outputs.reserve(8)
        outputs.append(to_any[DType.float16, 1](loss))
        var saved = List[AnyTensor]()
        saved.reserve(16)
        saved.append(inputs[0])
        saved.append(inputs[1])
        return (outputs^, saved^)
    unimplemented("cross_entropy_fws_cpu: unsupported dtype")
    return (List[AnyTensor](), List[AnyTensor]())


def cross_entropy_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    var targets = from_any[DType.int32, 1](saved[1])
    if dtype == DType.float32:
        var g = from_any[DType.float32, 1](grad_outputs[0])
        var logits = from_any[DType.float32, 2](saved[0])
        var grad = cross_entropy_backward[DType.float32](g, logits, targets)
        results.append(to_any[DType.float32, 2](grad))
        results.append(no_grad_any())
        return results^
    if dtype == DType.float16:
        var g = from_any[DType.float16, 1](grad_outputs[0])
        var logits = from_any[DType.float16, 2](saved[0])
        var grad = cross_entropy_backward[DType.float16](g, logits, targets)
        results.append(to_any[DType.float16, 2](grad))
        results.append(no_grad_any())
        return results^
    unimplemented("cross_entropy_bwd_cpu: unsupported dtype")
    return results^


def cross_entropy_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return cross_entropy_fws_cpu(inputs)


def cross_entropy_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return cross_entropy_bwd_cpu(grad_outputs, saved)


# -- identity (pass-through entry op) -----------------------------------------


def identity_fws(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var outputs = List[AnyTensor]()
    outputs.reserve(8)
    for t in inputs:
        outputs.append(t)
    return (outputs^, List[AnyTensor]())


def identity_bwd(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    _ = saved
    var results = List[AnyTensor]()
    results.reserve(8)
    for t in grad_outputs:
        results.append(t)
    return results^


# -- dynamic quantize / dequantize (M6 Phase 7) -------------------------------


def _round_i(v: Float32) -> Int:
    """Round half away from zero to Int."""
    if v >= Float32(0.0):
        return Int(v + Float32(0.5))
    return Int(v - Float32(0.5))
# inputs: [x(B,N)]; outputs: [q(i8 as i32-coded), scale(f32 [1]), zp(f32 [1])]


def dynamic_quantize_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    var dtype = inputs[0].dtype
    var x_any = inputs[0]
    var numel = x_any.numel
    var cols = x_any.shape[1]
    var rows = numel // cols
    if cols < 1:
        cols = 1
    var scale = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
    var zp = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](1))
    var q = tensor_zeros[DType.int32, 2](
        StaticTuple[Int, 2](rows, cols)
    )
    var mx = Float32(0)
    var mn = Float32(0)
    var first = True
    if dtype == DType.float32:
        var x = from_any[DType.float32, 2](x_any)
        for i in range(numel):
            var v = Float32(x.get(i))
            if first:
                mx = v
                mn = v
                first = False
            elif v > mx:
                mx = v
            elif v < mn:
                mn = v
        # asymmetric int8 quantization
        var s = (mx - mn) / Float32(255.0)
        if s < Float32(1e-12):
            s = Float32(1.0)
        scale.set(0, Scalar[DType.float32](s))
        zp.set(0, Scalar[DType.float32](mn))
        for i in range(numel):
            var v = Float32(x.get(i))
            var qv = _round_i((v - mn) / s)
            if qv < 0:
                qv = 0
            if qv > 255:
                qv = 255
            q.set(i, Scalar[DType.int32](qv - 128))
    elif dtype == DType.float16:
        var x = from_any[DType.float16, 2](x_any)
        for i in range(numel):
            var v = Float32(x.get(i))
            if first:
                mx = v
                mn = v
                first = False
            elif v > mx:
                mx = v
            elif v < mn:
                mn = v
        var s = (mx - mn) / Float32(255.0)
        if s < Float32(1e-12):
            s = Float32(1.0)
        scale.set(0, Scalar[DType.float32](s))
        zp.set(0, Scalar[DType.float32](mn))
        for i in range(numel):
            var v = Float32(x.get(i))
            var qv = _round_i((v - mn) / s)
            if qv < 0:
                qv = 0
            if qv > 255:
                qv = 255
            q.set(i, Scalar[DType.int32](qv - 128))
    else:
        unimplemented("dynamic_quantize_fws_cpu: unsupported dtype")
    var outputs = List[AnyTensor]()
    outputs.reserve(8)
    outputs.append(to_any[DType.int32, 2](q))
    outputs.append(to_any[DType.float32, 1](scale))
    outputs.append(to_any[DType.float32, 1](zp))
    var saved = List[AnyTensor]()
    saved.reserve(16)
    saved.append(to_any[DType.float32, 1](scale))
    saved.append(to_any[DType.float32, 1](zp))
    return (outputs^, saved^)


def dynamic_quantize_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    """Straight-through estimator: the quantization gradient passes through
    unchanged (scale/zero-point receive no gradient)."""
    _ = saved
    var results = List[AnyTensor]()
    results.reserve(8)
    results.append(grad_outputs[0])
    results.append(no_grad_any())
    results.append(no_grad_any())
    return results^


def dynamic_dequantize_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    """x = scale * (q - (-128)) + zero_point (asymmetric int8)."""
    var q = from_any[DType.int32, 2](inputs[0])
    var scale = from_any[DType.float32, 1](inputs[1])
    var zp = from_any[DType.float32, 1](inputs[2])
    var s = Float32(scale.get(0))
    var z = Float32(zp.get(0))
    var numel = q.numel()
    var cols = q.shape()[1]
    var rows = numel // cols
    if cols < 1:
        cols = 1
    var out = tensor_zeros[DType.float32, 2](
        StaticTuple[Int, 2](rows, cols)
    )
    for i in range(numel):
        out.set(i, Scalar[DType.float32](s * Float32(Int(q.get(i)) + 128) + z))
    var outputs = List[AnyTensor]()
    outputs.reserve(8)
    outputs.append(to_any[DType.float32, 2](out))
    var saved = List[AnyTensor]()
    saved.reserve(16)
    saved.append(inputs[1])
    return (outputs^, saved^)


def dynamic_dequantize_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    """grad_q = grad_x * scale (STE; scale/zero-point get no gradient)."""
    var scale = from_any[DType.float32, 1](saved[0])
    var g = from_any[DType.float32, 2](grad_outputs[0])
    var s = Float32(scale.get(0))
    var grad_q = tensor_zeros[DType.int32, 2](g.shape())
    for i in range(g.numel()):
        grad_q.set(i, Scalar[DType.int32](_round_i(Float32(g.get(i)) * s)))
    var results = List[AnyTensor]()
    results.reserve(8)
    results.append(to_any[DType.int32, 2](grad_q))
    results.append(no_grad_any())
    results.append(no_grad_any())
    return results^


# -- mha (cached, single-token decode) ----------------------------------------
#
# The cached decode path's backward treats the KV cache as constant: past
# keys/values receive no gradient (they were produced by earlier steps), so
# only the current token's x / projections / biases get gradients.
# saved = [x, wq, wk, wv, wo, bq, bk, bv, q_rot, k_rot, v3, p, o_flat, pos,
#          kc, vc]


def mha_fws_cpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    """Replay the cached MHA forward while saving the intermediates."""
    var dtype = inputs[0].dtype
    if dtype != DType.float16:
        unimplemented("mha_fws_cpu: cache path is fp16-only")
    var x = from_any[DType.float16, 2](inputs[0])
    var wq = from_any[DType.float16, 2](inputs[1])
    var wk = from_any[DType.float16, 2](inputs[2])
    var wv = from_any[DType.float16, 2](inputs[3])
    var wo = from_any[DType.float16, 2](inputs[4])
    var bq = from_any[DType.float16, 1](inputs[5])
    var bk = from_any[DType.float16, 1](inputs[6])
    var bv = from_any[DType.float16, 1](inputs[7])
    var kc = from_any[DType.float16, 3](inputs[8])
    var vc = from_any[DType.float16, 3](inputs[9])
    var pos_t = from_any[DType.float32, 1](inputs[10])
    var start_pos = Int(Float32(pos_t.get(0)))
    var head_dim = kc.shape()[2]
    var n_kv_heads = kc.shape()[0]
    var n_heads = wq.shape()[0] // head_dim
    var hidden = x.shape()[1]
    var kv_hidden = wk.shape()[0]
    var max_len = kc.shape()[1]

    var q_flat = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](1, hidden)
    )
    var k_flat = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](1, kv_hidden)
    )
    var v_flat = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](1, kv_hidden)
    )
    matmul_weight_3_threaded[DType.float16](x, wq, wk, wv, q_flat, k_flat, v_flat)
    q_flat = add_row_cpu[DType.float16](q_flat, bq)
    k_flat = add_row_cpu[DType.float16](k_flat, bk)
    v_flat = add_row_cpu[DType.float16](v_flat, bv)
    var q3 = Tensor[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, head_dim),
        q_flat.data(),
        q_flat.device(),
    )
    var k3 = Tensor[DType.float16, 3](
        StaticTuple[Int, 3](n_kv_heads, 1, head_dim),
        k_flat.data(),
        k_flat.device(),
    )
    var v3 = Tensor[DType.float16, 3](
        StaticTuple[Int, 3](n_kv_heads, 1, head_dim),
        v_flat.data(),
        v_flat.device(),
    )
    var q_rot = rope_cpu_dynamic[DType.float16](q3, start_pos)
    var k_rot = rope_cpu_dynamic[DType.float16](k3, start_pos)

    # store into the cache (idempotent replay)
    for h in range(n_kv_heads):
        for d in range(head_dim):
            kc.set(
                (h * max_len + start_pos) * head_dim + d,
                Scalar[DType.float16](Float32(k_rot.get(h * head_dim + d))),
            )
            vc.set(
                (h * max_len + start_pos) * head_dim + d,
                Scalar[DType.float16](Float32(v3.get(h * head_dim + d))),
            )

    var seq = start_pos + 1
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var p = tensor_zeros[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, seq)
    )
    var o = tensor_zeros[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, head_dim)
    )
    for h in range(n_heads):
        var kv = h * n_kv_heads // n_heads
        var scores = List[Float32]()
        for t in range(seq):
            var acc = Float32(0)
            for d in range(head_dim):
                acc += Float32(q_rot.get(h * head_dim + d)) * Float32(
                    kc.get((kv * max_len + t) * head_dim + d)
                )
            scores.append(acc * scale)
        var mx = Float32(-3.0e38)
        for i in range(len(scores)):
            if scores[i] > mx:
                mx = scores[i]
        var total = Float32(0)
        for i in range(len(scores)):
            var e = exp(scores[i] - mx)
            scores[i] = e
            total += e
        var inv = Float32(1.0) / total
        for i in range(len(scores)):
            scores[i] = scores[i] * inv
            p.set((h * 1 + 0) * seq + i, Scalar[DType.float16](scores[i]))
        for d in range(head_dim):
            var acc = Float32(0)
            for t in range(seq):
                acc += scores[t] * Float32(
                    vc.get((kv * max_len + t) * head_dim + d)
                )
            o.set(h * head_dim + d, Scalar[DType.float16](acc))
    var o_flat = Tensor[DType.float16, 2](
        StaticTuple[Int, 2](1, hidden), o.data(), o.device()
    )
    var out = matmul_weight_cpu[DType.float16](o_flat, wo)

    var outputs = List[AnyTensor]()
    outputs.reserve(8)
    outputs.append(to_any[DType.float16, 2](out))
    var saved = List[AnyTensor]()
    saved.reserve(16)
    saved.append(inputs[0])
    saved.append(inputs[1])
    saved.append(inputs[2])
    saved.append(inputs[3])
    saved.append(inputs[4])
    saved.append(inputs[5])
    saved.append(inputs[6])
    saved.append(inputs[7])
    saved.append(to_any[DType.float16, 3](q_rot))
    saved.append(to_any[DType.float16, 3](k_rot))
    saved.append(to_any[DType.float16, 3](v3))
    saved.append(to_any[DType.float16, 3](p))
    saved.append(to_any[DType.float16, 2](o_flat))
    saved.append(inputs[10])
    saved.append(inputs[8])
    saved.append(inputs[9])
    return (outputs^, saved^)


def mha_bwd_cpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    """Backward of the cached MHA (KV cache constant; T=1)."""
    var dtype = grad_outputs[0].dtype
    var results = List[AnyTensor]()
    results.reserve(8)
    if dtype != DType.float16:
        unimplemented("mha_bwd_cpu: cache path is fp16-only")
    var grad_out = from_any[DType.float16, 2](grad_outputs[0])
    var x = from_any[DType.float16, 2](saved[0])
    var wq = from_any[DType.float16, 2](saved[1])
    var wk = from_any[DType.float16, 2](saved[2])
    var wv = from_any[DType.float16, 2](saved[3])
    var wo = from_any[DType.float16, 2](saved[4])
    var q_rot = from_any[DType.float16, 3](saved[8])
    var k_rot = from_any[DType.float16, 3](saved[9])
    var v3 = from_any[DType.float16, 3](saved[10])
    var p = from_any[DType.float16, 3](saved[11])
    var o_flat = from_any[DType.float16, 2](saved[12])
    var pos_t = from_any[DType.float32, 1](saved[13])
    var kc = from_any[DType.float16, 3](saved[14])
    var vc = from_any[DType.float16, 3](saved[15])
    var start_pos = Int(Float32(pos_t.get(0)))
    var head_dim = kc.shape()[2]
    var n_kv_heads = kc.shape()[0]
    var n_heads = wq.shape()[0] // head_dim
    var max_len = kc.shape()[1]
    var seq = start_pos + 1
    var hidden = x.shape()[1]
    var kv_hidden = wk.shape()[0]
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    # output projection
    var mw_saved = List[Tensor[DType.float16, 2]]()
    mw_saved.append(o_flat)
    mw_saved.append(wo)
    var mw_grads = matmul_weight_cpu_backward[DType.float16](
        grad_out, mw_saved
    )
    var grad_o_flat = mw_grads[0]
    var grad_wo = mw_grads[1]
    var grad_o = Tensor[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, head_dim),
        grad_o_flat.data(),
        grad_o_flat.device(),
    )

    var grad_qrot = tensor_zeros[DType.float16, 3](q_rot.shape())
    var grad_krot = tensor_zeros[DType.float16, 3](k_rot.shape())
    var grad_v3 = tensor_zeros[DType.float16, 3](v3.shape())
    for h in range(n_heads):
        var kv = h * n_kv_heads // n_heads
        var dot = Float32(0)
        for t2 in range(seq):
            var acc = Float32(0)
            for d in range(head_dim):
                acc += Float32(grad_o.get(h * head_dim + d)) * Float32(
                    vc.get((kv * max_len + t2) * head_dim + d)
                )
            dot += Float32(p.get(h * seq + t2)) * acc
        for t2 in range(seq):
            var acc = Float32(0)
            for d in range(head_dim):
                acc += Float32(grad_o.get(h * head_dim + d)) * Float32(
                    vc.get((kv * max_len + t2) * head_dim + d)
                )
            var pt = Float32(p.get(h * seq + t2))
            var gs = pt * (acc - dot)
            for d in range(head_dim):
                grad_qrot.set(
                    h * head_dim + d,
                    Scalar[DType.float16](
                        Float32(grad_qrot.get(h * head_dim + d))
                        + gs * scale * Float32(
                            kc.get((kv * max_len + t2) * head_dim + d)
                        )
                    ),
                )
                if t2 == start_pos:
                    grad_krot.set(
                        kv * head_dim + d,
                        Scalar[DType.float16](
                            Float32(grad_krot.get(kv * head_dim + d))
                            + gs * scale * Float32(q_rot.get(h * head_dim + d))
                        ),
                    )
                grad_v3.set(
                    kv * head_dim + d,
                    Scalar[DType.float16](
                        Float32(grad_v3.get(kv * head_dim + d))
                        + pt * Float32(grad_o.get(h * head_dim + d))
                    ),
                )

    # RoPE backward
    var grad_q = rope_cpu_backward[DType.float16, 0, 0](
        grad_qrot, q_rot, start_pos
    )
    var grad_k = rope_cpu_backward[DType.float16, 0, 0](
        grad_krot, k_rot, start_pos
    )
    var grad_q_flat = Tensor[DType.float16, 2](
        StaticTuple[Int, 2](1, hidden), grad_q.data(), grad_q.device()
    )
    var grad_k_flat = Tensor[DType.float16, 2](
        StaticTuple[Int, 2](1, kv_hidden), grad_k.data(), grad_k.device()
    )
    var grad_v_flat = Tensor[DType.float16, 2](
        StaticTuple[Int, 2](1, kv_hidden), grad_v3.data(), grad_v3.device()
    )
    var grad_bq = tensor_zeros[DType.float16, 1](bq_shape_from(wq))
    var grad_bk = tensor_zeros[DType.float16, 1](bq_shape_from(wk))
    var grad_bv = tensor_zeros[DType.float16, 1](bq_shape_from(wv))
    for j in range(hidden):
        grad_bq.set(
            j,
            Scalar[DType.float16](Float32(grad_q_flat.get(j))),
        )
    for j in range(kv_hidden):
        grad_bk.set(
            j,
            Scalar[DType.float16](Float32(grad_k_flat.get(j))),
        )
        grad_bv.set(
            j,
            Scalar[DType.float16](Float32(grad_v_flat.get(j))),
        )

    # projection gradients
    var mwq_saved = List[Tensor[DType.float16, 2]]()
    mwq_saved.append(x)
    mwq_saved.append(wq)
    var mwq_grads = matmul_weight_cpu_backward[DType.float16](
        grad_q_flat, mwq_saved
    )
    var grad_x = mwq_grads[0]
    var grad_wq = mwq_grads[1]
    var mwk_saved = List[Tensor[DType.float16, 2]]()
    mwk_saved.append(x)
    mwk_saved.append(wk)
    var mwk_grads = matmul_weight_cpu_backward[DType.float16](
        grad_k_flat, mwk_saved
    )
    grad_x = _add_t2_f16(grad_x, mwk_grads[0])
    var grad_wk = mwk_grads[1]
    var mwv_saved = List[Tensor[DType.float16, 2]]()
    mwv_saved.append(x)
    mwv_saved.append(wv)
    var mwv_grads = matmul_weight_cpu_backward[DType.float16](
        grad_v_flat, mwv_saved
    )
    grad_x = _add_t2_f16(grad_x, mwv_grads[0])
    var grad_wv = mwv_grads[1]

    results.append(to_any[DType.float16, 2](grad_x))
    results.append(to_any[DType.float16, 2](grad_wq))
    results.append(to_any[DType.float16, 2](grad_wk))
    results.append(to_any[DType.float16, 2](grad_wv))
    results.append(to_any[DType.float16, 2](grad_wo))
    results.append(to_any[DType.float16, 1](grad_bq))
    results.append(to_any[DType.float16, 1](grad_bk))
    results.append(to_any[DType.float16, 1](grad_bv))
    results.append(no_grad_any())  # k cache
    results.append(no_grad_any())  # v cache
    results.append(no_grad_any())  # pos
    return results^


def _add_t2_f16(
    a: Tensor[DType.float16, 2], b: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    var out = tensor_zeros[DType.float16, 2](a.shape())
    for i in range(a.numel()):
        out.set(
            i, Scalar[DType.float16](Float32(a.get(i)) + Float32(b.get(i)))
        )
    return out


def bq_shape_from(w: Tensor[DType.float16, 2]) -> StaticTuple[Int, 1]:
    return StaticTuple[Int, 1](w.shape()[0])


def mha_fws_gpu(
    inputs: List[AnyTensor],
) -> Tuple[List[AnyTensor], List[AnyTensor]]:
    return mha_fws_cpu(inputs)


def mha_bwd_gpu(
    grad_outputs: List[AnyTensor], saved: List[AnyTensor]
) -> List[AnyTensor]:
    return mha_bwd_cpu(grad_outputs, saved)
