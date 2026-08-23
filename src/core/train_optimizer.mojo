# core/train_optimizer.mojo
#
# M6: training optimizers - AdamW and momentum SGD over type-erased
# parameters (the M6 spec's `src/core/optimizer.mojo`; the name avoids a
# clash with the M5 IR-optimizer package at src/core/optimizer/).
#
# Parameters live in caller-owned tensors; `ParamEntry` holds an erased
# view of each parameter plus a *persistent fp32 gradient buffer* owned by
# the optimizer (zeroed by `zero_grad`, accumulated by `accumulate_grads`).
# Per-parameter state (AdamW m/v, SGD momentum buffer) is allocated lazily
# on the first `step` and its opaque handle is stored in the parameter
# tensor's `opt_state` field (see tensor.mojo).
#
# Parameter groups: each `ParamGroup` carries its own lr/weight_decay so
# different layers can train at different rates.

from .tensor import Tensor, tensor_zeros
from .ops.base.op_interface import AnyTensor, from_any, to_any
from .utils import unimplemented
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.math import sqrt, pow
from std.utils.static_tuple import StaticTuple


# -- shared state -------------------------------------------------------------


struct ParamEntry(Copyable, Movable, ImplicitlyCopyable):
    var param: AnyTensor
    var grad: AnyTensor  # fp32 accumulation buffer (optimizer-owned)
    var state: Pointer[UInt8, MutUntrackedOrigin]
    var frozen: Bool  # M7 LoRA mode: frozen entries are skipped by step()

    def __init__(
        out self,
        param: AnyTensor,
        grad: AnyTensor,
        state: Pointer[UInt8, MutUntrackedOrigin],
    ):
        self.param = param
        self.grad = grad
        self.state = state
        self.frozen = False


struct ParamGroup(Copyable, Movable):
    var entries: List[ParamEntry]
    var lr: Float32
    var weight_decay: Float32

    def __init__(
        out self, lr: Float32 = Float32(1e-3), weight_decay: Float32 = Float32(0)
    ):
        self.entries = List[ParamEntry]()
        self.lr = lr
        self.weight_decay = weight_decay


struct AdamState(Movable):
    var m: AnyTensor  # fp32 first moment
    var v: AnyTensor  # fp32 second moment
    var m_owned: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var v_owned: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var t: Int

    def __init__(
        out self,
        m: AnyTensor,
        v: AnyTensor,
        m_owned: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
        v_owned: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
    ):
        self.m = m
        self.v = v
        self.m_owned = m_owned
        self.v_owned = v_owned
        self.t = 0


struct SGDState(Movable):
    var buf: AnyTensor  # fp32 momentum buffer
    var buf_owned: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var t: Int

    def __init__(
        out self,
        buf: AnyTensor,
        buf_owned: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
    ):
        self.buf = buf
        self.buf_owned = buf_owned
        self.t = 0


# -- dtype/rank helpers -------------------------------------------------------


def _numel_of(any: AnyTensor) -> Int:
    return any.numel


def _alloc_f32_like(any: AnyTensor) -> AnyTensor:
    """Allocate a zeroed fp32 buffer with `any`'s shape (as an AnyTensor)."""
    var numel = any.numel
    if numel < 1:
        numel = 1
    var buf = unsafe_alloc[Scalar[DType.float32]](numel)
    for i in range(numel):
        buf.unsafe_offset(i).unsafe_store(val=Scalar[DType.float32](0))
    var shape = any.shape
    var result = AnyTensor(
        DType.float32,
        any.rank,
        shape,
        any.numel,
        any.device,
        buf.unsafe_bitcast[UInt8](),
    )
    return result


def _zero_any(mut any: AnyTensor):
    if any.dtype == DType.float32:
        for i in range(any.numel):
            any.data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
                i, Scalar[DType.float32](0)
            )
    elif any.dtype == DType.float16:
        for i in range(any.numel):
            any.data.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
                i, Scalar[DType.float16](0)
            )
    else:
        unimplemented("_zero_any: unsupported dtype")


def _accumulate_f32_from(mut dest: AnyTensor, delta: AnyTensor):
    """dest (fp32) += delta (f32 or f16) elementwise, in place."""
    var numel = dest.numel
    if delta.dtype == DType.float32:
        var d = dest.data.unsafe_bitcast[Scalar[DType.float32]]()
        var g = delta.data.unsafe_bitcast[Scalar[DType.float32]]()
        for i in range(numel):
            d.unsafe_store(
                i,
                Scalar[DType.float32](
                    Float32(d.unsafe_load[width=1](offset=i))
                    + Float32(g.unsafe_load[width=1](offset=i))
                ),
            )
    elif delta.dtype == DType.float16:
        var d = dest.data.unsafe_bitcast[Scalar[DType.float32]]()
        var g = delta.data.unsafe_bitcast[Scalar[DType.float16]]()
        for i in range(numel):
            d.unsafe_store(
                i,
                Scalar[DType.float32](
                    Float32(d.unsafe_load[width=1](offset=i))
                    + Float32(g.unsafe_load[width=1](offset=i))
                ),
            )
    else:
        unimplemented("_accumulate_f32_from: unsupported dtype")


def _copy_f32_from(mut dest: AnyTensor, src: AnyTensor):
    """dest (fp32) = src (f32 or f16) elementwise, in place."""
    var numel = dest.numel
    if src.dtype == DType.float32:
        var d = dest.data.unsafe_bitcast[Scalar[DType.float32]]()
        var g = src.data.unsafe_bitcast[Scalar[DType.float32]]()
        for i in range(numel):
            d.unsafe_store(
                i,
                Scalar[DType.float32](
                    Float32(g.unsafe_load[width=1](offset=i))
                ),
            )
    elif src.dtype == DType.float16:
        var d = dest.data.unsafe_bitcast[Scalar[DType.float32]]()
        var g = src.data.unsafe_bitcast[Scalar[DType.float16]]()
        for i in range(numel):
            d.unsafe_store(
                i,
                Scalar[DType.float32](
                    Float32(g.unsafe_load[width=1](offset=i))
                ),
            )
    else:
        unimplemented("_copy_f32_from: unsupported dtype")


def _set_param_scaled(mut param: AnyTensor, f32_val: Float32, offset: Int):
    if param.dtype == DType.float32:
        param.data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
            offset, Scalar[DType.float32](f32_val)
        )
    elif param.dtype == DType.float16:
        param.data.unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
            offset, Scalar[DType.float16](f32_val)
        )
    else:
        unimplemented("_set_param_scaled: unsupported dtype")


# -- AdamW --------------------------------------------------------------------


struct AdamW(Movable):
    var groups: List[ParamGroup]
    var betas: StaticTuple[Float32, 2]
    var eps: Float32

    def __init__(
        out self,
        lr: Float32 = Float32(1e-3),
        betas: StaticTuple[Float32, 2] = StaticTuple[Float32, 2](
            Float32(0.9), Float32(0.999)
        ),
        eps: Float32 = Float32(1e-8),
        weight_decay: Float32 = Float32(0.01),
    ):
        self.groups = List[ParamGroup]()
        self.betas = betas
        self.eps = eps
        var g0 = ParamGroup(lr, weight_decay)
        self.groups.append(g0^)

    def add_param[dtype: DType, rank: Int](
        mut self, mut param: Tensor[dtype, rank], group_index: Int = 0
    ):
        """Register `param` (a caller-owned tensor) with the optimizer.

        Allocates the persistent fp32 gradient buffer and the AdamW state
        (m/v); the state handle is stored in `param.opt_state`.
        """
        if group_index < 0 or group_index >= len(self.groups):
            unimplemented("AdamW.add_param: bad group index")
        var any = to_any[dtype, rank](param)
        var grad = _alloc_f32_like(any)
        var m = _alloc_f32_like(any)
        var v = _alloc_f32_like(any)
        var state = unsafe_alloc[AdamState](1)
        state[0] = AdamState(
            m,
            v,
            m.data.unsafe_bitcast[Scalar[DType.float32]](),
            v.data.unsafe_bitcast[Scalar[DType.float32]](),
        )
        param.set_opt_state(state.unsafe_bitcast[UInt8]())
        var entry = ParamEntry(any, grad, state.unsafe_bitcast[UInt8]())
        self.groups[group_index].entries.append(entry)

    def set_frozen(mut self, index: Int, frozen: Bool, group_index: Int = 0):
        """M7 LoRA mode: freeze (skip in step()) or unfreeze one parameter."""
        if (
            group_index < 0
            or group_index >= len(self.groups)
            or index < 0
            or index >= len(self.groups[group_index].entries)
        ):
            unimplemented("AdamW.set_frozen: bad index")
        self.groups[group_index].entries[index].frozen = frozen

    def num_params(self) -> Int:
        return len(self.groups[0].entries)

    def set_group_hyperparams(
        mut self, group_index: Int, lr: Float32, weight_decay: Float32
    ):
        if group_index < 0 or group_index >= len(self.groups):
            unimplemented("AdamW.set_group_hyperparams: bad group index")
        self.groups[group_index].lr = lr
        self.groups[group_index].weight_decay = weight_decay

    def zero_grad(mut self):
        for gi in range(len(self.groups)):
            for ei in range(len(self.groups[gi].entries)):
                var entry = self.groups[gi].entries[ei]
                _zero_any(entry.grad)

    def accumulate_grads(mut self, grads: List[AnyTensor]):
        """Sum one backward pass's parameter gradients into the persistent
        buffers (fp32 accumulation - the AMP master-gradient path)."""
        var index = 0
        for gi in range(len(self.groups)):
            for ei in range(len(self.groups[gi].entries)):
                if index >= len(grads):
                    return
                var entry = self.groups[gi].entries[ei]
                var g = grads[index]
                index += 1
                if g.numel != entry.grad.numel:
                    unimplemented("AdamW.accumulate_grads: shape mismatch")
                _accumulate_f32_from(entry.grad, g)

    def step(mut self):
        """One AdamW update over every parameter group.

        m = b1*m + (1-b1)*g        (g = the accumulated gradient buffer)
        v = b2*v + (1-b2)*g^2
        m_hat = m/(1-b1^t); v_hat = v/(1-b2^t)
        p = p - lr * (m_hat/(sqrt(v_hat)+eps) + weight_decay*p)
        """
        var b1 = Float32(self.betas[0])
        var b2 = Float32(self.betas[1])
        var one_m_b1 = Float32(1.0) - b1
        var one_m_b2 = Float32(1.0) - b2
        for group in self.groups:
            var lr = Float32(group.lr)
            var wd = Float32(group.weight_decay)
            for entry in group.entries:
                if entry.frozen:
                    continue  # M7 LoRA: frozen params keep their values
                var state = entry.state.unsafe_bitcast[AdamState]()
                state[0].t += 1
                var t = state[0].t
                var numel = entry.param.numel
                var m = state[0].m.data.unsafe_bitcast[
                    Scalar[DType.float32]
                ]()
                var v = state[0].v.data.unsafe_bitcast[
                    Scalar[DType.float32]
                ]()
                var g = entry.grad.data.unsafe_bitcast[
                    Scalar[DType.float32]
                ]()
                var p = entry.param
                var bc1 = Float32(1.0) - pow(b1, Float32(t))
                var bc2 = Float32(1.0) - pow(b2, Float32(t))
                if bc1 < Float32(1e-12):
                    bc1 = Float32(1e-12)
                if bc2 < Float32(1e-12):
                    bc2 = Float32(1e-12)
                for i in range(numel):
                    var gv = Float32(g.unsafe_load[width=1](offset=i))
                    var mv = b1 * Float32(m.unsafe_load[width=1](offset=i)) + one_m_b1 * gv
                    var vv = b2 * Float32(v.unsafe_load[width=1](offset=i)) + one_m_b2 * gv * gv
                    m.unsafe_store(i, Scalar[DType.float32](mv))
                    v.unsafe_store(i, Scalar[DType.float32](vv))
                    var mh = mv / bc1
                    var vh = vv / bc2
                    var pv = Float32(
                        p.data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=i)
                    )
                    var update = mh / (sqrt(vh) + Float32(self.eps)) + wd * pv
                    _set_param_scaled(p, pv - lr * update, i)


# -- SGD with momentum --------------------------------------------------------


struct SGD(Movable):
    var groups: List[ParamGroup]
    var momentum: Float32

    def __init__(
        out self,
        lr: Float32 = Float32(1e-2),
        momentum: Float32 = Float32(0.0),
        weight_decay: Float32 = Float32(0.0),
    ):
        self.groups = List[ParamGroup]()
        self.momentum = momentum
        var g0 = ParamGroup(lr, weight_decay)
        self.groups.append(g0^)

    def add_param[dtype: DType, rank: Int](
        mut self, mut param: Tensor[dtype, rank], group_index: Int = 0
    ):
        if group_index < 0 or group_index >= len(self.groups):
            unimplemented("SGD.add_param: bad group index")
        var any = to_any[dtype, rank](param)
        var grad = _alloc_f32_like(any)
        var buf = _alloc_f32_like(any)
        var state = unsafe_alloc[SGDState](1)
        state[0] = SGDState(
            buf, buf.data.unsafe_bitcast[Scalar[DType.float32]]()
        )
        param.set_opt_state(state.unsafe_bitcast[UInt8]())
        var entry = ParamEntry(any, grad, state.unsafe_bitcast[UInt8]())
        self.groups[group_index].entries.append(entry)

    def zero_grad(mut self):
        for gi in range(len(self.groups)):
            for ei in range(len(self.groups[gi].entries)):
                var entry = self.groups[gi].entries[ei]
                _zero_any(entry.grad)

    def accumulate_grads(mut self, grads: List[AnyTensor]):
        var index = 0
        for gi in range(len(self.groups)):
            for ei in range(len(self.groups[gi].entries)):
                if index >= len(grads):
                    return
                var entry = self.groups[gi].entries[ei]
                var g = grads[index]
                index += 1
                if g.numel != entry.grad.numel:
                    unimplemented("SGD.accumulate_grads: shape mismatch")
                _accumulate_f32_from(entry.grad, g)

    def step(mut self):
        """SGD update: p = p - lr * (g + weight_decay * p), with optional
        momentum (buf = mu*buf + g)."""
        var mu = Float32(self.momentum)
        for group in self.groups:
            var lr = Float32(group.lr)
            var wd = Float32(group.weight_decay)
            for entry in group.entries:
                var state = entry.state.unsafe_bitcast[SGDState]()
                state[0].t += 1
                var numel = entry.param.numel
                var buf = state[0].buf.data.unsafe_bitcast[
                    Scalar[DType.float32]
                ]()
                var g = entry.grad.data.unsafe_bitcast[
                    Scalar[DType.float32]
                ]()
                var p = entry.param
                for i in range(numel):
                    var gv = Float32(g.unsafe_load[width=1](offset=i))
                    var bv = mu * Float32(buf.unsafe_load[width=1](offset=i)) + gv
                    buf.unsafe_store(i, Scalar[DType.float32](bv))
                    var pv = Float32(
                        p.data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=i)
                    )
                    var update = bv + wd * pv
                    _set_param_scaled(p, pv - lr * update, i)


# -- stateless update kernels (for the C-API bindings) ------------------------
#


def adamw_step_raw[dtype: DType](
    mut param: AnyTensor,
    mut grad: AnyTensor,
    mut m: AnyTensor,
    mut v: AnyTensor,
    mut t: Int,
    lr: Float32,
    b1: Float32,
    b2: Float32,
    eps: Float32,
    weight_decay: Float32,
):
    """One AdamW update over raw buffers; `t` is the (mutated) step count."""
    t += 1
    var numel = param.numel
    var one_m_b1 = Float32(1.0) - b1
    var one_m_b2 = Float32(1.0) - b2
    var bc1 = Float32(1.0) - pow(b1, Float32(t))
    var bc2 = Float32(1.0) - pow(b2, Float32(t))
    if bc1 < Float32(1e-12):
        bc1 = Float32(1e-12)
    if bc2 < Float32(1e-12):
        bc2 = Float32(1e-12)
    if dtype == DType.float32:
        var p = param.data.unsafe_bitcast[Scalar[DType.float32]]()
        var g = grad.data.unsafe_bitcast[Scalar[DType.float32]]()
        var mp = m.data.unsafe_bitcast[Scalar[DType.float32]]()
        var vp = v.data.unsafe_bitcast[Scalar[DType.float32]]()
        for i in range(numel):
            var gv = Float32(g.unsafe_load[width=1](offset=i))
            var mv = b1 * Float32(mp.unsafe_load[width=1](offset=i)) + one_m_b1 * gv
            var vv = b2 * Float32(vp.unsafe_load[width=1](offset=i)) + one_m_b2 * gv * gv
            mp.unsafe_store(i, Scalar[DType.float32](mv))
            vp.unsafe_store(i, Scalar[DType.float32](vv))
            var pv = Float32(p.unsafe_load[width=1](offset=i))
            p.unsafe_store(
                i,
                Scalar[DType.float32](
                    pv - lr * (mv / bc1 / (sqrt(vv / bc2) + eps) + weight_decay * pv)
                ),
            )
    elif dtype == DType.float16:
        var p = param.data.unsafe_bitcast[Scalar[DType.float16]]()
        var g = grad.data.unsafe_bitcast[Scalar[DType.float16]]()
        var mp = m.data.unsafe_bitcast[Scalar[DType.float32]]()
        var vp = v.data.unsafe_bitcast[Scalar[DType.float32]]()
        for i in range(numel):
            var gv = Float32(g.unsafe_load[width=1](offset=i))
            var mv = b1 * Float32(mp.unsafe_load[width=1](offset=i)) + one_m_b1 * gv
            var vv = b2 * Float32(vp.unsafe_load[width=1](offset=i)) + one_m_b2 * gv * gv
            mp.unsafe_store(i, Scalar[DType.float32](mv))
            vp.unsafe_store(i, Scalar[DType.float32](vv))
            var pv = Float32(p.unsafe_load[width=1](offset=i))
            p.unsafe_store(
                i,
                Scalar[DType.float16](
                    pv - lr * (mv / bc1 / (sqrt(vv / bc2) + eps) + weight_decay * pv)
                ),
            )
    else:
        unimplemented("adamw_step_raw: unsupported dtype")
