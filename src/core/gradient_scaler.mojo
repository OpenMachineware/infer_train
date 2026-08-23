# core/gradient_scaler.mojo
#
# M6 Phase 6: mixed-precision (AMP) gradient scaling - the engine's
# counterpart to PyTorch's torch.amp.GradScaler.
#
# FP16/BF16 backprop can underflow: tiny gradients round to zero before the
# fp32 master update.  The scaler multiplies the loss by `scale` before the
# backward (the seed gradient carries the scale), so gradients land in a
# representable range; before the optimizer step they are divided back out
# (`unscale`).  When an Inf/NaN gradient is found the step is skipped and
# the scale is backed off; after a healthy interval it grows again.

from .ops.base.op_interface import AnyTensor
from .utils import unimplemented
from std.math import isnan, isinf


struct GradScaler(Movable):
    var scale: Float32
    var growth_interval: Int
    var backoff_factor: Float32
    var growth_factor: Float32
    var steps_since_growth: Int

    def __init__(
        out self,
        init_scale: Float32 = Float32(65536.0),
        growth_interval: Int = 2000,
        backoff_factor: Float32 = Float32(0.5),
        growth_factor: Float32 = Float32(2.0),
    ):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.backoff_factor = backoff_factor
        self.growth_factor = growth_factor
        self.steps_since_growth = 0

    def current_scale(self) -> Float32:
        return self.scale

    def found_inf(self, grads: List[AnyTensor]) -> Bool:
        """True when any gradient contains Inf/NaN (step must be skipped)."""
        for g in grads:
            if g.numel == 0:
                continue
            if g.dtype == DType.float32:
                var data = g.data.unsafe_bitcast[Scalar[DType.float32]]()
                for i in range(g.numel):
                    var v = Float32(data.unsafe_load[width=1](offset=i))
                    if isnan(v) or isinf(v):
                        return True
            elif g.dtype == DType.float16:
                var data = g.data.unsafe_bitcast[Scalar[DType.float16]]()
                for i in range(g.numel):
                    var v = Float32(data.unsafe_load[width=1](offset=i))
                    if isnan(v) or isinf(v):
                        return True
            else:
                unimplemented("GradScaler.found_inf: unsupported dtype")
        return False

    def scale_grads(mut self, grads: List[AnyTensor]):
        """Multiply every gradient by the current scale (post-backward)."""
        var s = Float32(self.scale)
        for g in grads:
            if g.numel == 0:
                continue
            if g.dtype == DType.float32:
                var data = g.data.unsafe_bitcast[Scalar[DType.float32]]()
                for i in range(g.numel):
                    var v = Float32(data.unsafe_load[width=1](offset=i))
                    data.unsafe_store(i, Scalar[DType.float32](v * s))
            elif g.dtype == DType.float16:
                var data = g.data.unsafe_bitcast[Scalar[DType.float16]]()
                for i in range(g.numel):
                    var v = Float32(data.unsafe_load[width=1](offset=i))
                    data.unsafe_store(i, Scalar[DType.float16](v * s))
            else:
                unimplemented("GradScaler.scale_grads: unsupported dtype")

    def unscale_grads(mut self, grads: List[AnyTensor]):
        """Divide every gradient by the current scale (pre-step)."""
        var s = Float32(self.scale)
        for g in grads:
            if g.numel == 0:
                continue
            if g.dtype == DType.float32:
                var data = g.data.unsafe_bitcast[Scalar[DType.float32]]()
                for i in range(g.numel):
                    var v = Float32(data.unsafe_load[width=1](offset=i))
                    data.unsafe_store(i, Scalar[DType.float32](v / s))
            elif g.dtype == DType.float16:
                var data = g.data.unsafe_bitcast[Scalar[DType.float16]]()
                for i in range(g.numel):
                    var v = Float32(data.unsafe_load[width=1](offset=i))
                    data.unsafe_store(i, Scalar[DType.float16](v / s))
            else:
                unimplemented("GradScaler.unscale_grads: unsupported dtype")

    def update(mut self, found_inf: Bool):
        """Adjust the scale: back off on Inf/NaN, grow after a healthy
        interval."""
        if found_inf:
            self.scale = self.scale * self.backoff_factor
            self.steps_since_growth = 0
        else:
            self.steps_since_growth += 1
            if self.steps_since_growth >= self.growth_interval:
                self.scale = self.scale * self.growth_factor
                self.steps_since_growth = 0
