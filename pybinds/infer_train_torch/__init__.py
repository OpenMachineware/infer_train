import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import math

# 导入 Rust 引擎
from .infer_train_torch import PyTensor

# 引擎激活标志
_ENGINE_ACTIVATED = False

# ============================================================
# 工具函数：torch.Tensor <-> PyTensor 互转
# ============================================================
def torch_to_pytensor(t: torch.Tensor) -> PyTensor:
    """将 torch.Tensor 转成 PyTensor（只支持 CPU）"""
    if t.device.type != "cpu":
        raise RuntimeError("InferTrain only supports CPU tensors")
    data = t.detach().numpy().flatten().tolist()
    shape = list(t.shape)
    dtype = str(t.dtype).replace("torch.", "")
    return PyTensor(data, shape, dtype=dtype)


def pytensor_to_torch(t: PyTensor) -> torch.Tensor:
    """将 PyTensor 转成 torch.Tensor"""
    data = t.data()
    shape = t.shape()
    dtype_str = t.dtype()
    dtype_map = {
        "f32": torch.float32,
        "f64": torch.float64,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "i8": torch.int8,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    return torch.tensor(data, dtype=dtype).reshape(shape)


def _fallback_to_torch(*args, **kwargs):
    """用于 fallback 的占位函数"""
    raise RuntimeError("Fallback should not be called directly")


# ============================================================
# 数学算子
# ============================================================

_original_add = Tensor.__add__


def _it_add(self, other):
    # 只在 CPU 上、且两个都是 Tensor 时才用我们的引擎
    if (self.device.type == "cpu" and
            isinstance(other, Tensor) and
            other.device.type == "cpu"):
        try:
            a = torch_to_pytensor(self)
            b = torch_to_pytensor(other)
            c = a.add(b)
            return pytensor_to_torch(c)
        except Exception as e:
            # 出错时 fallback 到 PyTorch 原生
            print(f"InferTrain fallback: {e}")
            return _original_add(self, other)
    return _original_add(self, other)


Tensor.__add__ = _it_add

_original_sub = Tensor.__sub__


def _it_sub(self, other):
    if (
        self.device.type == "cpu"
        and isinstance(other, Tensor)
        and other.device.type == "cpu"
    ):
        try:
            a = torch_to_pytensor(self)
            b = torch_to_pytensor(other)
            c = a.sub(b)
            return pytensor_to_torch(c)
        except Exception:
            return _original_sub(self, other)
    return _original_sub(self, other)


Tensor.__sub__ = _it_sub

_original_mul = Tensor.__mul__


def _it_mul(self, other):
    if (
        self.device.type == "cpu"
        and isinstance(other, Tensor)
        and other.device.type == "cpu"
    ):
        try:
            a = torch_to_pytensor(self)
            b = torch_to_pytensor(other)
            c = a.mul(b)
            return pytensor_to_torch(c)
        except Exception:
            return _original_mul(self, other)
    return _original_mul(self, other)


Tensor.__mul__ = _it_mul

_original_div = Tensor.__truediv__


def _it_div(self, other):
    if (
        self.device.type == "cpu"
        and isinstance(other, Tensor)
        and other.device.type == "cpu"
    ):
        try:
            a = torch_to_pytensor(self)
            b = torch_to_pytensor(other)
            c = a.div(b)
            return pytensor_to_torch(c)
        except Exception:
            return _original_div(self, other)
    return _original_div(self, other)


Tensor.__truediv__ = _it_div

# ---------- pow ----------
_original_pow = torch.pow

def _it_pow(input, exponent):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            if isinstance(exponent, torch.Tensor):
                e = torch_to_pytensor(exponent)
                out = t.pow(e)
            else:
                # scalar exponent
                # 用标量版本：创建临时张量
                e_tensor = torch.tensor([exponent], dtype=input.dtype)
                e = torch_to_pytensor(e_tensor)
                out = t.pow(e)
            return pytensor_to_torch(out)
        except Exception:
            return _original_pow(input, exponent)
    return _original_pow(input, exponent)

torch.pow = _it_pow

# ---------- exp ----------
_original_exp = torch.exp

def _it_exp(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.exp()
            return pytensor_to_torch(out)
        except Exception:
            return _original_exp(input)
    return _original_exp(input)

torch.exp = _it_exp

# ---------- sqrt ----------
_original_sqrt = torch.sqrt

def _it_sqrt(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.sqrt()
            return pytensor_to_torch(out)
        except Exception:
            return _original_sqrt(input)
    return _original_sqrt(input)

torch.sqrt = _it_sqrt

# ---------- log ----------
_original_log = torch.log

def _it_log(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.log()
            return pytensor_to_torch(out)
        except Exception:
            return _original_log(input)
    return _original_log(input)

torch.log = _it_log

# ---------- log2 ----------
_original_log2 = torch.log2

def _it_log2(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.log2()
            return pytensor_to_torch(out)
        except Exception:
            return _original_log2(input)
    return _original_log2(input)

torch.log2 = _it_log2

# ---------- log10 ----------
_original_log10 = torch.log10

def _it_log10(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.log10()
            return pytensor_to_torch(out)
        except Exception:
            return _original_log10(input)
    return _original_log10(input)

torch.log10 = _it_log10

# ---------- abs ----------
_original_abs = torch.abs

def _it_abs(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.abs()
            return pytensor_to_torch(out)
        except Exception:
            return _original_abs(input)
    return _original_abs(input)

torch.abs = _it_abs

# ---------- neg ----------
_original_neg = torch.neg

def _it_neg(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.neg()
            return pytensor_to_torch(out)
        except Exception:
            return _original_neg(input)
    return _original_neg(input)

torch.neg = _it_neg

# ---------- clamp ----------
_original_clamp = torch.clamp

def _it_clamp(input, min_val=None, max_val=None):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            if min_val is not None and max_val is not None:
                out = t.clamp(min_val, max_val)
            elif min_val is not None:
                out = t.clamp(min_val, 1e9)
            elif max_val is not None:
                out = t.clamp(-1e9, max_val)
            else:
                return input
            return pytensor_to_torch(out)
        except Exception:
            return _original_clamp(input, min_val, max_val)
    return _original_clamp(input, min_val, max_val)

torch.clamp = _it_clamp

# ---------- floor ----------
_original_floor = torch.floor

def _it_floor(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.floor()
            return pytensor_to_torch(out)
        except Exception:
            return _original_floor(input)
    return _original_floor(input)

torch.floor = _it_floor

# ---------- ceil ----------
_original_ceil = torch.ceil

def _it_ceil(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.ceil()
            return pytensor_to_torch(out)
        except Exception:
            return _original_ceil(input)
    return _original_ceil(input)

torch.ceil = _it_ceil

# ---------- round ----------
_original_round = torch.round

def _it_round(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.round()
            return pytensor_to_torch(out)
        except Exception:
            return _original_round(input)
    return _original_round(input)

torch.round = _it_round

# ---------- compare ----------
_original_eq = torch.eq

def _it_eq(input, other):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            o = torch_to_pytensor(other)
            out = t.eq(o)
            return torch.tensor(out, dtype=torch.bool).reshape(t.shape())
        except Exception:
            return _original_eq(input, other)
    return _original_eq(input, other)

torch.eq = _it_eq

_original_ne = torch.ne

def _it_ne(input, other):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            o = torch_to_pytensor(other)
            out = t.ne(o)
            return torch.tensor(out, dtype=torch.bool).reshape(t.shape())
        except Exception:
            return _original_ne(input, other)
    return _original_ne(input, other)

torch.ne = _it_ne

_original_gt = torch.gt

def _it_gt(input, other):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            o = torch_to_pytensor(other)
            out = t.gt(o)
            return torch.tensor(out, dtype=torch.bool).reshape(t.shape())
        except Exception:
            return _original_gt(input, other)
    return _original_gt(input, other)

torch.gt = _it_gt

_original_lt = torch.lt

def _it_lt(input, other):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            o = torch_to_pytensor(other)
            out = t.lt(o)
            return torch.tensor(out, dtype=torch.bool).reshape(t.shape())
        except Exception:
            return _original_lt(input, other)
    return _original_lt(input, other)

torch.lt = _it_lt

_original_ge = torch.ge

def _it_ge(input, other):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            o = torch_to_pytensor(other)
            out = t.ge(o)
            return torch.tensor(out, dtype=torch.bool).reshape(t.shape())
        except Exception:
            return _original_ge(input, other)
    return _original_ge(input, other)

torch.ge = _it_ge

_original_le = torch.le

def _it_le(input, other):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            o = torch_to_pytensor(other)
            out = t.le(o)
            return torch.tensor(out, dtype=torch.bool).reshape(t.shape())
        except Exception:
            return _original_le(input, other)
    return _original_le(input, other)

torch.le = _it_le

# ============================================================
# 矩阵算子
# ============================================================
_original_matmul = torch.matmul


def _it_matmul(a, b):
    if (isinstance(a, Tensor) and isinstance(b, Tensor) and
            a.device.type == "cpu" and b.device.type == "cpu"):
        try:
            at = torch_to_pytensor(a)
            bt = torch_to_pytensor(b)
            ct = at.matmul(bt)
            return pytensor_to_torch(ct)
        except Exception:
            return _original_matmul(a, b)
    return _original_matmul(a, b)


torch.matmul = _it_matmul

# ---------- batch_matmul ----------
_original_bmm = torch.bmm

def _it_bmm(input, mat2):
    if input.device.type == "cpu" and mat2.device.type == "cpu":
        try:
            t1 = torch_to_pytensor(input)
            t2 = torch_to_pytensor(mat2)
            out = t1.batch_matmul(t2)
            return pytensor_to_torch(out)
        except Exception:
            return _original_bmm(input, mat2)
    return _original_bmm(input, mat2)

torch.bmm = _it_bmm


# ============================================================
# embedding
# ============================================================
_original_embedding = F.embedding

def _it_embedding(input, weight, padding_idx=-1, max_norm=None, norm_type=2.0,
                  scale_grad_by_freq=False, sparse=False):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            # input 是 1D 或 2D 索引
            indices = input.detach().numpy().flatten().tolist()
            indices_int = [int(x) for x in indices]
            w = torch_to_pytensor(weight)
            out = w.embedding(indices_int, padding_idx)
            return pytensor_to_torch(out)
        except Exception:
            return _original_embedding(input, weight, padding_idx, max_norm, norm_type,
                                        scale_grad_by_freq, sparse)
    return _original_embedding(input, weight, padding_idx, max_norm, norm_type,
                                scale_grad_by_freq, sparse)

F.embedding = _it_embedding


# ============================================================
# slice
# ============================================================
# torch 的 slice 是通过索引实现的，不需要覆盖
# 但我们可以提供一个函数

def infer_train_slice(input, dim, start, end, step=1):
    """使用 InferTrain 引擎的 slice 操作"""
    if input.device.type != "cpu":
        return input
    t = torch_to_pytensor(input)
    out = t.slice(dim, start, end, step)
    return pytensor_to_torch(out)


# ---------- where ----------
_original_where = torch.where

def _it_where(condition, x, y):
    if condition.device.type == "cpu":
        try:
            cond = condition.detach().numpy().flatten().tolist()
            cond_shape = list(condition.shape)
            t = torch_to_pytensor(x)
            f = torch_to_pytensor(y)
            out = t.where_(cond, cond_shape, f)
            return pytensor_to_torch(out)
        except Exception:
            return _original_where(condition, x, y)
    return _original_where(condition, x, y)

torch.where = _it_where

# ============================================================
# NN 算子
# ============================================================

# ---------- conv2d ----------
_original_conv2d = F.conv2d

def _it_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_pytensor(input)
            w = torch_to_pytensor(weight)
            b = torch_to_pytensor(bias) if bias is not None else None
            out = inp.conv2d(w, b, stride, padding, dilation, groups)
            return pytensor_to_torch(out)
        except Exception:
            return _original_conv2d(input, weight, bias, stride, padding, dilation, groups)
    return _original_conv2d(input, weight, bias, stride, padding, dilation, groups)

F.conv2d = _it_conv2d

# ---------- conv1d ----------
_original_conv1d = F.conv1d

def _it_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_pytensor(input)
            w = torch_to_pytensor(weight)
            b = torch_to_pytensor(bias) if bias is not None else None
            out = inp.conv1d(w, b, stride, padding, dilation, groups)
            return pytensor_to_torch(out)
        except Exception:
            return _original_conv1d(input, weight, bias, stride, padding, dilation, groups)
    return _original_conv1d(input, weight, bias, stride, padding, dilation, groups)

F.conv1d = _it_conv1d

# ---------- conv3d ----------
_original_conv3d = F.conv3d

def _it_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_pytensor(input)
            w = torch_to_pytensor(weight)
            b = torch_to_pytensor(bias) if bias is not None else None
            out = inp.conv3d(w, b, stride, padding, dilation, groups)
            return pytensor_to_torch(out)
        except Exception:
            return _original_conv3d(input, weight, bias, stride, padding, dilation, groups)
    return _original_conv3d(input, weight, bias, stride, padding, dilation, groups)

F.conv3d = _it_conv3d

# ---------- maxpool2d ----------
_original_maxpool2d = F.max_pool2d

def _it_maxpool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if input.device.type == "cpu" and not return_indices and not ceil_mode:
        try:
            inp = torch_to_pytensor(input)
            if stride is None:
                stride = kernel_size
            out = inp.maxpool2d(kernel_size, stride, padding)
            return pytensor_to_torch(out)
        except Exception:
            return _original_maxpool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    return _original_maxpool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

F.max_pool2d = _it_maxpool2d

# ---------- maxpool1d ----------
_original_maxpool1d = F.max_pool1d

def _it_maxpool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if input.device.type == "cpu" and not return_indices and not ceil_mode:
        try:
            inp = torch_to_pytensor(input)
            if stride is None:
                stride = kernel_size
            out = inp.maxpool1d(kernel_size, stride, padding)
            return pytensor_to_torch(out)
        except Exception:
            return _original_maxpool1d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    return _original_maxpool1d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

F.max_pool1d = _it_maxpool1d

# ---------- maxpool3d ----------
_original_maxpool3d = F.max_pool3d

def _it_maxpool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if input.device.type == "cpu" and not return_indices and not ceil_mode:
        try:
            inp = torch_to_pytensor(input)
            if stride is None:
                stride = kernel_size
            out = inp.maxpool3d(kernel_size, stride, padding)
            return pytensor_to_torch(out)
        except Exception:
            return _original_maxpool3d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    return _original_maxpool3d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

F.max_pool3d = _it_maxpool3d

# ---------- avgpool2d ----------
_original_avgpool2d = F.avg_pool2d

def _it_avgpool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    if input.device.type == "cpu" and not ceil_mode:
        try:
            inp = torch_to_pytensor(input)
            if stride is None:
                stride = kernel_size
            out = inp.avgpool2d(kernel_size, stride, padding)
            return pytensor_to_torch(out)
        except Exception:
            return _original_avgpool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    return _original_avgpool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)

F.avg_pool2d = _it_avgpool2d

# ---------- avgpool1d ----------
_original_avgpool1d = F.avg_pool1d

def _it_avgpool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    if input.device.type == "cpu" and not ceil_mode:
        try:
            inp = torch_to_pytensor(input)
            if stride is None:
                stride = kernel_size
            out = inp.avgpool1d(kernel_size, stride, padding)
            return pytensor_to_torch(out)
        except Exception:
            return _original_avgpool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    return _original_avgpool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)

F.avg_pool1d = _it_avgpool1d

# ---------- avgpool3d ----------
_original_avgpool3d = F.avg_pool3d

def _it_avgpool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    if input.device.type == "cpu" and not ceil_mode:
        try:
            inp = torch_to_pytensor(input)
            if stride is None:
                stride = kernel_size
            out = inp.avgpool3d(kernel_size, stride, padding)
            return pytensor_to_torch(out)
        except Exception:
            return _original_avgpool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    return _original_avgpool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)

F.avg_pool3d = _it_avgpool3d

# ---------- batchnorm2d ----------
_original_batchnorm2d = F.batch_norm

def _it_batchnorm2d(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    if input.device.type == "cpu" and not training:
        try:
            inp = torch_to_pytensor(input)
            w = torch_to_pytensor(weight) if weight is not None else None
            b = torch_to_pytensor(bias) if bias is not None else None
            rm = torch_to_pytensor(running_mean) if running_mean is not None else None
            rv = torch_to_pytensor(running_var) if running_var is not None else None
            # 注意：batchnorm2d 需要 weight 和 bias
            if w is not None and b is not None and rm is not None and rv is not None:
                out = inp.batchnorm2d(w, b, rm, rv, eps)
                return pytensor_to_torch(out)
            else:
                return _original_batchnorm2d(input, running_mean, running_var, weight, bias, training, momentum, eps)
        except Exception:
            return _original_batchnorm2d(input, running_mean, running_var, weight, bias, training, momentum, eps)
    return _original_batchnorm2d(input, running_mean, running_var, weight, bias, training, momentum, eps)

F.batch_norm = _it_batchnorm2d

# ---------- layernorm ----------
_original_layernorm = F.layer_norm

def _it_layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if input.device.type == "cpu":
        try:
            inp = torch_to_pytensor(input)
            if weight is not None and bias is not None:
                w = torch_to_pytensor(weight)
                b = torch_to_pytensor(bias)
                out = inp.layernorm(w, b, eps)
                return pytensor_to_torch(out)
            else:
                return _original_layernorm(input, normalized_shape, weight, bias, eps)
        except Exception:
            return _original_layernorm(input, normalized_shape, weight, bias, eps)
    return _original_layernorm(input, normalized_shape, weight, bias, eps)

F.layer_norm = _it_layernorm

# ---------- linear ----------
_original_linear = F.linear

def _it_linear(input, weight, bias=None):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_pytensor(input)
            w = torch_to_pytensor(weight)
            b = torch_to_pytensor(bias) if bias is not None else None
            out = inp.linear(w, b)
            return pytensor_to_torch(out)
        except Exception:
            return _original_linear(input, weight, bias)
    return _original_linear(input, weight, bias)

F.linear = _it_linear

# ---------- dropout ----------
# Dropout 在推理模式下是恒等函数，直接用 PyTorch 的
# 不需要覆盖，因为推理时 dropout 不做任何事

# ============================================================
# 激活函数
# ============================================================
_original_sigmoid = F.sigmoid


def _it_sigmoid(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.sigmoid()
            return pytensor_to_torch(out)
        except Exception:
            return _original_sigmoid(input)
    return _original_sigmoid(input)


F.sigmoid = _it_sigmoid

_original_tanh = F.tanh


def _it_tanh(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.tanh()
            return pytensor_to_torch(out)
        except Exception:
            return _original_tanh(input)
    return _original_tanh(input)


F.tanh = _it_tanh

_original_relu = F.relu


def _it_relu(input, inplace=False):
    if input.device.type == "cpu" and not inplace:
        try:
            t = torch_to_pytensor(input)
            out = t.relu()
            return pytensor_to_torch(out)
        except Exception:
            return _original_relu(input, inplace)
    return _original_relu(input, inplace)


F.relu = _it_relu

_original_gelu = F.gelu


def _it_gelu(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.gelu()
            return pytensor_to_torch(out)
        except Exception:
            return _original_gelu(input)
    return _original_gelu(input)


F.gelu = _it_gelu

_original_silu = F.silu


def _it_silu(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.silu()
            return pytensor_to_torch(out)
        except Exception:
            return _original_silu(input)
    return _original_silu(input)


F.silu = _it_silu

_original_leaky_relu = F.leaky_relu


def _it_leaky_relu(input, negative_slope=0.01, inplace=False):
    if input.device.type == "cpu" and not inplace:
        try:
            t = torch_to_pytensor(input)
            out = t.leaky_relu(negative_slope)
            return pytensor_to_torch(out)
        except Exception:
            return _original_leaky_relu(input, negative_slope, inplace)
    return _original_leaky_relu(input, negative_slope, inplace)


F.leaky_relu = _it_leaky_relu

_original_elu = F.elu


def _it_elu(input, alpha=1.0, inplace=False):
    if input.device.type == "cpu" and not inplace:
        try:
            t = torch_to_pytensor(input)
            out = t.elu(alpha)
            return pytensor_to_torch(out)
        except Exception:
            return _original_elu(input, alpha, inplace)
    return _original_elu(input, alpha, inplace)


F.elu = _it_elu

_original_relu6 = F.relu6


def _it_relu6(input, inplace=False):
    if input.device.type == "cpu" and not inplace:
        try:
            t = torch_to_pytensor(input)
            out = t.relu6()
            return pytensor_to_torch(out)
        except Exception:
            return _original_relu6(input, inplace)
    return _original_relu6(input, inplace)


F.relu6 = _it_relu6

_original_softmax = F.softmax


def _it_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if input.device.type == "cpu" and dim is not None:
        try:
            t = torch_to_pytensor(input)
            out = t.softmax(dim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_softmax(input, dim, _stacklevel, dtype)
    return _original_softmax(input, dim, _stacklevel, dtype)


F.softmax = _it_softmax

_original_softplus = F.softplus


def _it_softplus(input, beta=1.0, threshold=20.0):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.softplus(beta, threshold)
            return pytensor_to_torch(out)
        except Exception:
            return _original_softplus(input, beta, threshold)
    return _original_softplus(input, beta, threshold)


F.softplus = _it_softplus

# ---------- log_softmax ----------
_original_log_softmax = F.log_softmax

def _it_log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if input.device.type == "cpu" and dim is not None:
        try:
            t = torch_to_pytensor(input)
            out = t.log_softmax(dim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_log_softmax(input, dim, _stacklevel, dtype)
    return _original_log_softmax(input, dim, _stacklevel, dtype)

F.log_softmax = _it_log_softmax

# ---------- hard_swish ----------
_original_hard_swish = F.hardswish

def _it_hard_swish(input, inplace=False):
    if input.device.type == "cpu" and not inplace:
        try:
            t = torch_to_pytensor(input)
            out = t.hard_swish()
            return pytensor_to_torch(out)
        except Exception:
            return _original_hard_swish(input, inplace)
    return _original_hard_swish(input, inplace)

F.hardswish = _it_hard_swish

# ---------- hard_sigmoid ----------
_original_hard_sigmoid = F.hardsigmoid

def _it_hard_sigmoid(input, inplace=False):
    if input.device.type == "cpu" and not inplace:
        try:
            t = torch_to_pytensor(input)
            out = t.hard_sigmoid()
            return pytensor_to_torch(out)
        except Exception:
            return _original_hard_sigmoid(input, inplace)
    return _original_hard_sigmoid(input, inplace)

F.hardsigmoid = _it_hard_sigmoid

# ---------- softshrink ----------
_original_softshrink = F.softshrink

def _it_softshrink(input, lambd=0.5):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.softshrink(lambd)
            return pytensor_to_torch(out)
        except Exception:
            return _original_softshrink(input, lambd)
    return _original_softshrink(input, lambd)

F.softshrink = _it_softshrink

# ---------- celu ----------
_original_celu = F.celu

def _it_celu(input, alpha=1.0, inplace=False):
    if input.device.type == "cpu" and not inplace:
        try:
            t = torch_to_pytensor(input)
            out = t.celu(alpha)
            return pytensor_to_torch(out)
        except Exception:
            return _original_celu(input, alpha, inplace)
    return _original_celu(input, alpha, inplace)

F.celu = _it_celu


# ============================================================
# 张量操作
# ============================================================

# ---------- transpose ----------
_original_transpose = Tensor.transpose

def _it_transpose(self, dim0, dim1):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            # PyTensor 的 transpose 是转置整个张量，不是交换两个维度
            # 所以需要用 permute
            # 简单起见：如果转置 2D 张量，用我们的 transpose
            if len(self.shape) == 2 and dim0 == 0 and dim1 == 1:
                out = t.transpose()
                return pytensor_to_torch(out)
            else:
                return _original_transpose(self, dim0, dim1)
        except Exception:
            return _original_transpose(self, dim0, dim1)
    return _original_transpose(self, dim0, dim1)

# 不能直接覆盖 Tensor.transpose，因为它是方法
# 改用 torch.transpose 函数

_original_torch_transpose = torch.transpose

def _it_torch_transpose(input, dim0, dim1):
    if input.device.type == "cpu" and len(input.shape) == 2 and dim0 == 0 and dim1 == 1:
        try:
            t = torch_to_pytensor(input)
            out = t.transpose()
            return pytensor_to_torch(out)
        except Exception:
            return _original_torch_transpose(input, dim0, dim1)
    return _original_torch_transpose(input, dim0, dim1)

torch.transpose = _it_torch_transpose

# ---------- slice ----------
# torch 的 slice 是通过索引实现的，不需要覆盖

# ---------- gather ----------
_original_gather = torch.gather

def _it_gather(input, dim, index, *, sparse_grad=False):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            idx = index.detach().numpy().flatten().tolist()
            idx_shape = list(index.shape)
            out = t.gather(idx, idx_shape, dim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_gather(input, dim, index, sparse_grad=sparse_grad)
    return _original_gather(input, dim, index, sparse_grad=sparse_grad)

torch.gather = _it_gather

# ---------- scatter ----------
_original_scatter = torch.scatter

def _it_scatter(input, dim, index, src):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            idx = index.detach().numpy().flatten().tolist()
            idx_shape = list(index.shape)
            s = torch_to_pytensor(src)
            out = t.scatter(idx, idx_shape, s, dim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_scatter(input, dim, index, src)
    return _original_scatter(input, dim, index, src)

torch.scatter = _it_scatter

# ---------- sort ----------
_original_sort = torch.sort

def _it_sort(input, dim=-1, descending=False, stable=False):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            ascending = not descending
            values, indices = t.sort(dim, ascending)
            return pytensor_to_torch(values), pytensor_to_torch(indices)
        except Exception:
            return _original_sort(input, dim, descending, stable)
    return _original_sort(input, dim, descending, stable)

torch.sort = _it_sort

# ---------- cumsum ----------
_original_cumsum = torch.cumsum

def _it_cumsum(input, dim, dtype=None):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.cumsum(dim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_cumsum(input, dim, dtype)
    return _original_cumsum(input, dim, dtype)

torch.cumsum = _it_cumsum

# ---------- cumprod ----------
_original_cumprod = torch.cumprod

def _it_cumprod(input, dim, dtype=None):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.cumprod(dim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_cumprod(input, dim, dtype)
    return _original_cumprod(input, dim, dtype)

torch.cumprod = _it_cumprod

# ============================================================
# 损失函数
# ============================================================

# ---------- cross_entropy_loss ----------
_original_cross_entropy = F.cross_entropy

def _it_cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                      reduce=None, reduction='mean', label_smoothing=0.0):
    if input.device.type == "cpu" and len(target.shape) == 1:
        try:
            inp = torch_to_pytensor(input)
            # target 是 1D 索引
            target_list = target.detach().numpy().flatten().tolist()
            target_int = [int(x) for x in target_list]
            out = inp.cross_entropy_loss(target_int, reduction == 'mean')
            return pytensor_to_torch(out)
        except Exception:
            return _original_cross_entropy(input, target, weight, size_average, ignore_index,
                                           reduce, reduction, label_smoothing)
    return _original_cross_entropy(input, target, weight, size_average, ignore_index,
                                    reduce, reduction, label_smoothing)

F.cross_entropy = _it_cross_entropy

# ---------- mse_loss ----------
_original_mse = F.mse_loss

def _it_mse(input, target, reduction='mean'):
    if input.device.type == "cpu" and target.device.type == "cpu":
        try:
            inp = torch_to_pytensor(input)
            tgt = torch_to_pytensor(target)
            out = inp.mse_loss(tgt, reduction == 'mean')
            return pytensor_to_torch(out)
        except Exception:
            return _original_mse(input, target, reduction)
    return _original_mse(input, target, reduction)

F.mse_loss = _it_mse

# ---------- l1_loss ----------
_original_l1 = F.l1_loss

def _it_l1(input, target, reduction='mean'):
    if input.device.type == "cpu" and target.device.type == "cpu":
        try:
            inp = torch_to_pytensor(input)
            tgt = torch_to_pytensor(target)
            out = inp.l1_loss(tgt, reduction == 'mean')
            return pytensor_to_torch(out)
        except Exception:
            return _original_l1(input, target, reduction)
    return _original_l1(input, target, reduction)

F.l1_loss = _it_l1

# ---------- bce_loss ----------
_original_bce = F.binary_cross_entropy

def _it_bce(input, target, weight=None, reduction='mean'):
    if input.device.type == "cpu" and target.device.type == "cpu" and weight is None:
        try:
            inp = torch_to_pytensor(input)
            tgt = torch_to_pytensor(target)
            out = inp.bce_loss(tgt, reduction == 'mean', 1e-7)
            return pytensor_to_torch(out)
        except Exception:
            return _original_bce(input, target, weight, reduction)
    return _original_bce(input, target, weight, reduction)

F.binary_cross_entropy = _it_bce

# ============================================================
# 规约算子
# ============================================================
_original_sum = Tensor.sum


def _it_sum(self, dim=None, keepdim=False):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            if dim is None:
                out = t.sum([], keepdim)
            else:
                if isinstance(dim, int):
                    dim = [dim]
                out = t.sum(dim, keepdim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_sum(self, dim, keepdim)
    return _original_sum(self, dim, keepdim)


Tensor.sum = _it_sum

_original_mean = Tensor.mean


def _it_mean(self, dim=None, keepdim=False):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            if dim is None:
                out = t.mean([], keepdim)
            else:
                if isinstance(dim, int):
                    dim = [dim]
                out = t.mean(dim, keepdim)
            return pytensor_to_torch(out)
        except Exception:
            return _original_mean(self, dim, keepdim)
    return _original_mean(self, dim, keepdim)


Tensor.mean = _it_mean

# ---------- max ----------
_original_max = Tensor.max

def _it_max(self, dim=None, keepdim=False):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            if dim is None:
                out = t.max_all()
                return pytensor_to_torch(out)
            else:
                # 沿指定维度取 max 比较复杂，fallback
                return _original_max(self, dim, keepdim)
        except Exception:
            return _original_max(self, dim, keepdim)
    return _original_max(self, dim, keepdim)

Tensor.max = _it_max

# ---------- min ----------
_original_min = Tensor.min

def _it_min(self, dim=None, keepdim=False):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            if dim is None:
                out = t.min_all()
                return pytensor_to_torch(out)
            else:
                return _original_min(self, dim, keepdim)
        except Exception:
            return _original_min(self, dim, keepdim)
    return _original_min(self, dim, keepdim)

Tensor.min = _it_min

# ---------- std ----------
_original_std = Tensor.std

def _it_std(self, unbiased=True):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            out = t.std(unbiased)
            return pytensor_to_torch(out)
        except Exception:
            return _original_std(self, unbiased)
    return _original_std(self, unbiased)

Tensor.std = _it_std

# ---------- var ----------
_original_var = Tensor.var

def _it_var(self, unbiased=True):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            out = t.var(unbiased)
            return pytensor_to_torch(out)
        except Exception:
            return _original_var(self, unbiased)
    return _original_var(self, unbiased)

Tensor.var = _it_var

# ---------- argmax ----------
_original_argmax = torch.argmax

def _it_argmax(input, dim=None, keepdim=False):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.argmax()
            return pytensor_to_torch(out)
        except Exception:
            return _original_argmax(input, dim, keepdim)
    return _original_argmax(input, dim, keepdim)

torch.argmax = _it_argmax

# ---------- argmin ----------
_original_argmin = torch.argmin

def _it_argmin(input, dim=None, keepdim=False):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            out = t.argmin()
            return pytensor_to_torch(out)
        except Exception:
            return _original_argmin(input, dim, keepdim)
    return _original_argmin(input, dim, keepdim)

torch.argmin = _it_argmin

# ---------- topk ----------
_original_topk = torch.topk

def _it_topk(input, k, dim=None, largest=True, sorted=True):
    if input.device.type == "cpu":
        try:
            t = torch_to_pytensor(input)
            values, indices = t.topk(k, dim if dim is not None else -1, largest)
            return pytensor_to_torch(values), pytensor_to_torch(indices)
        except Exception:
            return _original_topk(input, k, dim, largest, sorted)
    return _original_topk(input, k, dim, largest, sorted)

torch.topk = _it_topk

# ---------- prod ----------
_original_prod = Tensor.prod

def _it_prod(self, dim=None, keepdim=False):
    if self.device.type == "cpu":
        try:
            t = torch_to_pytensor(self)
            out = t.prod_all()
            return pytensor_to_torch(out)
        except Exception:
            return _original_prod(self, dim, keepdim)
    return _original_prod(self, dim, keepdim)

Tensor.prod = _it_prod

# ============================================================
# 量化算子
# ============================================================

def torch_to_quantized_pytensor(t: torch.Tensor) -> PyTensor:
    """将 torch.Tensor 转成量化 PyTensor"""
    if t.device.type != "cpu":
        raise RuntimeError("InferTrain only supports CPU tensors")
    # 检查是否是量化张量
    if hasattr(t, '_scale') and hasattr(t, '_zero_point'):
        scale = t._scale
        zero_point = t._zero_point
    else:
        # 默认量化参数
        scale = 1.0 / 255.0
        zero_point = 0.0
    data = t.detach().numpy().flatten().tolist()
    shape = list(t.shape)
    # 将数据转为 int8
    data_i8 = [int(x) for x in data]
    return PyTensor(data_i8, shape, dtype="i8", scale=scale, zero_point=zero_point)


# ---------- quantized_add ----------
def quantized_add(a, b):
    if a.device.type == "cpu" and b.device.type == "cpu":
        try:
            a_q = torch_to_quantized_pytensor(a)
            b_q = torch_to_quantized_pytensor(b)
            out = a_q.quantized_add(b_q)
            return pytensor_to_torch(out)
        except Exception:
            return a + b
    return a + b


# ---------- quantized_matmul ----------
def quantized_matmul(a, b):
    if a.device.type == "cpu" and b.device.type == "cpu":
        try:
            a_q = torch_to_quantized_pytensor(a)
            b_q = torch_to_quantized_pytensor(b)
            out = a_q.quantized_matmul(b_q)
            return pytensor_to_torch(out)
        except Exception:
            return torch.matmul(a, b)
    return torch.matmul(a, b)


# ---------- quantized_relu ----------
def quantized_relu(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_quantized_pytensor(input)
            out = t.quantized_relu()
            return pytensor_to_torch(out)
        except Exception:
            return torch.relu(input)
    return torch.relu(input)


# ---------- quantized_sigmoid ----------
def quantized_sigmoid(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_quantized_pytensor(input)
            out = t.quantized_sigmoid()
            return pytensor_to_torch(out)
        except Exception:
            return torch.sigmoid(input)
    return torch.sigmoid(input)


# ---------- quantized_conv2d ----------
def quantized_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_quantized_pytensor(input)
            w = torch_to_quantized_pytensor(weight)
            b = torch_to_quantized_pytensor(bias) if bias is not None else None
            out = inp.quantized_conv2d(w, b, stride, padding, dilation, groups)
            return pytensor_to_torch(out)
        except Exception:
            return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)


# ---------- quantized_linear ----------
def quantized_linear(input, weight, bias=None):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_quantized_pytensor(input)
            w = torch_to_quantized_pytensor(weight)
            b = torch_to_quantized_pytensor(bias) if bias is not None else None
            out = inp.quantized_linear(w, b)
            return pytensor_to_torch(out)
        except Exception:
            return torch.nn.functional.linear(input, weight, bias)
    return torch.nn.functional.linear(input, weight, bias)

# ============================================================
# cat 算子
# ============================================================
_original_cat = torch.cat

def _it_cat(tensors, dim=0):
    if all(t.device.type == "cpu" for t in tensors):
        try:
            # 将所有张量转成 PyTensor
            pytensors = [torch_to_pytensor(t) for t in tensors]
            # 调用 PyTensor.cat（静态方法）
            result = PyTensor.cat(pytensors, dim)
            return pytensor_to_torch(result)
        except Exception:
            return _original_cat(tensors, dim)
    return _original_cat(tensors, dim)

torch.cat = _it_cat

# ============================================================
# 导出量化函数
# ============================================================

def torch_to_quantized_pytensor(t: torch.Tensor) -> PyTensor:
    """将 torch.Tensor 转成量化 PyTensor"""
    if t.device.type != "cpu":
        raise RuntimeError("InferTrain only supports CPU tensors")
    scale = 1.0 / 255.0
    zero_point = 0.0
    data = t.detach().numpy().flatten().tolist()
    shape = list(t.shape)
    data_i8 = [int(x) for x in data]
    return PyTensor(data_i8, shape, dtype="i8", scale=scale, zero_point=zero_point)


def quantized_add(a, b):
    if a.device.type == "cpu" and b.device.type == "cpu":
        try:
            a_q = torch_to_quantized_pytensor(a)
            b_q = torch_to_quantized_pytensor(b)
            out = a_q.quantized_add(b_q)
            return pytensor_to_torch(out)
        except Exception:
            return a + b
    return a + b


def quantized_matmul(a, b):
    if a.device.type == "cpu" and b.device.type == "cpu":
        try:
            a_q = torch_to_quantized_pytensor(a)
            b_q = torch_to_quantized_pytensor(b)
            out = a_q.quantized_matmul(b_q)
            return pytensor_to_torch(out)
        except Exception:
            return torch.matmul(a, b)
    return torch.matmul(a, b)


def quantized_relu(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_quantized_pytensor(input)
            out = t.quantized_relu()
            return pytensor_to_torch(out)
        except Exception:
            return torch.relu(input)
    return torch.relu(input)


def quantized_sigmoid(input):
    if input.device.type == "cpu":
        try:
            t = torch_to_quantized_pytensor(input)
            out = t.quantized_sigmoid()
            return pytensor_to_torch(out)
        except Exception:
            return torch.sigmoid(input)
    return torch.sigmoid(input)


def quantized_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_quantized_pytensor(input)
            w = torch_to_quantized_pytensor(weight)
            b = torch_to_quantized_pytensor(bias) if bias is not None else None
            out = inp.quantized_conv2d(w, b, stride, padding, dilation, groups)
            return pytensor_to_torch(out)
        except Exception:
            return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)


def quantized_linear(input, weight, bias=None):
    if input.device.type == "cpu" and weight.device.type == "cpu":
        try:
            inp = torch_to_quantized_pytensor(input)
            w = torch_to_quantized_pytensor(weight)
            b = torch_to_quantized_pytensor(bias) if bias is not None else None
            out = inp.quantized_linear(w, b)
            return pytensor_to_torch(out)
        except Exception:
            return torch.nn.functional.linear(input, weight, bias)
    return torch.nn.functional.linear(input, weight, bias)


__all__ = [
    'quantized_add',
    'quantized_matmul',
    'quantized_relu',
    'quantized_sigmoid',
    'quantized_conv2d',
    'quantized_linear'
]

# ============================================================
# 打印欢迎信息
# ============================================================
print("🧠 InferTrain Engine activated for PyTorch!!")
print("   Supported ops: add, sub, mul, div, matmul, bmm, pow, exp, sqrt, log, log2, log10")
print("   abs, neg, clamp, floor, ceil, round")
print("   conv1d/2d/3d, maxpool1d/2d/3d, avgpool1d/2d/3d, batchnorm2d, layernorm, linear, embedding")
print("   relu, sigmoid, tanh, gelu, silu, leaky_relu, elu, relu6, softmax, log_softmax")
print("   sum, mean, max, min, std, var, prod")
print("   cross_entropy_loss, mse_loss, l1_loss, bce_loss")
print("   Quantized ops: quantized_add, quantized_matmul, quantized_relu, quantized_sigmoid")
print("   quantized_conv2d, quantized_linear")
print("   GPU tensors fallback to PyTorch native")
