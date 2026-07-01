import torch
from torch import Tensor
import torch.nn.functional as F

# 导入 Rust 引擎
from ._rust import PyTensor


# ============================================================
# 工具函数：torch.Tensor <-> PyTensor 互转
# ============================================================
def torch_to_pytensor(t: torch.Tensor) -> PyTensor:
    """将 torch.Tensor 转成我们的 PyTensor（只支持 CPU）"""
    if t.device.type != "cpu":
        raise RuntimeError("InferTrain only supports CPU tensors")
    # 处理 requires_grad 和 dtype
    data = t.detach().numpy().flatten().tolist()
    shape = list(t.shape)
    dtype = str(t.dtype).replace("torch.", "")
    return PyTensor(data, shape, dtype=dtype)


def pytensor_to_torch(t: PyTensor) -> torch.Tensor:
    """将 PyTensor 转成 torch.Tensor"""
    data = t.data()
    shape = t.shape()
    dtype_str = t.dtype()

    # 映射 dtype
    dtype_map = {
        "f32": torch.float32,
        "f64": torch.float64,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "i8": torch.int8,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    return torch.tensor(data, dtype=dtype).reshape(shape)


# ============================================================
# 无感侵入：覆盖 torch.Tensor 的算子
# ============================================================

# 1. 保存原始方法
_original_add = Tensor.__add__


# 2. 替换为我们的实现
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


# 3. 覆盖 torch.matmul
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


# 4. 覆盖 torch.nn.functional.relu
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


# 5. 覆盖 torch.nn.functional.softmax
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


# ============================================================
# 打印欢迎信息
# ============================================================
print("🧠 InferTrain Engine activated!")
print(f"   CPU tensors will use Rust engine for: add, matmul, relu, softmax")
print(f"   GPU tensors fallback to PyTorch native")
