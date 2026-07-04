import tensorflow as tf
from .infer_train_tf import PyTensor


def tf_to_pytensor(t: tf.Tensor) -> PyTensor:
    """将 tf.Tensor 转成 PyTensor（只支持 CPU）"""
    if t.device is not None and "GPU" in str(t.device):
        raise RuntimeError("InferTrain only supports CPU tensors")
    data = t.numpy().flatten().tolist()
    shape = list(t.shape)
    dtype_map = {
        "float32": "f32",
        "float64": "f64",
        "float16": "f16",
        "bfloat16": "bf16",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
    }
    dtype = dtype_map.get(str(t.dtype), "f32")
    return PyTensor(data, shape, dtype=dtype)


def pytensor_to_tf(t: PyTensor) -> tf.Tensor:
    """将 PyTensor 转成 tf.Tensor"""
    data = t.data()
    shape = t.shape()
    dtype_str = t.dtype()
    dtype_map = {
        "f32": tf.float32,
        "f64": tf.float64,
        "f16": tf.float16,
        "bf16": tf.bfloat16,
        "i8": tf.int8,
    }
    dtype = dtype_map.get(dtype_str, tf.float32)
    return tf.constant(data, dtype=dtype, shape=shape)


__all__ = [
    'PyTensor',
    'tf_to_pytensor',
    'pytensor_to_tf',
]

print("🧠 InferTrain Engine activated for TensorFlow!")
print("   Use PyTensor class for tensor operations")
print("   Convert: tf_to_pytensor() / pytensor_to_tf()")
print("   GPU tensors fallback to TensorFlow native")
