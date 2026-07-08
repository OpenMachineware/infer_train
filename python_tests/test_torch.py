# python_tests/test_torch.py

import infer_train_torch as it
import torch
import torch.nn as nn

print("=" * 50)
print("InferTrain Torch Plugin Test")
print("=" * 50)

# 1. 测试导入
print("\n1. Testing import...")
print(f"   Version: {it.__version__}")
print(f"   PyTensor: {it.PyTensor}")
print(f"   PyModelFile: {it.PyModelFile}")

# 2. 测试 PyTensor
print("\n2. Testing PyTensor...")
a = it.PyTensor([1.0, 2.0, 3.0], [3])
b = it.PyTensor([4.0, 5.0, 6.0], [3])
c = a.add(b)
print(f"   a + b = {c.data()}")

# 3. 测试模型追踪
print("\n3. Testing model tracing...")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

model = SimpleModel()
x = torch.randn(1, 10)

# ✅ 正确：调用 trace_model
print("   Calling trace_model...")
dag = it.trace_model(model, [x])
print(f"   DAG: {dag}")

# 4. 测试导出
print("\n4. Testing export...")
it.export_model("test_model.itm", "simple_model", trainable=False)
print("   Exported to test_model.itm")

# 5. 测试加载
print("\n5. Testing load...")
model_file = it.PyModelFile.load("test_model.itm")
print(f"   Loaded: {model_file}")

print("\n" + "=" * 50)
print("All tests passed! 🎉")
