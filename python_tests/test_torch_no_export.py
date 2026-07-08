# python_tests/test_torch.py

import torch
import torch.nn as nn
import infer_train_torch as it

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

model = SimpleModel()
x = torch.randn(1, 10)

# ✅ 直接使用 tracer
tracer = it.HookTracer()
tracer.start_tracing()
output = model(x)
tracer.stop_tracing()

# ✅ 导出
tracer.trace_and_export(
    model,      # ← 传模型，不是 None
    [x],        # ← 传输入，不是 []
    "test_model.itm",
    "simple_model"
)

print("Exported!")
