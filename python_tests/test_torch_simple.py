# python_tests/test_torch_simple.py

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

print("1. Creating tracer...")
tracer = it.HookTracer()

print("2. Starting tracing...")
tracer.start_tracing()

print("3. Running model...")
output = model(x)
print(f"   Output shape: {output.shape}")

print("4. Stopping tracing...")
tracer.stop_tracing()

print("5. Getting DAG...")
dag = tracer.get_dag()
print(f"   DAG: {dag}")

print("6. Exporting...")
tracer.trace_and_export(
    model=model,
    inputs=[x],
    path="test_model.itm",
    model_name="simple_model"
)
print("   Exported to test_model.itm")

print("\n✅ All done!")
