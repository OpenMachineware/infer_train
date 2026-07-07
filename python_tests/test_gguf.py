import infer_train_torch as it

# ============================================================
# 1. 导入 GGUF 模型
# ============================================================
dag = it.import_gguf("model.gguf")
print(f"Model loaded: {dag.num_ops()} ops, {dag.num_constants()} weights")

# ============================================================
# 2. 边推边训 (GGUF → 训练 → 重新量化)
# ============================================================
# 加载为可训练模型
model_file = it.PyModelFile.new("model", "gguf", dag)
trainer = it.Trainer(model_file)

# 训练
for epoch in range(10):
    for batch in dataset:
        loss = trainer.train_step(batch.input, batch.target)
        print(f"Loss: {loss}")

# ============================================================
# 3. 导出为 GGUF (重新量化)
# ============================================================
it.export_gguf("model_updated.gguf", dag, "Q8_0")

# ============================================================
# 4. 导出为 ITM (原生格式，不需要重新量化)
# ============================================================
model_file.export("model.itm")

# ============================================================
# 5. 下次加载 ITM (最快)
# ============================================================
model_file = it.PyModelFile.load("model.itm")
executor = it.Executor(model_file.get_graph())
output = executor.execute(inputs)
