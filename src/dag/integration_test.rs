// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Integration test for DAG-based forward pass

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod tests {
    use crate::dag::{DagBuilder, DagExecutor};
    use crate::inference::{Device, InferenceContext};
    use crate::model::GgufFile;

    #[test]
    fn test_dag_layer_forward() {
        let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

        let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
        let mut ctx =
            InferenceContext::from_gguf(model_path, Device::Gpu).expect("Failed to load model");

        if let (Some(ref gpu_ctx), Some(ref model), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.model.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let layer = &model.layers[0];

            // Build DAG for layer 0
            let mut builder = DagBuilder::new();
            let dag = builder.build_layer_dag(model, layer, 0, 0);

            println!("\n=== DAG Construction ===");
            println!("Nodes: {}", dag.nodes.len());
            println!("Entry points: {:?}", dag.entry_points);
            println!("Exit points: {:?}", dag.exit_points);

            let exec_order = dag.execution_order();
            println!("Execution order: {} nodes", exec_order.len());

            // TODO: Execute DAG
            // let executor = DagExecutor::new(buffer_mgr);
            // let output = executor.execute_gpu_batched(&dag, model, kv_cache, gpu_ctx);
        }
    }
}
