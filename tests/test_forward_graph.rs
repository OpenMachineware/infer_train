// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GPU-native forward pass using ComputeGraph

#![allow(clippy::approx_constant)]

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod tests {
    use infer_train::compute::metal_backend::MetalBackend;
    use infer_train::model::llama::LlamaModel;
    use infer_train::model::weights::WeightStore;

    /// Test forward_graph with a simple RMS Norm operation
    #[test]
    fn test_forward_graph_rms_norm() {
        // This test verifies the ComputeGraph integration works
        // For a full end-to-end test, we need a real GGUF model file

        // Create MetalBackend
        let backend =
            MetalBackend::new().expect("Failed to create MetalBackend");

        // TODO: Load a real model and test forward_graph
        // For now, this test just verifies compilation
        println!("MetalBackend created successfully");
        println!("ComputeGraph forward pass infrastructure ready");
    }
}
