// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test compute graph for inference
// Note: MetalBackend is not Send+Sync due to Rc<Buffer> and RefCell
// Tests run in single-threaded mode

#![allow(clippy::approx_constant)]

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod tests {
    use infer_train::compute::backend::Backend;
    use infer_train::compute::graph::ComputeGraph;
    use infer_train::compute::metal_backend::MetalBackend;
    use infer_train::compute::op::{OpNode, OpType};
    use infer_train::compute::tensor::Tensor;
    use infer_train::model::tensor::DataType;

    /// Test basic GPU-native RMS Norm through ComputeGraph
    #[test]
    fn test_rms_norm_graph() {
        let backend =
            MetalBackend::new().expect("Failed to create MetalBackend");

        let hidden_size = 4096;
        let n_tokens = 1;

        // Create input tensor
        let input_data: Vec<f32> = (0..hidden_size * n_tokens)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let mut input = backend
            .alloc_tensor("input", vec![hidden_size, n_tokens], DataType::F32)
            .expect("Failed to alloc input");
        backend
            .copy_to_tensor(&mut input, bytemuck::cast_slice(&input_data))
            .expect("Failed to copy input");

        // Create weight tensor
        let weight_data: Vec<f32> = (0..hidden_size).map(|_| 1.0).collect();
        let mut weight = backend
            .alloc_tensor("weight", vec![hidden_size], DataType::F32)
            .expect("Failed to alloc weight");
        backend
            .copy_to_tensor(&mut weight, bytemuck::cast_slice(&weight_data))
            .expect("Failed to copy weight");

        // Build compute graph
        let mut graph = ComputeGraph::new();
        graph.add_tensor(input.clone());
        graph.add_tensor(weight.clone());
        graph.add_input("input");
        graph.add_input("weight");

        let op = OpNode::new(
            OpType::RMSNorm,
            vec!["input".into(), "weight".into()],
            "output".to_string(),
        );
        graph.add_op(op);

        // Execute
        backend.execute(&mut graph).expect("Failed to execute");

        // Get output
        let output = graph.get_tensor("output").expect("Failed to get output");
        let mut output_data = vec![0.0f32; hidden_size * n_tokens];
        backend
            .copy_from_tensor(
                output,
                bytemuck::cast_slice_mut(&mut output_data),
            )
            .expect("Failed to copy output");

        // Verify output is reasonable (not NaN, not all zeros)
        let has_nan = output_data.iter().any(|x| x.is_nan());
        let all_zero = output_data.iter().all(|x| *x == 0.0);
        assert!(!has_nan, "Output contains NaN");
        assert!(!all_zero, "Output is all zeros");

        // Check magnitude is similar to input (RMSNorm should not change magnitude much)
        let input_rms: f32 = (input_data.iter().map(|x| x * x).sum::<f32>()
            / hidden_size as f32)
            .sqrt();
        let output_rms: f32 = (output_data.iter().map(|x| x * x).sum::<f32>()
            / hidden_size as f32)
            .sqrt();
        println!("Input RMS: {}, Output RMS: {}", input_rms, output_rms);
        println!("Input[0..4]: {:?}", &input_data[0..4]);
        println!("Output[0..4]: {:?}", &output_data[0..4]);
        assert!(
            (input_rms - output_rms).abs() < 0.5,
            "RMS changed too much: input_rms={}, output_rms={}",
            input_rms,
            output_rms
        );
    }

    /// Test GPU-native MatVec through ComputeGraph
    #[test]
    fn test_matvec_graph() {
        let backend =
            MetalBackend::new().expect("Failed to create MetalBackend");

        let m = 1024; // Output size
        let k = 4096; // Input size

        // Create input vector (F32)
        let input_data: Vec<f32> =
            (0..k).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut input = backend
            .alloc_tensor("input", vec![k], DataType::F32)
            .expect("Failed to alloc input");
        backend
            .copy_to_tensor(&mut input, bytemuck::cast_slice(&input_data))
            .expect("Failed to copy input");

        // Create Q4_0 weight matrix (simulated)
        // Q4_0 block: 32 weights per block, each block = [d: f16, qs: [u8; 16]]
        use infer_train::quant::types::BlockQ4_0;
        let n_blocks = (k / 32) * m;
        let mut weight_blocks: Vec<BlockQ4_0> = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            weight_blocks.push(BlockQ4_0 {
                d: half::f16::to_bits(half::f16::from_f32(1.0)), // FP16 scale as u16
                qs: [8u8; 16], // Midpoint for Q4_0
            });
        }

        let mut weight = backend
            .alloc_tensor("weight", vec![k, m], DataType::Q4_0)
            .expect("Failed to alloc weight");
        backend
            .copy_to_tensor(&mut weight, bytemuck::cast_slice(&weight_blocks))
            .expect("Failed to copy weight");

        // Build compute graph
        let mut graph = ComputeGraph::new();
        graph.add_tensor(weight.clone());
        graph.add_tensor(input.clone());
        graph.add_input("weight");
        graph.add_input("input");

        let op = OpNode::new(
            OpType::MatVec,
            vec!["weight".into(), "input".into()],
            "output".to_string(),
        );
        graph.add_op(op);

        // Execute
        backend.execute(&mut graph).expect("Failed to execute");

        // Get output
        let output = graph.get_tensor("output").expect("Failed to get output");
        let mut output_data = vec![0.0f32; m];
        backend
            .copy_from_tensor(
                output,
                bytemuck::cast_slice_mut(&mut output_data),
            )
            .expect("Failed to copy output");

        // Verify output is reasonable
        let has_nan = output_data.iter().any(|x| x.is_nan());
        assert!(!has_nan, "Output contains NaN");
    }
}
