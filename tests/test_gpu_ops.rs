// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GPU-native forward pass with mock weights

#![allow(clippy::approx_constant)]

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod tests {
    use infer_train::compute::backend::Backend;
    use infer_train::compute::graph::ComputeGraph;
    use infer_train::compute::metal_backend::MetalBackend;
    use infer_train::compute::op::{OpNode, OpType};
    use infer_train::model::tensor::DataType;

    /// Test GPU-native Add operation
    #[test]
    fn test_add_gpu() {
        let backend =
            MetalBackend::new().expect("Failed to create MetalBackend");

        let n = 1024;

        // Create two input tensors
        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();

        let mut a = backend.alloc_tensor("a", vec![n], DataType::F32).unwrap();
        backend.copy_to_tensor(&mut a, bytemuck::cast_slice(&a_data)).unwrap();

        let mut b = backend.alloc_tensor("b", vec![n], DataType::F32).unwrap();
        backend.copy_to_tensor(&mut b, bytemuck::cast_slice(&b_data)).unwrap();

        // Build compute graph
        let mut graph = ComputeGraph::new();
        graph.add_tensor(a);
        graph.add_tensor(b);
        graph.add_input("a");
        graph.add_input("b");
        graph.add_op(OpNode::new(
            OpType::Add,
            vec!["a".into(), "b".into()],
            "c".to_string(),
        ));

        // Execute
        backend.execute(&mut graph).expect("Failed to execute");

        // Get output
        let c = graph.get_tensor("c").expect("Failed to get output");
        let mut c_data = vec![0.0f32; n];
        backend
            .copy_from_tensor(c, bytemuck::cast_slice_mut(&mut c_data))
            .unwrap();

        // Verify
        for i in 0..n {
            let expected = a_data[i] + b_data[i];
            let diff = (c_data[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Add mismatch at {}: expected {}, got {}",
                i,
                expected,
                c_data[i]
            );
        }
        println!("Add GPU: {} elements verified", n);
    }

    /// Test GPU-native Mul operation
    #[test]
    fn test_mul_gpu() {
        let backend =
            MetalBackend::new().expect("Failed to create MetalBackend");

        let n = 1024;

        // Create two input tensors
        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();

        let mut a = backend.alloc_tensor("a", vec![n], DataType::F32).unwrap();
        backend.copy_to_tensor(&mut a, bytemuck::cast_slice(&a_data)).unwrap();

        let mut b = backend.alloc_tensor("b", vec![n], DataType::F32).unwrap();
        backend.copy_to_tensor(&mut b, bytemuck::cast_slice(&b_data)).unwrap();

        // Build compute graph
        let mut graph = ComputeGraph::new();
        graph.add_tensor(a);
        graph.add_tensor(b);
        graph.add_input("a");
        graph.add_input("b");
        graph.add_op(OpNode::new(
            OpType::Mul,
            vec!["a".into(), "b".into()],
            "c".to_string(),
        ));

        // Execute
        backend.execute(&mut graph).expect("Failed to execute");

        // Get output
        let c = graph.get_tensor("c").expect("Failed to get output");
        let mut c_data = vec![0.0f32; n];
        backend
            .copy_from_tensor(c, bytemuck::cast_slice_mut(&mut c_data))
            .unwrap();

        // Verify
        for i in 0..n {
            let expected = a_data[i] * b_data[i];
            let diff = (c_data[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Mul mismatch at {}: expected {}, got {}",
                i,
                expected,
                c_data[i]
            );
        }
        println!("Mul GPU: {} elements verified", n);
    }

    /// Test chained operations: a + b * c
    #[test]
    fn test_chained_ops() {
        let backend =
            MetalBackend::new().expect("Failed to create MetalBackend");

        let n = 1024;

        let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
        let c_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();

        let mut a = backend.alloc_tensor("a", vec![n], DataType::F32).unwrap();
        backend.copy_to_tensor(&mut a, bytemuck::cast_slice(&a_data)).unwrap();

        let mut b = backend.alloc_tensor("b", vec![n], DataType::F32).unwrap();
        backend.copy_to_tensor(&mut b, bytemuck::cast_slice(&b_data)).unwrap();

        let mut c = backend.alloc_tensor("c", vec![n], DataType::F32).unwrap();
        backend.copy_to_tensor(&mut c, bytemuck::cast_slice(&c_data)).unwrap();

        // Build graph: d = b * c; e = a + d
        let mut graph = ComputeGraph::new();
        graph.add_tensor(a);
        graph.add_tensor(b);
        graph.add_tensor(c);
        graph.add_input("a");
        graph.add_input("b");
        graph.add_input("c");

        graph.add_op(OpNode::new(
            OpType::Mul,
            vec!["b".into(), "c".into()],
            "d".to_string(),
        ));
        graph.add_op(OpNode::new(
            OpType::Add,
            vec!["a".into(), "d".into()],
            "e".to_string(),
        ));

        // Execute
        backend.execute(&mut graph).expect("Failed to execute");

        // Get output
        let e = graph.get_tensor("e").expect("Failed to get output");
        let mut e_data = vec![0.0f32; n];
        backend
            .copy_from_tensor(e, bytemuck::cast_slice_mut(&mut e_data))
            .unwrap();

        // Verify: e = a + b * c
        for i in 0..n {
            let expected = a_data[i] + b_data[i] * c_data[i];
            let diff = (e_data[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "Chained mismatch at {}: expected {}, got {}",
                i,
                expected,
                e_data[i]
            );
        }
        println!("Chained ops: {} elements verified", n);
    }
}
