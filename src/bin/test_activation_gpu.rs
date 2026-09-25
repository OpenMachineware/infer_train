// Test GPU SiLU/GELU correctness
use metal::{Device, CommandQueue, ComputePipelineState, MTLSize, Buffer};

struct ActivationMetalContext {
    device: Device,
    queue: CommandQueue,
    silu_pipeline: ComputePipelineState,
    gelu_pipeline: ComputePipelineState,
    gelu_quick_pipeline: ComputePipelineState,
}

impl ActivationMetalContext {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);

        let source = include_str!("../../shaders/activation.metal");
        let library = device.new_library_with_source(source, &compile_options)
            .map_err(|e| format!("Failed to compile library: {}", e))?;

        let silu = library.get_function("kernel_silu_f32", None)
            .map_err(|e| format!("Failed to get silu kernel: {}", e))?;
        let silu_pipeline = device.new_compute_pipeline_state_with_function(&silu)
            .map_err(|e| format!("Failed to create silu pipeline: {}", e))?;

        let gelu = library.get_function("kernel_gelu_f32", None)
            .map_err(|e| format!("Failed to get gelu kernel: {}", e))?;
        let gelu_pipeline = device.new_compute_pipeline_state_with_function(&gelu)
            .map_err(|e| format!("Failed to create gelu pipeline: {}", e))?;

        let gelu_quick = library.get_function("kernel_gelu_quick_f32", None)
            .map_err(|e| format!("Failed to get gelu_quick kernel: {}", e))?;
        let gelu_quick_pipeline = device.new_compute_pipeline_state_with_function(&gelu_quick)
            .map_err(|e| format!("Failed to create gelu_quick pipeline: {}", e))?;

        Ok(Self {
            device, queue,
            silu_pipeline, gelu_pipeline, gelu_quick_pipeline,
        })
    }

    fn run(&self, src: &[f32], pipeline: &ComputePipelineState) -> Vec<f32> {
        let total_size = src.len();
        let input_buffer = self.device.new_buffer(total_size as u64 * 4, metal::MTLResourceOptions::StorageModeShared);
        let output_buffer = self.device.new_buffer(total_size as u64 * 4, metal::MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), input_buffer.contents() as *mut f32, total_size);
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let threadgroup_size = MTLSize { width: 256, height: 1, depth: 1 };
        let grid_size = MTLSize { width: total_size as u64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut result = vec![0.0f32; total_size];
        unsafe {
            std::ptr::copy_nonoverlapping(output_buffer.contents() as *const f32, result.as_mut_ptr(), total_size);
        }
        result
    }
}

fn main() {
    println!("=== GPU Activation Functions Correctness Test ===\n");

    let ctx = ActivationMetalContext::new().expect("Failed to create Metal context");

    let test_data: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0];

    let silu_out = ctx.run(&test_data, &ctx.silu_pipeline);
    let gelu_out = ctx.run(&test_data, &ctx.gelu_pipeline);
    let gelu_quick_out = ctx.run(&test_data, &ctx.gelu_quick_pipeline);

    // Reference implementations
    let silu_ref: Vec<f32> = test_data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let gelu_ref: Vec<f32> = test_data.iter().map(|&x| {
        0.5 * x * (1.0 + (0.7978845608 * x * (1.0 + 0.044715 * x * x)).tanh())
    }).collect();
    let gelu_quick_ref: Vec<f32> = test_data.iter().map(|&x| x / (1.0 + (-1.702 * x).exp())).collect();

    println!("{:8} | {:10} | {:10} | {:10}", "Input", "GPU SiLU", "Ref SiLU", "Diff");
    for i in 0..test_data.len() {
        println!("{:8.4} | {:10.6} | {:10.6} | {:10.6}",
            test_data[i], silu_out[i], silu_ref[i], (silu_out[i] - silu_ref[i]).abs());
    }

    let mut max_diff = 0.0f32;
    for i in 0..test_data.len() {
        max_diff = max_diff.max((silu_out[i] - silu_ref[i]).abs());
    }
    println!("SiLU max diff: {}\n", max_diff);

    println!("{:8} | {:10} | {:10} | {:10}", "Input", "GPU GELU", "Ref GELU", "Diff");
    for i in 0..test_data.len() {
        println!("{:8.4} | {:10.6} | {:10.6} | {:10.6}",
            test_data[i], gelu_out[i], gelu_ref[i], (gelu_out[i] - gelu_ref[i]).abs());
    }

    max_diff = 0.0;
    for i in 0..test_data.len() {
        max_diff = max_diff.max((gelu_out[i] - gelu_ref[i]).abs());
    }
    println!("GELU max diff: {}\n", max_diff);

    println!("{:8} | {:10} | {:10} | {:10}", "Input", "GPU Quick", "Ref Quick", "Diff");
    for i in 0..test_data.len() {
        println!("{:8.4} | {:10.6} | {:10.6} | {:10.6}",
            test_data[i], gelu_quick_out[i], gelu_quick_ref[i], (gelu_quick_out[i] - gelu_quick_ref[i]).abs());
    }

    max_diff = 0.0;
    for i in 0..test_data.len() {
        max_diff = max_diff.max((gelu_quick_out[i] - gelu_quick_ref[i]).abs());
    }
    println!("GELU-Quick max diff: {}\n", max_diff);

    // Test large size for performance
    println!("=== Performance Test (16384 elements) ===");
    let large_data: Vec<f32> = (0..16384).map(|i| ((i as f32 - 8192.0) * 0.01).sin()).collect();

    let start = std::time::Instant::now();
    let _ = ctx.run(&large_data, &ctx.silu_pipeline);
    println!("SiLU: {:?} (includes kernel compile)", start.elapsed());

    let start = std::time::Instant::now();
    let _ = ctx.run(&large_data, &ctx.silu_pipeline);
    println!("SiLU: {:?} (after compile)", start.elapsed());

    let start = std::time::Instant::now();
    let _ = ctx.run(&large_data, &ctx.gelu_pipeline);
    println!("GELU: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let _ = ctx.run(&large_data, &ctx.gelu_quick_pipeline);
    println!("GELU-Quick: {:?}", start.elapsed());
}
