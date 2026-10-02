// Rust Q4_K MV test - optimized version WITHOUT RefCell overhead
// Run: cargo run --bin rust_q4_k_test_optimized --release

use half::f16;
use infer_train::quant::types::BlockQ4K;
use metal::*;
use std::cell::{Cell, UnsafeCell};
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;
use std::time::Instant;

fn generate_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

// Optimized context - no RefCell, no Option
struct OptimizedMvContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    // Direct storage - no RefCell overhead
    weights_buffer: UnsafeCell<MaybeUninit<Buffer>>,
    input_buffer: UnsafeCell<MaybeUninit<Buffer>>,
    output_buffer: UnsafeCell<MaybeUninit<Buffer>>,
    weights_size: Cell<usize>,
    input_size: Cell<usize>,
    output_size: Cell<usize>,
    weights_valid: Cell<bool>,
    input_valid: Cell<bool>,
    output_valid: Cell<bool>,
}

const FC_MUL_MV_NSG: NSUInteger = 600;
const FC_MUL_MV_NE12: NSUInteger = 602;
const FC_MUL_MV_R2: NSUInteger = 603;
const FC_MUL_MV_R3: NSUInteger = 604;

fn create_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_NSG,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_NE12,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_R2,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_R3,
        );
    }
    cv
}

impl OptimizedMvContext {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal device");
        let queue = device.new_command_queue();

        let source = include_str!("../../shaders/gemm.metal");
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .expect("Failed to compile");
        let cv = create_function_constants(2, 1, 1, 1);
        let kernel = library
            .get_function("kernel_gemm_q4_K_f32", Some(cv))
            .expect("Failed to get kernel");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline");

        Self {
            device,
            queue,
            pipeline,
            weights_buffer: UnsafeCell::new(MaybeUninit::uninit()),
            input_buffer: UnsafeCell::new(MaybeUninit::uninit()),
            output_buffer: UnsafeCell::new(MaybeUninit::uninit()),
            weights_size: Cell::new(0),
            input_size: Cell::new(0),
            output_size: Cell::new(0),
            weights_valid: Cell::new(false),
            input_valid: Cell::new(false),
            output_valid: Cell::new(false),
        }
    }

    fn mv(&self, m: usize, k: usize, weights: &[BlockQ4K], input: &[f32]) {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Optimized buffer access - no borrow_mut, no Option
        unsafe {
            let weights_buffer = if self.weights_size.get() != weights_size
                || !self.weights_valid.get()
            {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const c_void,
                    weights_size as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                ptr::write(self.weights_buffer.get(), MaybeUninit::new(buf));
                self.weights_size.set(weights_size);
                self.weights_valid.set(true);
                (*self.weights_buffer.get()).assume_init_ref()
            } else {
                (*self.weights_buffer.get()).assume_init_ref()
            };

            let input_buffer = if self.input_size.get() != input_size
                || !self.input_valid.get()
            {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const c_void,
                    input_size as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                ptr::write(self.input_buffer.get(), MaybeUninit::new(buf));
                self.input_size.set(input_size);
                self.input_valid.set(true);
                (*self.input_buffer.get()).assume_init_ref()
            } else {
                (*self.input_buffer.get()).assume_init_ref()
            };

            let output_buffer = if self.output_size.get() != output_size
                || !self.output_valid.get()
            {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                ptr::write(self.output_buffer.get(), MaybeUninit::new(buf));
                self.output_size.set(output_size);
                self.output_valid.set(true);
                (*self.output_buffer.get()).assume_init_ref()
            } else {
                (*self.output_buffer.get()).assume_init_ref()
            };

            // Execute kernel
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(weights_buffer), 0);
            encoder.set_buffer(2, Some(input_buffer), 0);
            encoder.set_buffer(3, Some(output_buffer), 0);

            let rows = 4u64;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (m as u64 + rows - 1) / rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize { width: 32, height: 2, depth: 1 },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m = if args.len() > 1 { args[1].parse().unwrap() } else { 4096 };
    let k = if args.len() > 2 { args[2].parse().unwrap() } else { 4096 };
    let iter = if args.len() > 3 { args[3].parse().unwrap() } else { 10000 };

    println!("Rust Q4_K MV (Optimized): M={}, K={}, iter={}", m, k, iter);

    let weights = generate_weights(m, k);
    let input = generate_input(k);

    let ctx = OptimizedMvContext::new();

    // Warmup
    ctx.mv(m, k, &weights, &input);

    // Benchmark
    let t0 = Instant::now();
    for _ in 0..iter {
        ctx.mv(m, k, &weights, &input);
    }
    let us = t0.elapsed().as_micros() as f64 / iter as f64;
    println!(
        "Avg: {:.2} us, GFLOPS: {:.2}",
        us,
        (2.0 * m as f64 * k as f64) / (us * 1e-6) / 1e9
    );
}
