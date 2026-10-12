// Comprehensive benchmark for all K-quant formats
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const QK_K: usize = 256;

#[repr(C)]
struct MulMVArgs {
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

struct KQuantInfo {
    name: &'static str,
    block_size: usize,
    nsg: i16,
    nr0: i16,
}

fn create_fc(nsg: i16, ne12: i16, r2: i16, r3: i16) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const _ as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const _ as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const _ as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let shader = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let compile_opts = CompileOptions::new();
    let lib = device.new_library_with_source(shader, &compile_opts).unwrap();

    let m = 1024usize;
    let k = 4096usize;

    println!("=== All K-quant MV Optimized Benchmark ===");
    println!("M={}, K={}, 500 iterations, same shader\n", m, k);
    println!(
        "{:8} | {:>10} | {:>10} | {:>6}",
        "Format", "Time (us)", "GFLOPS", "Status"
    );
    println!("{}", "-".repeat(50));

    let k_quants = vec![
        KQuantInfo { name: "q2_K", block_size: 164, nsg: 2, nr0: 4 },
        KQuantInfo { name: "q3_K", block_size: 110, nsg: 2, nr0: 2 },
        KQuantInfo { name: "q4_K", block_size: 144, nsg: 2, nr0: 2 },
        KQuantInfo { name: "q5_K", block_size: 176, nsg: 2, nr0: 1 },
        KQuantInfo { name: "q6_K", block_size: 210, nsg: 2, nr0: 2 },
    ];

    for info in &k_quants {
        let kernel_name = format!("kernel_mul_mm_{}_f32", info.name);
        let func = lib
            .get_function(&kernel_name, Some(create_fc(info.nsg, 1, 1, 1)))
            .unwrap();
        let pipeline =
            device.new_compute_pipeline_state_with_function(&func).unwrap();

        let nb = k / QK_K;
        let weights_size = m * nb * info.block_size;
        let weights_data: Vec<u8> =
            (0..weights_size).map(|i| ((i * 17 + 13) % 256) as u8).collect();
        let input: Vec<f32> =
            (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

        let weights_buf = device.new_buffer_with_data(
            weights_data.as_ptr() as *const _,
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buf = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device
            .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

        let args = MulMVArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01: (nb * info.block_size) as u64,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: (k * 4) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 1,
            r2: 1,
            r3: 1,
        };

        let grid = MTLSize::new(
            ((m + (info.nsg as usize * info.nr0 as usize) - 1)
                / (info.nsg as usize * info.nr0 as usize)) as u64,
            1,
            1,
        );
        let tg = MTLSize::new((info.nsg as u64) * 32, 1, 1);

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MulMVArgs>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&output_buf), 0);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark
        const ITERATIONS: usize = 500;
        let mut times = Vec::with_capacity(ITERATIONS);

        for _ in 0..ITERATIONS {
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MulMVArgs>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&output_buf), 0);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(start.elapsed().as_nanos() as f64);
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[ITERATIONS / 2];
        let gflops = (2.0 * m as f64 * k as f64) / median / 1000.0;

        println!(
            "{:8} | {:>10.2} | {:>10.1} | ✓",
            info.name,
            median / 1000.0,
            gflops
        );
    }
}
