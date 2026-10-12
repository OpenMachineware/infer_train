// K-quant MV kernels: llama.cpp vs self-written
// Alternating benchmark, fair comparison

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const FC_MUL_MV: NSUInteger = 600;

fn create_fc(nsg: i16, ne12: i16, r2: i16, r3: i16) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 0,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 2,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 3,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 4,
        );
    }
    cv
}

#[repr(C)]
struct Args {
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

struct Info {
    name: &'static str,
    nsg: i16,
    nr0: i32,
    block_size: usize,
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let m = 1024;
    let k = 4096;

    // Load shaders from project's tmp directory
    let llama_shader = std::fs::read_to_string("/tmp/llama_full.metal")
        .expect("Run: cp /tmp/llama_full.metal to project");
    let my_shader = include_str!("../src/ops/metal/gemm/mul_mm.metal");

    let opts = CompileOptions::new();
    let llama_lib = match device.new_library_with_source(&llama_shader, &opts) {
        Ok(lib) => {
            println!("llama.cpp shader compiled successfully");
            lib
        }
        Err(e) => {
            eprintln!("llama.cpp shader compilation failed: {:?}", e);
            std::process::exit(1);
        }
    };
    println!("My shader compiling...");
    let my_lib = device.new_library_with_source(my_shader, &opts).unwrap();
    println!("My shader compiled successfully");
    let my_lib = device.new_library_with_source(my_shader, &opts).unwrap();

    let k_quants = vec![
        Info { name: "q2_K", nsg: 2, nr0: 4, block_size: 164 },
        Info { name: "q3_K", nsg: 2, nr0: 2, block_size: 110 },
        Info { name: "q4_K", nsg: 2, nr0: 2, block_size: 144 },
        Info { name: "q5_K", nsg: 2, nr0: 1, block_size: 208 },
        Info { name: "q6_K", nsg: 2, nr0: 2, block_size: 210 },
    ];

    println!("=== K-quant MV: llama.cpp vs Self-written ===");
    println!("M={}, K={}, 500 alternating iterations\n", m, k);
    println!(
        "{:8} | {:>10} | {:>10} | {:>6} | {:>6}",
        "Format", "llama.cpp", "Self", "Ratio", "Correct"
    );
    println!("{}", "-".repeat(55));

    for info in &k_quants {
        let cv = create_fc(info.nsg, 1, 1, 1);

        let llama_func = llama_lib
            .get_function(
                &format!("kernel_mul_mm_{}_f32", info.name),
                Some(cv.clone()),
            )
            .unwrap();
        let my_func = my_lib
            .get_function(&format!("kernel_mul_mm_{}_f32", info.name), Some(cv))
            .unwrap();

        let llama_pipe = device
            .new_compute_pipeline_state_with_function(&llama_func)
            .unwrap();
        let my_pipe =
            device.new_compute_pipeline_state_with_function(&my_func).unwrap();

        let nb = k / 256;
        let weights_size = m * nb * info.block_size;
        let weights: Vec<u8> =
            (0..weights_size).map(|i| ((i * 17 + 13) % 256) as u8).collect();
        let input: Vec<f32> =
            (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

        let weights_buf = device.new_buffer_with_data(
            weights.as_ptr() as *const _,
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buf = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let llama_out = device
            .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
        let my_out = device
            .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

        let args = Args {
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
            ((m + info.nsg as usize * info.nr0 as usize - 1)
                / (info.nsg as usize * info.nr0 as usize)) as u64,
            1,
            1,
        );
        let tg = MTLSize::new(info.nsg as u64 * 32, 1, 1);

        // Warmup
        for _ in 0..20 {
            for (pipe, out) in [(&llama_pipe, &llama_out), (&my_pipe, &my_out)]
            {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                enc.set_bytes(
                    0,
                    std::mem::size_of::<Args>() as u64,
                    &args as *const _ as *const c_void,
                );
                enc.set_buffer(1, Some(&weights_buf), 0);
                enc.set_buffer(2, Some(&input_buf), 0);
                enc.set_buffer(3, Some(out), 0);
                enc.dispatch_thread_groups(grid, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }

        // Alternating benchmark
        const ITER: usize = 500;
        let mut llama_times = Vec::with_capacity(ITER);
        let mut my_times = Vec::with_capacity(ITER);

        for _ in 0..ITER {
            let t1 = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&llama_pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<Args>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&llama_out), 0);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            llama_times.push(t1.elapsed().as_nanos() as f64);

            let t2 = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&my_pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<Args>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&my_out), 0);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            my_times.push(t2.elapsed().as_nanos() as f64);
        }

        llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        my_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let llama_med = llama_times[ITER / 2];
        let my_med = my_times[ITER / 2];
        let ratio = my_med / llama_med * 100.0;

        // Correctness
        let llama_o = unsafe {
            std::slice::from_raw_parts(llama_out.contents() as *const f32, m)
        };
        let my_o = unsafe {
            std::slice::from_raw_parts(my_out.contents() as *const f32, m)
        };
        let correct = (0..m)
            .filter(|&i| {
                let rel = if llama_o[i].abs() > 1e-6 {
                    (my_o[i] - llama_o[i]).abs() / llama_o[i].abs()
                } else {
                    (my_o[i] - llama_o[i]).abs()
                };
                rel < 0.01
            })
            .count();

        let status = if ratio < 100.0 { "✓" } else { "" };
        println!(
            "{:8} | {:>8.2} µs | {:>8.2} µs | {:>5.1}% | {:>5.1}% {}",
            info.name,
            llama_med / 1000.0,
            my_med / 1000.0,
            ratio,
            correct as f64 / m as f64 * 100.0,
            status
        );
    }

    println!("\nRatio < 100% means self-written is FASTER than llama.cpp");
}
