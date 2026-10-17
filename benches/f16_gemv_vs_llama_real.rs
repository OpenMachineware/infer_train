// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark: Our F16 GEMV vs llama.cpp REAL Metal kernel
// Fair comparison: both allocate new buffers each iteration

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
#[repr(C)]
#[derive(Clone, Copy)]
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
    nsg: i16,
    ne12_batch: i16,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::MTLSize;

    let ctx = MetalContext::new().expect("Failed to create MetalContext");
    let device = ctx.device();
    let queue = ctx.queue();

    // Compile both kernels with same options
    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Our kernel source
    let our_source =
        include_str!("../src/ops/metal/gemv/mul_mv_f16_kernels.metal");

    let our_library = device
        .new_library_with_source(our_source, &compile_options)
        .expect("Failed to compile our kernels");

    // Create function constants for different NSG values
    let mut fc_nsg1 = metal::FunctionConstantValues::new();
    let nsg1: i16 = 1;
    fc_nsg1.set_constant_value_at_index(
        &nsg1 as *const i16 as *const std::ffi::c_void,
        metal::MTLDataType::Short,
        600,
    );
    let our_nsg1_kernel = our_library
        .get_function("kernel_mul_mv_f16_f32_4", Some(fc_nsg1))
        .expect("Failed to get our nsg1 kernel");
    let our_nsg1_pipeline = device
        .new_compute_pipeline_state_with_function(&our_nsg1_kernel)
        .expect("Failed to create our nsg1 pipeline");

    let mut fc_nsg2 = metal::FunctionConstantValues::new();
    let nsg2: i16 = 2;
    fc_nsg2.set_constant_value_at_index(
        &nsg2 as *const i16 as *const std::ffi::c_void,
        metal::MTLDataType::Short,
        600,
    );
    let our_nsg2_kernel = our_library
        .get_function("kernel_mul_mv_f16_f32_4", Some(fc_nsg2))
        .expect("Failed to get our nsg2 kernel");
    let our_nsg2_pipeline = device
        .new_compute_pipeline_state_with_function(&our_nsg2_kernel)
        .expect("Failed to create our nsg2 pipeline");

    let mut fc_nsg4 = metal::FunctionConstantValues::new();
    let nsg4: i16 = 4;
    fc_nsg4.set_constant_value_at_index(
        &nsg4 as *const i16 as *const std::ffi::c_void,
        metal::MTLDataType::Short,
        600,
    );
    let our_nsg4_kernel = our_library
        .get_function("kernel_mul_mv_f16_f32_4", Some(fc_nsg4))
        .expect("Failed to get our nsg4 kernel");
    let our_nsg4_pipeline = device
        .new_compute_pipeline_state_with_function(&our_nsg4_kernel)
        .expect("Failed to create our nsg4 pipeline");

    // llama.cpp kernel source (exact copy from their implementation)
    let llama_source = r#"
#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
    int16_t  nsg;
    int16_t  ne12_batch;
} MvArgs;

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<short NSG, short NR0, typename T0, typename T04, typename T1, typename T14>
void kernel_mul_mv_t_t_4_impl(
        constant MvArgs & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;

    device const T1  * y  = (device const T1  *) (src1 + r1*args.nb11);
    device const T14 * y4 = (device const T14 *) (src1 + r1*args.nb11);

    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax [row] = (device const T0  *) (src0 + (r0 + row)*args.nb01);
        ax4[row] = (device const T04 *) (src0 + (r0 + row)*args.nb01);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<short NSG>
kernel void llama_kernel_mul_mv_f16_f32_4_nsg(
        constant MvArgs & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_impl<NSG, 2, half, half4, float, float4>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(llama_kernel_mul_mv_f16_f32_4_nsg<1>) llama_nsg1_t;
typedef decltype(llama_kernel_mul_mv_f16_f32_4_nsg<2>) llama_nsg2_t;
typedef decltype(llama_kernel_mul_mv_f16_f32_4_nsg<4>) llama_nsg4_t;

template [[host_name("llama_kernel_mul_mv_f16_f32_4_nsg1")]] kernel llama_nsg1_t llama_kernel_mul_mv_f16_f32_4_nsg<1>;
template [[host_name("llama_kernel_mul_mv_f16_f32_4_nsg2")]] kernel llama_nsg2_t llama_kernel_mul_mv_f16_f32_4_nsg<2>;
template [[host_name("llama_kernel_mul_mv_f16_f32_4_nsg4")]] kernel llama_nsg4_t llama_kernel_mul_mv_f16_f32_4_nsg<4>;
"#;

    let llama_library = device
        .new_library_with_source(llama_source, &compile_options)
        .expect("Failed to compile llama.cpp kernels");

    let llama_nsg1_kernel = llama_library
        .get_function("llama_kernel_mul_mv_f16_f32_4_nsg1", None)
        .expect("Failed to get llama nsg1 kernel");
    let llama_nsg1_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_nsg1_kernel)
        .expect("Failed to create llama nsg1 pipeline");

    let llama_nsg2_kernel = llama_library
        .get_function("llama_kernel_mul_mv_f16_f32_4_nsg2", None)
        .expect("Failed to get llama nsg2 kernel");
    let llama_nsg2_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_nsg2_kernel)
        .expect("Failed to create llama nsg2 pipeline");

    let llama_nsg4_kernel = llama_library
        .get_function("llama_kernel_mul_mv_f16_f32_4_nsg4", None)
        .expect("Failed to get llama nsg4 kernel");
    let llama_nsg4_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_nsg4_kernel)
        .expect("Failed to create llama nsg4 pipeline");

    println!("=== F16 GEMV: Our Kernel vs llama.cpp Kernel ===");
    println!(
        "Fair comparison: both allocate new output buffers each iteration\n"
    );

    let test_sizes: Vec<(usize, usize)> = vec![
        (128, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (8192, 16384),
    ];

    println!("Size (K×M) | Our (μs) | llama.cpp (μs) | Ratio");
    println!("-----------|----------|----------------|-------");

    for (k, m) in test_sizes {
        // Generate test data
        let weights_f16: Vec<u16> = (0..m * k)
            .map(|i| ((i as f32 * 0.1).sin() * 100.0) as u16)
            .collect();
        let input_f32: Vec<f32> =
            (0..k).map(|i| (i as f32 * 0.05).cos()).collect();

        // Create input buffers once
        let weights_buffer = device.new_buffer_with_data(
            weights_f16.as_ptr() as *const std::ffi::c_void,
            (m * k * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input_f32.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let iterations = 100;
        let nsg = std::cmp::min(4, (k + 127) / 128);

        // Prepare args once
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 2,
            nb01: (k * 2) as u64,
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
            nr0: 2,
            r2: 1,
            r3: 1,
            nsg: nsg as i16,
            ne12_batch: 1,
        };

        let threadgroup_mem_size =
            (2 * 32 * nsg as usize) * std::mem::size_of::<f32>();
        let tg_count =
            MTLSize { width: ((m + 1) / 2) as u64, height: 1, depth: 1 };
        let tg_size = MTLSize { width: (nsg as u64 * 32), height: 1, depth: 1 };

        let (our_pipeline, llama_pipeline) = match nsg {
            1 => (&our_nsg1_pipeline, &llama_nsg1_pipeline),
            2 => (&our_nsg2_pipeline, &llama_nsg2_pipeline),
            _ => (&our_nsg4_pipeline, &llama_nsg4_pipeline),
        };

        // Warmup both
        for _ in 0..10 {
            // Our kernel
            {
                let output_buffer = device.new_buffer(
                    (m * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(our_pipeline);
                encoder.set_bytes(
                    0,
                    std::mem::size_of::<MvArgs>() as u64,
                    &args as *const MvArgs as *const std::ffi::c_void,
                );
                encoder.set_buffer(1, Some(&weights_buffer), 0);
                encoder.set_buffer(2, Some(&input_buffer), 0);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                encoder.set_threadgroup_memory_length(
                    0,
                    threadgroup_mem_size as u64,
                );
                encoder.dispatch_thread_groups(tg_count, tg_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            // llama.cpp kernel
            {
                let output_buffer = device.new_buffer(
                    (m * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(llama_pipeline);
                encoder.set_bytes(
                    0,
                    std::mem::size_of::<MvArgs>() as u64,
                    &args as *const MvArgs as *const std::ffi::c_void,
                );
                encoder.set_buffer(1, Some(&weights_buffer), 0);
                encoder.set_buffer(2, Some(&input_buffer), 0);
                encoder.set_buffer(3, Some(&output_buffer), 0);
                encoder.set_threadgroup_memory_length(
                    0,
                    threadgroup_mem_size as u64,
                );
                encoder.dispatch_thread_groups(tg_count, tg_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
        }

        // Benchmark our kernel
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let output_buffer = device.new_buffer(
                (m * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(our_pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder
                .set_threadgroup_memory_length(0, threadgroup_mem_size as u64);
            encoder.dispatch_thread_groups(tg_count, tg_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let our_time_us =
            start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark llama.cpp kernel
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let output_buffer = device.new_buffer(
                (m * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(llama_pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder
                .set_threadgroup_memory_length(0, threadgroup_mem_size as u64);
            encoder.dispatch_thread_groups(tg_count, tg_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let llama_time_us =
            start.elapsed().as_micros() as f64 / iterations as f64;

        let ratio = (llama_time_us / our_time_us) * 100.0;

        println!(
            "{:4}×{:5} | {:8.2} | {:14.2} | {:5.1}%",
            k, m, our_time_us, llama_time_us, ratio
        );
    }

    println!("\n=== Interpretation ===");
    println!("Ratio > 100%: Our kernel is FASTER than llama.cpp");
    println!("Ratio < 100%: Our kernel is SLOWER - need to optimize");
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support on aarch64");
}
