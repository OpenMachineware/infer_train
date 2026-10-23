// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile buffer allocation and reuse during decode

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static BUFFER_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static BUFFER_REUSE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{Buffer, Device, MTLResourceOptions};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
struct CountingForwardBufferPool {
    device: Device,
    buffers: Vec<Rc<Buffer>>,
    reuse_pool: Rc<RefCell<Vec<Rc<Buffer>>>>,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl CountingForwardBufferPool {
    fn get(&mut self, size: usize) -> Rc<Buffer> {
        let size_class = size.next_power_of_two();

        // Try to reuse
        {
            let mut pool = self.reuse_pool.borrow_mut();
            if let Some(idx) = pool.iter().position(|b| b.length() as usize >= size_class) {
                let buf = pool.remove(idx);
                self.buffers.push(buf.clone());
                BUFFER_REUSE_COUNT.fetch_add(1, Ordering::SeqCst);
                return buf;
            }
        }

        // Allocate new
        let buf = Rc::new(self.device.new_buffer(
            size_class as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        self.buffers.push(buf.clone());
        BUFFER_ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        buf
    }
}

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("=== Buffer Lifecycle Profile ===\n");
        println!("Model: {}", model_path);

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device().clone();
        let mut ctx = InferenceContext::from_gguf_with_gpu_ctx(model_path, gpu_ctx.clone()).unwrap();

        // Create KV cache
        let mut kv = KVCacheGPU::new(&device, &ctx.model.config, 2048);

        // Prefill
        println!("\n=== Prefill ===");
        let prefill_tokens: Vec<u32> = (1..=10).collect();
        BUFFER_ALLOC_COUNT.store(0, Ordering::SeqCst);
        BUFFER_REUSE_COUNT.store(0, Ordering::SeqCst);
        ctx.model.forward_gpu_native_batched(&prefill_tokens, &mut kv, &gpu_ctx).unwrap();
        println!("Allocated: {}", BUFFER_ALLOC_COUNT.load(Ordering::SeqCst));
        println!("Reused: {}", BUFFER_REUSE_COUNT.load(Ordering::SeqCst));

        // Decode multiple tokens
        println!("\n=== Decode (10 tokens) ===");
        for i in 0..10 {
            BUFFER_ALLOC_COUNT.store(0, Ordering::SeqCst);
            BUFFER_REUSE_COUNT.store(0, Ordering::SeqCst);
            let token = (65 + i) as u32;
            ctx.model.forward_gpu_native_batched(&[token], &mut kv, &gpu_ctx).unwrap();
            println!("Token {}: alloc={}, reuse={}", 
                i + 1,
                BUFFER_ALLOC_COUNT.load(Ordering::SeqCst),
                BUFFER_REUSE_COUNT.load(Ordering::SeqCst));
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}