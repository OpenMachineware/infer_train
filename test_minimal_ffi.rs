use std::time::Instant;
use std::alloc::{alloc, Layout};

extern "C" {
    fn decode_attention_f32_q_f16_kv(
        q: *const f32, k: *const u16, v: *const u16, output: *mut f32,
        num_heads: i32, head_dim: i32, seq_len: i32, scale: f32,
        vkq32: *mut f32, q_f16: *mut u16, partials: *mut f32, chunk_size: i32
    ) -> i32;
}

fn main() {
    let num_heads = 32;
    let head_dim = 128;
    let seq_len = 1024;
    
    let q_layout = Layout::from_size_align(num_heads * head_dim * 4, 64).unwrap();
    let kv_layout = Layout::from_size_align(seq_len * num_heads * head_dim * 2, 64).unwrap();
    let out_layout = Layout::from_size_align(num_heads * head_dim * 4, 64).unwrap();
    
    unsafe {
        let q = alloc(q_layout) as *mut f32;
        let k = alloc(kv_layout) as *mut u16;
        let v = alloc(kv_layout) as *mut u16;
        let output = alloc(out_layout) as *mut f32;
        let vkq32 = alloc(Layout::from_size_align(head_dim * 4, 64).unwrap()) as *mut f32;
        let q_f16 = alloc(Layout::from_size_align(head_dim * 2, 64).unwrap()) as *mut u16;
        
        // Initialize data without half crate
        for i in 0..(num_heads * head_dim) {
            *q.add(i) = ((i as i32 % 2000) - 1000) as f32 / 1000.0;
        }
        for i in 0..(seq_len * num_heads * head_dim) {
            *k.add(i) = 0x3800; // 0.5 in FP16
            *v.add(i) = 0x3400; // 0.3 in FP16
        }
        
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Warmup
        for _ in 0..10 {
            decode_attention_f32_q_f16_kv(q, k, v, output, num_heads as i32, head_dim as i32, seq_len as i32, scale, vkq32, q_f16, std::ptr::null_mut(), 0);
        }
        
        // Measure
        let start = Instant::now();
        for _ in 0..20 {
            decode_attention_f32_q_f16_kv(q, k, v, output, num_heads as i32, head_dim as i32, seq_len as i32, scale, vkq32, q_f16, std::ptr::null_mut(), 0);
        }
        let elapsed = start.elapsed();
        
        let flops = 2.0 * num_heads as f64 * head_dim as f64 * seq_len as f64;
        let gflops = (flops * 20.0) / elapsed.as_secs_f64() / 1e9;
        println!("Minimal FFI: {:.2} GFLOPS", gflops);
    }
}
