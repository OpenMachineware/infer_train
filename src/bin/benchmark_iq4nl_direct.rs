// 直接测量，排除框架开销
use std::time::Instant;
use infer_train::quant::kv_cache::{BlockIQ4NL, QK};

fn main() {
    let num_heads = 32;
    let head_dim = 128;
    let seq_len = 512;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let blocks_per_row = head_dim / QK;
    let num_blocks = seq_len * num_heads * blocks_per_row;

    let q: Vec<f32> = (0..num_heads * head_dim).map(|i| (i % 100) as f32 / 100.0).collect();
    let mut output = vec![0.0f32; num_heads * head_dim];
    let k: Vec<BlockIQ4NL> = (0..num_blocks).map(|_| BlockIQ4NL { d: 0x3C00, qs: [0x88u8; 16] }).collect();
    let v: Vec<BlockIQ4NL> = (0..num_blocks).map(|_| BlockIQ4NL { d: 0x3C00, qs: [0x88u8; 16] }).collect();

    unsafe {
        // Warmup
        infer_train::ops::attention::decode_attention_iq4_nl_neon(&q, &k, &v, &mut output, num_heads, head_dim, seq_len, scale);

        let start = Instant::now();
        for _ in 0..20 {
            infer_train::ops::attention::decode_attention_iq4_nl_neon(&q, &k, &v, &mut output, num_heads, head_dim, seq_len, scale);
        }
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64 / 20.0;
        let flops = 2.0 * num_heads as f64 * head_dim as f64 * seq_len as f64;
        let gflops = flops / (ns / 1e9) / 1e9;
        println!("IQ4_NL Direct: {:.2} GFLOPS (avg {:.0} ns)", gflops, ns);
    }
}
