// Check SiLU activation behavior
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn main() {
    // Test SiLU with the observed gate values
    let test_values = [26.7985f32, 26.6214, -22.5288, -5.3355];
    
    println!("Testing SiLU activation:");
    for &x in &test_values {
        let silu_x = silu(x);
        println!("  silu({:.4}) = {:.4}", x, silu_x);
    }
    
    // Check multiplication with up values
    let gate_values = [26.7985, -22.5288, -5.3355];
    let up_values = [26.6214, -22.5288, 26.7985];
    
    println!("\nTesting SiLU(gate) * up:");
    for (&gate, &up) in gate_values.iter().zip(up_values.iter()) {
        let silu_gate = silu(gate);
        let product = silu_gate * up;
        println!("  silu({:.4}) * {:.4} = {:.4}", gate, up, product);
    }
    
    // Check if the multiplication is correct
    let gate_max = 26.7985f32;
    let up_max = 26.6214f32;
    let silu_max = silu(gate_max);
    let expected_max = silu_max * up_max;
    
    println!("\nExpected max FFN hidden:");
    println!("  silu({:.4}) * {:.4} = {:.4}", gate_max, up_max, expected_max);
    println!("  Observed max: 713.4146");
}