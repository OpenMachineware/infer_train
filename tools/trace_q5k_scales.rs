// Trace Q5_K scale extraction
use bytemuck::Zeroable;
use half::f16;
use infer_train::quant::types::BlockQ5K;

fn main() {
    let mut block = BlockQ5K::zeroed();
    block.d = f16::from_f32(1.0).to_bits();
    block.dmin = f16::from_f32(0.0).to_bits();

    // Set scales to distinct values for debugging
    // scales[0] = 10 (scale for group 0, lower nibbles)
    // scales[1] = 20 (scale for group 0, upper nibbles)
    // scales[4] = 30 (min for group 0, lower nibbles)
    // scales[5] = 40 (min for group 0, upper nibbles)
    for i in 0..12 {
        block.scales[i] = (i * 10 + 10) as u8;
    }

    println!("Scales array:");
    for i in 0..12 {
        println!("  scales[{}] = {}", i, block.scales[i]);
    }

    println!("\nCPU extraction for group 0 (is=0, is+1=1):");
    let scale0 = block.scales[0] & 63;
    let min0 = block.scales[4] & 63;
    let scale1 = block.scales[1] & 63;
    let min1 = block.scales[5] & 63;
    println!("  scale0 = scales[0] & 63 = {}", scale0);
    println!("  min0 = scales[4] & 63 = {}", min0);
    println!("  scale1 = scales[1] & 63 = {}", scale1);
    println!("  min1 = scales[5] & 63 = {}", min1);

    println!("\nGPU extraction with iq=0:");
    // Cast to uint16_t array
    let a: &[u16] = unsafe {
        std::slice::from_raw_parts(block.scales.as_ptr() as *const u16, 6)
    };

    let kmask1: u16 = 0x3f3f;
    let kmask2: u16 = 0x0f0f;
    let kmask3: u16 = 0xc0c0;

    let sc16_0 = a[0] & kmask1;
    let sc16_1 = a[2] & kmask1;
    let sc16_2 = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
    let sc16_3 = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

    let sc8: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &[sc16_0, sc16_1, sc16_2, sc16_3] as *const u16 as *const u8,
            8,
        )
    };

    println!("  a[0] = 0x{:04x} (scales[0..1])", a[0]);
    println!("  a[2] = 0x{:04x} (scales[4..5])", a[2]);
    println!("  a[4] = 0x{:04x} (scales[8..9])", a[4]);

    println!("\n  sc16[0] = 0x{:04x}", sc16_0);
    println!("  sc16[1] = 0x{:04x}", sc16_1);
    println!("  sc16[2] = 0x{:04x}", sc16_2);
    println!("  sc16[3] = 0x{:04x}", sc16_3);

    println!("\n  sc8[0] = {} (should be scale0 = {})", sc8[0], scale0);
    println!("  sc8[1] = {} (should be scale1 = {})", sc8[1], scale1);
    println!("  sc8[2] = {} (should be min0 = {})", sc8[2], min0);
    println!("  sc8[3] = {} (should be min1 = {})", sc8[3], min1);
    println!(
        "  sc8[4] = {} (should be scales[4]&63 = {})",
        sc8[4],
        block.scales[4] & 63
    );
    println!(
        "  sc8[5] = {} (should be scales[5]&63 = {})",
        sc8[5],
        block.scales[5] & 63
    );
    println!("  sc8[6] = {} (from packed)", sc8[6]);
    println!("  sc8[7] = {} (from packed)", sc8[7]);

    // Verify GPU formula usage
    println!("\nGPU formula uses:");
    println!(
        "  dh[0] * (sc8[0] * ... + sc8[1] * ... + sc8[4] * ... + sc8[5] * ...)"
    );
    println!("  dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + ...)");

    println!("\nBut CPU expects:");
    println!("  d * scale * q - dmin * min");
    println!("  Group 0: d * {} * q - dmin * {} (lower nibbles)", scale0, min0);
    println!("  Group 0: d * {} * q - dmin * {} (upper nibbles)", scale1, min1);
}
