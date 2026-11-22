//! Test Q6_K bit decoding

fn main() {
    // Test the bit decoding logic
    // Q6_K format:
    // - ql[128]: low 4 bits (each byte holds 2 elements: low nibble + high nibble)
    // - qh[64]: high 2 bits (each byte holds 4 elements in bits 0-1, 2-3, 4-5, 6-7)

    // Test case: q6 = 33 (binary: 100001)
    // low 4 bits = 0001 = 1
    // high 2 bits = 10 = 2
    // Combined: 1 | (2 << 4) = 33

    let ql: [u8; 128] = {
        let mut arr = [0u8; 128];
        arr[0] = 1; // low nibble = 1 for first element
        arr
    };

    let qh: [u8; 64] = {
        let mut arr = [0u8; 64];
        arr[0] = 2; // bits 0-1 = 2
        arr
    };

    // Decode using llama.cpp logic:
    // a[l+0] = (ql[l+0] & 0xF) | (((qh[l] >> 0) & 3) << 4)
    let a0 = ((ql[0] & 0xF) as i32) | ((((qh[0] >> 0) & 3) as i32) << 4);
    println!("Decoded q6 value for a[0]: {} (should be 33)", a0);

    // After -32:
    let a0_corrected = a0 - 32;
    println!("After -32 correction: {} (should be 1)", a0_corrected);

    // Test with our NEON logic:
    // q6h_0 = ((qh & 3) << 4)
    let q6h_0 = ((qh[0] & 3) << 4) as i32;
    println!("\nOur NEON decoding:");
    println!("  q6h_0 = {} (should be 32)", q6h_0);

    // ql low nibble
    let ql_low = (ql[0] & 0xF) as i32;
    println!("  ql_low = {} (should be 1)", ql_low);

    // Combined
    let q6_val = ql_low | q6h_0;
    println!("  Combined q6 = {} (should be 33)", q6_val);

    // Test 2: Check multiple elements from a qh byte
    println!("\n=== Test qh byte with multiple values ===");
    // qh[0] = 0b10101010 = 170
    // bits 0-1 = 2, bits 2-3 = 2, bits 4-5 = 1, bits 6-7 = 2
    let qh_byte: u8 = 0b10101010;
    println!("qh = {} (0b{:08b})", qh_byte, qh_byte);
    println!("  bits 0-1: {} (should be 2)", qh_byte & 3);
    println!("  bits 2-3: {} (should be 2)", (qh_byte >> 2) & 3);
    println!("  bits 4-5: {} (should be 1)", (qh_byte >> 4) & 3);
    println!("  bits 6-7: {} (should be 2)", (qh_byte >> 6) & 3);

    // Corresponding q6 values (with ql=0):
    println!("\nWith ql=0:");
    println!(
        "  a[0]: {} (q6={}-32={})",
        (qh_byte & 3) << 4,
        (qh_byte & 3) << 4,
        ((qh_byte & 3) << 4) as i32 - 32
    );
    println!(
        "  a[32]: {} (q6={}-32={})",
        ((qh_byte >> 2) & 3) << 4,
        ((qh_byte >> 2) & 3) << 4,
        (((qh_byte >> 2) & 3) << 4) as i32 - 32
    );
    println!(
        "  a[64]: {} (q6={}-32={})",
        ((qh_byte >> 4) & 3) << 4,
        ((qh_byte >> 4) & 3) << 4,
        (((qh_byte >> 4) & 3) << 4) as i32 - 32
    );
    println!(
        "  a[96]: {} (q6={}-32={})",
        ((qh_byte >> 6) & 3) << 4,
        ((qh_byte >> 6) & 3) << 4,
        (((qh_byte >> 6) & 3) << 4) as i32 - 32
    );

    println!("\n=== Test simple case ===");
    // All zeros: q6 = 0, after -32 = -32
    println!("ql=0, qh=0:");
    println!("  q6 value = 0");
    println!("  after -32 = -32");
    println!("  With q8=1, scale=1, expected result = -32 * 256 = -8192");

    // All 32: q6 = 32, after -32 = 0
    // To get q6=32: low 4 bits = 0, high 2 bits = 2
    // qh needs bits 0-1, 2-3, 4-5, 6-7 all = 2
    // qh = (2<<0) | (2<<2) | (2<<4) | (2<<6) = 2 | 8 | 32 | 128 = 170
    println!("\nql=0, qh=170:");
    println!("  q6 value for a[0] = 0 | (2<<4) = 32");
    println!("  after -32 = 0");
    println!("  With q8=1, scale=1, expected result = 0");
}
