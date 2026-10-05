// Trace sign encoding for IQ2_XXS to find the exact issue
// Compare quantizer output vs kernel interpretation

fn main() {
    println!("=== IQ2_XXS Sign Encoding Trace ===\n");

    // The ksigns_iq2xs table from llama.cpp (uint8_t)
    let ksigns_iq2xs: [u8; 128] = [
        0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144,
        17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160,
        33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48,
        177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
        192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
        80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
        96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237,
        238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
        252, 125, 126, 255,
    ];

    let kmask_iq2xs: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    // Test case 1: All positive values
    println!("Test 1: All positive values [1, 1, 1, 1, 1, 1, 1, 1]");
    let input = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let weight = [1.0f32; 8];

    // Quantization logic (llama.cpp lines 3340-3361)
    let mut nflip = 0;
    let mut s: u8 = 0;
    let mut xval = [0.0f32; 8];

    for i in 0..8 {
        if input[i] >= 0.0 {
            xval[i] = input[i];
        } else {
            xval[i] = -input[i];
            nflip += 1;
            s |= 1 << i;
        }
    }

    println!("  nflip = {}", nflip);
    println!("  s (before parity fix) = {:07b}", s);

    // Parity constraint
    if nflip % 2 != 0 {
        let mut min_idx = 0;
        let mut min_val = weight[0] * input[0] * input[0];
        for i in 1..8 {
            let ax = weight[i] * input[i] * input[i];
            if ax < min_val {
                min_val = ax;
                min_idx = i;
            }
        }
        xval[min_idx] = -xval[min_idx];
        s ^= 1 << min_idx;
        println!("  Flipped bit {} to fix parity", min_idx);
    }

    let stored_index = s & 127;
    println!("  Stored 7-bit index = {} ({:07b})", stored_index, stored_index);

    // Kernel interpretation
    let signs = ksigns_iq2xs[stored_index as usize];
    println!("  ksigns_iq2xs[{}] = {} ({:08b})", stored_index, signs, signs);

    print!("  Dequantized signs: ");
    for j in 0..8 {
        let is_neg = (signs & kmask_iq2xs[j]) != 0;
        let sign = if is_neg { -1.0f32 } else { 1.0f32 };
        print!("{} ", sign);
    }
    println!();

    // Test case 2: One negative value
    println!("\nTest 2: One negative [1, -1, 1, 1, 1, 1, 1, 1]");
    let input2 = [1.0f32, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let mut nflip2 = 0;
    let mut s2: u8 = 0;
    let mut xval2 = [0.0f32; 8];

    for i in 0..8 {
        if input2[i] >= 0.0 {
            xval2[i] = input2[i];
        } else {
            xval2[i] = -input2[i];
            nflip2 += 1;
            s2 |= 1 << i;
        }
    }

    println!("  nflip = {}", nflip2);
    println!("  s (before parity fix) = {:07b}", s2);

    if nflip2 % 2 != 0 {
        let mut min_idx = 0;
        let mut min_val = weight[0] * input2[0] * input2[0];
        for i in 1..8 {
            let ax = weight[i] * input2[i] * input2[i];
            if ax < min_val {
                min_val = ax;
                min_idx = i;
            }
        }
        xval2[min_idx] = -xval2[min_idx];
        s2 ^= 1 << min_idx;
        println!("  Flipped bit {} to fix parity", min_idx);
    }

    let stored_index2 = s2 & 127;
    println!(
        "  Stored 7-bit index = {} ({:07b})",
        stored_index2, stored_index2
    );

    let signs2 = ksigns_iq2xs[stored_index2 as usize];
    println!("  ksigns_iq2xs[{}] = {} ({:08b})", stored_index2, signs2, signs2);

    print!("  Dequantized signs: ");
    for j in 0..8 {
        let is_neg = (signs2 & kmask_iq2xs[j]) != 0;
        let sign = if is_neg { -1.0f32 } else { 1.0f32 };
        print!("{} ", sign);
    }
    println!();

    // Test case 3: Two negative values (even parity)
    println!("\nTest 3: Two negatives [1, -1, -1, 1, 1, 1, 1, 1]");
    let input3 = [1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let mut nflip3 = 0;
    let mut s3: u8 = 0;
    let mut xval3 = [0.0f32; 8];

    for i in 0..8 {
        if input3[i] >= 0.0 {
            xval3[i] = input3[i];
        } else {
            xval3[i] = -input3[i];
            nflip3 += 1;
            s3 |= 1 << i;
        }
    }

    println!("  nflip = {}", nflip3);
    println!("  s (before parity fix) = {:07b}", s3);

    if nflip3 % 2 != 0 {
        let mut min_idx = 0;
        let mut min_val = weight[0] * input3[0] * input3[0];
        for i in 1..8 {
            let ax = weight[i] * input3[i] * input3[i];
            if ax < min_val {
                min_val = ax;
                min_idx = i;
            }
        }
        xval3[min_idx] = -xval3[min_idx];
        s3 ^= 1 << min_idx;
        println!("  Flipped bit {} to fix parity", min_idx);
    }

    let stored_index3 = s3 & 127;
    println!(
        "  Stored 7-bit index = {} ({:07b})",
        stored_index3, stored_index3
    );

    let signs3 = ksigns_iq2xs[stored_index3 as usize];
    println!("  ksigns_iq2xs[{}] = {} ({:08b})", stored_index3, signs3, signs3);

    print!("  Dequantized signs: ");
    for j in 0..8 {
        let is_neg = (signs3 & kmask_iq2xs[j]) != 0;
        let sign = if is_neg { -1.0f32 } else { 1.0f32 };
        print!("{} ", sign);
    }
    println!();

    // Verify ksigns table logic
    println!("\n=== ksigns table verification ===");
    println!("Index | Parity | Output | Meaning");
    println!("------|--------|--------|--------");
    for i in 0..16 {
        let parity = (0..7).filter(|&b| (i >> b) & 1 == 1).count() % 2;
        let output = ksigns_iq2xs[i];
        let parity_str = if parity == 0 { "even" } else { "odd" };
        println!(
            "  {:3} | {:4}   | {:3}    | {:08b}",
            i, parity_str, output, output
        );
    }

    // Key insight: For odd parity input, output has bit 7 set + same bits 0-6
    // For even parity input, output is just the input value
    println!("\nKey insight:");
    println!(
        "  Odd parity input (nflip odd): ksigns preserves bits 0-6, adds bit 7"
    );
    println!(
        "  Even parity input (nflip even): ksigns is just the input value"
    );
}
