// Verify all K-quant block layouts
use infer_train::types::{
    BlockIQ4XS, BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_K,
};

fn main() {
    // Q4_K
    println!("=== BlockQ4_K ===");
    println!("Size: {} bytes", std::mem::size_of::<BlockQ4_K>());
    let test_block: BlockQ4_K = unsafe { std::mem::zeroed() };
    let base_ptr = &test_block as *const BlockQ4_K as usize;
    let d_offset = &test_block.d as *const _ as usize - base_ptr;
    let dmin_offset = &test_block.dmin as *const _ as usize - base_ptr;
    let scales_offset = &test_block.scales as *const _ as usize - base_ptr;
    let qs_offset = &test_block.qs as *const _ as usize - base_ptr;
    println!(
        "  d: offset {}, dmin: offset {}, scales: offset {}, qs: offset {}",
        d_offset, dmin_offset, scales_offset, qs_offset
    );
    // Expected: d=0, dmin=2, scales=4, qs=16

    // Q5_K
    println!("\n=== BlockQ5_K ===");
    println!("Size: {} bytes", std::mem::size_of::<BlockQ5_K>());
    let test_block: BlockQ5_K = unsafe { std::mem::zeroed() };
    let base_ptr = &test_block as *const BlockQ5_K as usize;
    let d_offset = &test_block.d as *const _ as usize - base_ptr;
    let dmin_offset = &test_block.dmin as *const _ as usize - base_ptr;
    let scales_offset = &test_block.scales as *const _ as usize - base_ptr;
    let qh_offset = &test_block.qh as *const _ as usize - base_ptr;
    let qs_offset = &test_block.qs as *const _ as usize - base_ptr;
    println!(
        "  d: offset {}, dmin: offset {}, scales: offset {}, qh: offset {}, qs: offset {}",
        d_offset, dmin_offset, scales_offset, qh_offset, qs_offset
    );
    // Expected: d=0, dmin=2, scales=4, qh=16, qs=48

    // Q6_K
    println!("\n=== BlockQ6_K ===");
    println!("Size: {} bytes", std::mem::size_of::<BlockQ6_K>());
    let test_block: BlockQ6_K = unsafe { std::mem::zeroed() };
    let base_ptr = &test_block as *const BlockQ6_K as usize;
    let d_offset = &test_block.d as *const _ as usize - base_ptr;
    let scales_offset = &test_block.scales as *const _ as usize - base_ptr;
    let ql_offset = &test_block.ql as *const _ as usize - base_ptr;
    let qh_offset = &test_block.qh as *const _ as usize - base_ptr;
    println!(
        "  d: offset {}, scales: offset {}, ql: offset {}, qh: offset {}",
        d_offset, scales_offset, ql_offset, qh_offset
    );
    // Expected: d=0, scales=2, ql=18, qh=146

    // Q8_K
    println!("\n=== BlockQ8_K ===");
    println!("Size: {} bytes", std::mem::size_of::<BlockQ8_K>());
    let test_block: BlockQ8_K = unsafe { std::mem::zeroed() };
    let base_ptr = &test_block as *const BlockQ8_K as usize;
    let d_offset = &test_block.d as *const _ as usize - base_ptr;
    let qs_offset = &test_block.qs as *const _ as usize - base_ptr;
    let bsums_offset = &test_block.bsums as *const _ as usize - base_ptr;
    println!(
        "  d: offset {}, qs: offset {}, bsums: offset {}",
        d_offset, qs_offset, bsums_offset
    );
    // Expected: d=0, qs=4, bsums=260

    // IQ4_XS
    println!("\n=== BlockIQ4XS ===");
    println!("Size: {} bytes", std::mem::size_of::<BlockIQ4XS>());
    let test_block: BlockIQ4XS = unsafe { std::mem::zeroed() };
    let base_ptr = &test_block as *const BlockIQ4XS as usize;
    let d_offset = &test_block.d as *const _ as usize - base_ptr;
    let scales_h_offset = &test_block.scales_h as *const _ as usize - base_ptr;
    let scales_l_offset = &test_block.scales_l as *const _ as usize - base_ptr;
    let qs_offset = &test_block.qs as *const _ as usize - base_ptr;
    println!(
        "  d: offset {}, scales_h: offset {}, scales_l: offset {}, qs: offset {}",
        d_offset, scales_h_offset, scales_l_offset, qs_offset
    );
    // Expected: d=0, scales_h=2, scales_l=4, qs=8
}
