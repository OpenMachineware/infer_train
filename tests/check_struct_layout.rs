// Check Rust struct layout
use std::mem;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],
    pub qs: [u8; 64],
    pub scales: [u8; 12],
    pub d: u16,
}

fn main() {
    println!("Size: {}", mem::size_of::<BlockQ3K>());
    println!("hmask offset: {}", mem::offset_of!(BlockQ3K, hmask));
    println!("qs offset: {}", mem::offset_of!(BlockQ3K, qs));
    println!("scales offset: {}", mem::offset_of!(BlockQ3K, scales));
    println!("d offset: {}", mem::offset_of!(BlockQ3K, d));
}
