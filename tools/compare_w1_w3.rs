// Check w1 vs w3 data
use infer_train::inference::Device;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];
    let ctx = infer_train::inference::InferenceContext::from_gguf(model_path, Device::Cpu).unwrap();

    let layer = &ctx.model.layers[0];
    let w1 = layer.w1.as_ref().unwrap();
    let w3 = layer.w3.as_ref().unwrap();

    println!("w1: shape={:?} dtype={:?} bytes={}", w1.shape, w1.dtype, w1.bytes().len());
    println!("w3: shape={:?} dtype={:?} bytes={}", w3.shape, w3.dtype, w3.bytes().len());

    // Compare first 100 bytes
    let w1_first = &w1.bytes()[..100];
    let w3_first = &w3.bytes()[..100];
    println!("\nw1 first 100 bytes: {:?}", w1_first);
    println!("w3 first 100 bytes: {:?}", w3_first);

    // Check if they're identical (shouldn't be)
    let same = w1.bytes() == w3.bytes();
    println!("\nw1 == w3: {}", same);
}
