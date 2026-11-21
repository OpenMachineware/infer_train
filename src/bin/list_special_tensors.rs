use infer_train::model::GgufFile;

fn main() {
    let path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(path).unwrap();

    println!("=== Special Tensors (not starting with blk.) ===\n");

    for tensor in &gguf.tensors {
        if !tensor.name.starts_with("blk.") {
            println!(
                "{}: dims={:?}, strides={:?}, quant={:?}",
                tensor.name, tensor.dims, tensor.strides, tensor.quant_type
            );
        }
    }
}
