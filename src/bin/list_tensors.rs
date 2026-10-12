// List tensor names in a GGUF file
use infer_train::model::gguf::GgufFile;
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file.gguf>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let gguf = match GgufFile::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== Tensors in {} ===\n", path);
    println!("Total tensors: {}\n", gguf.tensors.len());

    // Group by layer
    let mut layers: HashSet<String> = HashSet::new();
    for tensor in &gguf.tensors {
        let name = &tensor.name;
        if let Some(idx) = name.find('.') {
            let prefix = &name[..idx];
            if prefix.starts_with("blk") || prefix.starts_with("layers") {
                if let Some(next_dot) = name[idx + 1..].find('.') {
                    let layer_key = format!(
                        "{}{}",
                        &name[..idx + 1],
                        &name[idx + 1..idx + 1 + next_dot]
                    );
                    layers.insert(layer_key);
                }
            }
        }
    }

    // Sort layers
    let mut layer_list: Vec<_> = layers.into_iter().collect();
    layer_list.sort();

    // Show first 3 layers' tensors
    for layer_prefix in layer_list.iter().take(5) {
        println!("--- {} ---", layer_prefix);
        for tensor in &gguf.tensors {
            if tensor.name.starts_with(layer_prefix) {
                println!("  {}", tensor.name);
            }
        }
        println!();
    }

    // Show non-layer tensors
    println!("--- Non-layer tensors ---");
    for tensor in &gguf.tensors {
        if !tensor.name.starts_with("blk.")
            && !tensor.name.starts_with("layers.")
        {
            println!("  {}", tensor.name);
        }
    }
}
