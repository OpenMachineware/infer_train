// List metadata in a GGUF file
use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        std::process::exit(1);
    }

    let gguf = GgufFile::open(&args[1]).unwrap();

    for (key, value) in &gguf.metadata {
        println!("{}: {:?}", key, value);
    }
}
