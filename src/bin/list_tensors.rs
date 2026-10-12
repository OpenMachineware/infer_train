// List all tensor names in a GGUF file
use infer_train::model::WeightStore;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        std::process::exit(1);
    }

    let store = WeightStore::from_gguf(&args[1]).unwrap();

    for name in store.tensor_names() {
        println!("{}", name);
    }
}
