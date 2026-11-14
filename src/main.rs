// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use clap::Parser;
use infer_train::inference::SimpleModel;
use infer_train::model::GgufFile;
use infer_train::tokenizer::{
    ChatMessage, ChatTemplate, Tokenizer, create_tokenizer,
};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(name = "infer_train")]
#[command(about = "Inference engine CLI")]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: String,

    /// Prompt to generate (if not provided, enters interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// System prompt
    #[arg(short, long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Maximum tokens to generate
    #[arg(short, long, default_value = "128")]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(short, long, default_value = "0.7")]
    temperature: f32,

    /// Show tensor summary
    #[arg(long)]
    show_tensors: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    println!("infer_train - unified inference engine");
    println!("Loading model: {}", args.model);

    // Load GGUF file for tokenizer
    let gguf = GgufFile::load(&args.model)?;
    print_model_info(&gguf);

    // Create tokenizer
    let meta = gguf.tokenizer_metadata()?;
    let tokenizer = create_tokenizer(meta).map_err(|e| e.to_string())?;

    // Load model with backend
    println!("\n=== Loading Model with Backend ===");
    let model = SimpleModel::load(&args.model, tokenizer)?;

    // Print model config
    let config = model.config();
    println!("Layers: {}", config.n_layers);
    println!("Hidden size: {}", config.hidden_size);
    println!("Heads: {} (KV: {})", config.n_heads, config.n_kv_heads);
    println!("Head dim: {}", config.head_dim());
    println!("Vocab size: {}", config.vocab_size);
    println!("Intermediate: {}", config.intermediate_size);
    println!("Context length: {}", config.context_length);

    // Show tensor summary if requested
    if args.show_tensors {
        model.print_tensor_summary();
    }

    // Get chat template
    let chat_template_str = gguf.kv.iter().find_map(|(key, value)| {
        if key == "tokenizer.chat_template" {
            if let infer_train::model::GgufValue::String(s) = value {
                return Some(s.clone());
            }
        }
        None
    });

    let chat_template = if let Some(template_str) = chat_template_str {
        println!("Chat template found ({} chars)", template_str.len());
        Some(ChatTemplate::new(&template_str)?)
    } else {
        println!("No chat template found in model");
        None
    };

    // Run inference loop
    if let Some(prompt) = args.prompt {
        run_single_prompt(
            &model,
            chat_template.as_ref(),
            &args.system,
            &prompt,
            args.max_tokens,
            args.temperature,
        )?;
    } else {
        run_interactive(
            &model,
            chat_template.as_ref(),
            &args.system,
            args.max_tokens,
            args.temperature,
        )?;
    }

    Ok(())
}

fn print_model_info(gguf: &GgufFile) {
    println!("\n=== Model Info ===");
    println!("GGUF version: {}", gguf.version);
    println!("Number of tensors: {}", gguf.tensors.len());
    println!("Number of KV pairs: {}", gguf.kv.len());

    // Print some key metadata
    let keys_to_show = [
        "general.name",
        "general.architecture",
        "general.quantization_type",
        "general.file_type",
    ];

    for key in &keys_to_show {
        if let Some(value) = gguf.kv.get(*key) {
            println!("{}: {:?}", key, value);
        }
    }

    // Count tensor types
    let mut quant_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for tensor in &gguf.tensors {
        let qname = format!("{:?}", tensor.quant_type);
        *quant_counts.entry(qname).or_insert(0) += 1;
    }

    println!("\nQuantization types:");
    for (qname, count) in &quant_counts {
        println!("  {}: {} tensors", qname, count);
    }
}

fn run_single_prompt(
    model: &SimpleModel,
    chat_template: Option<&ChatTemplate>,
    system: &str,
    prompt: &str,
    _max_tokens: usize,
    _temperature: f32,
) -> Result<(), String> {
    println!("\n=== Processing Prompt ===");
    println!("System: {}", system);
    println!("User: {}", prompt);

    // Create messages
    let messages = vec![
        ChatMessage { role: "system".to_string(), content: system.to_string() },
        ChatMessage { role: "user".to_string(), content: prompt.to_string() },
    ];

    // Apply chat template
    let prompt_text = if let Some(ct) = chat_template {
        let rendered = ct.apply(&messages)?;
        println!("\n=== Chat Template Result ===");
        for (i, line) in rendered.lines().enumerate() {
            println!("{:3}: {}", i, line);
        }
        rendered
    } else {
        format!("System: {}\nUser: {}\nAssistant:", system, prompt)
    };

    // Tokenize
    println!("\n=== Tokenization ===");
    let tokenizer = model.tokenizer();
    let tokens = tokenizer.encode(&prompt_text).map_err(|e| e.to_string())?;
    println!(
        "Prompt tokens: {} (first 20: {:?})",
        tokens.len(),
        &tokens[..tokens.len().min(20)]
    );

    // Decode to verify
    let decoded =
        tokenizer.decode_skip_special(&tokens).map_err(|e| e.to_string())?;
    println!("Decoded length: {} chars", decoded.len());

    if prompt_text == decoded {
        println!("✓ Perfect encode/decode match");
    } else {
        println!("⚠ Encode/decode mismatch");
    }

    // Show embedding lookup
    println!("\n=== Embedding Layer ===");
    if let Some(_emb) = model.get_embeddings() {
        println!("✓ Embedding weights found");
        // Test embedding lookup for first token
        if let Some(_hidden) = model.embed_token(tokens[0]) {
            println!(
                "✓ Embedding lookup works (hidden_size: {})",
                model.config().hidden_size
            );
        }
    } else {
        println!("✗ No embedding weights found");
    }

    // Show output layer
    println!("\n=== Output Layer ===");
    if let Some(_output) = model.get_output_weight() {
        println!("✓ Output weights found");
    } else {
        println!("✗ No output weights found");
    }

    println!("\n=== Inference Status ===");
    println!("Model loaded and ready for inference.");
    println!("Full forward pass requires additional implementation.");

    Ok(())
}

fn run_interactive(
    model: &SimpleModel,
    chat_template: Option<&ChatTemplate>,
    system: &str,
    _max_tokens: usize,
    _temperature: f32,
) -> Result<(), String> {
    println!("\n=== Interactive Mode ===");
    println!("System: {}", system);
    println!(
        "Type 'quit' to exit, 'clear' to clear history, 'info' for model info.\n"
    );

    let mut history: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: system.to_string(),
    }];

    let tokenizer = model.tokenizer();

    loop {
        print!("User: ");
        io::stdout().flush().map_err(|e| e.to_string())?;

        let mut input = String::new();
        io::stdin().read_line(&mut input).map_err(|e| e.to_string())?;
        let input = input.trim();

        if input == "quit" {
            break;
        }
        if input == "clear" {
            history = vec![ChatMessage {
                role: "system".to_string(),
                content: system.to_string(),
            }];
            println!("History cleared.\n");
            continue;
        }
        if input == "info" {
            println!("Model config: {:?}", model.config());
            model.print_tensor_summary();
            continue;
        }

        if input.is_empty() {
            continue;
        }

        // Add to history
        history.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // Apply chat template
        let prompt_text = if let Some(ct) = chat_template {
            ct.apply(&history)?
        } else {
            let mut s = String::new();
            for msg in &history {
                s.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
            s.push_str("Assistant:");
            s
        };

        // Tokenize
        let tokens =
            tokenizer.encode(&prompt_text).map_err(|e| e.to_string())?;
        println!("Tokens: {}", tokens.len());

        // TODO: Run actual inference here
        println!("Assistant: [Inference not implemented yet]\n");

        // Add placeholder response to history
        history.push(ChatMessage {
            role: "assistant".to_string(),
            content: "[placeholder]".to_string(),
        });
    }

    println!("Goodbye!");
    Ok(())
}
