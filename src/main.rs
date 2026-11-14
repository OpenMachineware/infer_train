// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use clap::{Parser, Subcommand};
use infer_train::inference::{ForwardExecutor, SimpleModel};
use infer_train::model::GgufFile;
use infer_train::tokenizer::{ChatMessage, ChatTemplate, create_tokenizer};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(name = "infer_train")]
#[command(about = "Unified inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// CLI inference mode
    Cli {
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
        #[arg(long, default_value = "128")]
        max_tokens: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Show tensor summary
        #[arg(long)]
        show_tensors: bool,
    },
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Cli {
            model,
            prompt,
            system,
            max_tokens,
            temperature,
            show_tensors,
        } => run_cli(
            &model,
            prompt.as_deref(),
            &system,
            max_tokens,
            temperature,
            show_tensors,
        ),
    }
}

fn run_cli(
    model_path: &str,
    prompt: Option<&str>,
    system: &str,
    max_tokens: usize,
    temperature: f32,
    show_tensors: bool,
) -> Result<(), String> {
    println!("infer_train - unified inference engine");
    println!("Loading model: {}", model_path);

    // Load GGUF file for tokenizer
    let gguf = GgufFile::load(model_path)?;
    print_model_info(&gguf);

    // Create tokenizer
    let meta = gguf.tokenizer_metadata()?;
    let tokenizer = create_tokenizer(meta).map_err(|e| e.to_string())?;

    // Load model with backend
    println!("\n=== Loading Model with Backend ===");
    let model = SimpleModel::load(model_path, tokenizer)?;

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
    if show_tensors {
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
    if let Some(prompt) = prompt {
        run_single_prompt(
            model,
            chat_template.as_ref(),
            system,
            prompt,
            max_tokens,
            temperature,
        )?;
    } else {
        run_interactive(
            model,
            chat_template.as_ref(),
            system,
            max_tokens,
            temperature,
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
    model: SimpleModel,
    chat_template: Option<&ChatTemplate>,
    system: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
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
        ct.apply(&messages)?
    } else {
        format!("System: {}\nUser: {}\nAssistant:", system, prompt)
    };

    // Create forward executor first (takes ownership of model)
    let mut executor = ForwardExecutor::new(model);

    // Get tokenizer and tokenize
    let tokens = {
        let tokenizer = executor.model.tokenizer();
        let tokens =
            tokenizer.encode(&prompt_text).map_err(|e| e.to_string())?;
        println!("\nPrompt tokens: {}", tokens.len());
        tokens
    };

    // Generate (mutable borrow)
    println!("\n=== Generating ===");
    let output_tokens = executor.generate(&tokens, max_tokens, temperature);

    // Decode after mutable borrow ends
    let new_tokens = &output_tokens[tokens.len()..];
    let new_text = {
        let tokenizer = executor.model.tokenizer();
        tokenizer.decode(new_tokens).map_err(|e| e.to_string())?
    };

    println!("\nAssistant: {}", new_text);

    Ok(())
}

fn run_interactive(
    model: SimpleModel,
    chat_template: Option<&ChatTemplate>,
    system: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<(), String> {
    println!("\n=== Interactive Mode ===");
    println!("System: {}", system);
    println!(
        "Type 'quit' to exit, 'clear' to clear history, 'info' for model info.\n"
    );

    let mut executor = ForwardExecutor::new(model);

    let mut history: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: system.to_string(),
    }];

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
            // Reset executor (clears KV cache)
            executor.seq_pos = 0;
            for layer in 0..executor.model.config().n_layers {
                executor.kv_cache_k[layer].fill(0.0);
                executor.kv_cache_v[layer].fill(0.0);
            }
            println!("History cleared.\n");
            continue;
        }
        if input == "info" {
            println!("Model config: {:?}", executor.model.config());
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
        let tokens = {
            let tokenizer = executor.model.tokenizer();
            tokenizer.encode(&prompt_text).map_err(|e| e.to_string())?
        };

        // Generate
        let output_tokens = executor.generate(&tokens, max_tokens, temperature);

        // Decode new tokens
        let new_tokens = &output_tokens[tokens.len()..];
        let response = {
            let tokenizer = executor.model.tokenizer();
            tokenizer.decode(new_tokens).map_err(|e| e.to_string())?
        };

        println!("Assistant: {}\n", response);

        // Add response to history
        history.push(ChatMessage {
            role: "assistant".to_string(),
            content: response,
        });
    }

    println!("Goodbye!");
    Ok(())
}
