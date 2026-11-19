// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// Tokenization tool - tokenizer utilities and testing
//
// Usage:
//   tokenize_compare -m model.gguf info
//   tokenize_compare -m model.gguf encode -t "Hello"
//   tokenize_compare -m model.gguf decode -t 9707,198,872
//   tokenize_compare -m model.gguf roundtrip
//   tokenize_compare -m model.gguf chat -u "Hello" -s "You are helpful"
//   tokenize_compare -m model.gguf compare

use clap::{Parser, Subcommand};
use infer_train::model::GgufFile;
use infer_train::tokenizer::{ChatMessage, ChatTemplate, create_tokenizer};

#[derive(Parser)]
#[command(name = "tokenize_compare")]
#[command(
    about = "Tokenization tool - tokenizer utilities and testing",
    version
)]
struct Cli {
    /// GGUF model path
    #[arg(short, long)]
    model: String,

    #[command(subcommand)]
    mode: TokenizeMode,
}

#[derive(Subcommand)]
enum TokenizeMode {
    /// Show tokenizer metadata
    Info,

    /// Encode text to tokens
    Encode {
        /// Text to encode
        #[arg(short, long)]
        text: String,
    },

    /// Decode tokens to text
    Decode {
        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: String,

        /// Skip special tokens in output
        #[arg(long)]
        skip_special: bool,
    },

    /// Round-trip test (encode then decode)
    Roundtrip {
        /// Text to test (optional, uses built-in test strings if not provided)
        #[arg(short, long)]
        text: Option<String>,
    },

    /// Apply chat template
    Chat {
        /// User message
        #[arg(short, long)]
        user: String,

        /// System message
        #[arg(short, long, default_value = "You are a helpful assistant.")]
        system: String,

        /// Add generation prompt
        #[arg(long, default_value = "true")]
        add_generation: bool,
    },

    /// Run all comparison tests
    Compare,
}

fn main() {
    let cli = Cli::parse();

    let gguf = GgufFile::load(&cli.model).expect("Failed to load model");
    let meta =
        gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer =
        create_tokenizer(meta.clone()).expect("Failed to create tokenizer");

    match cli.mode {
        TokenizeMode::Info => show_info(&gguf, &meta),
        TokenizeMode::Encode { text } => encode_text(&*tokenizer, &text),
        TokenizeMode::Decode { tokens, skip_special } => {
            decode_tokens(&*tokenizer, &tokens, skip_special)
        }
        TokenizeMode::Roundtrip { text } => roundtrip_test(&*tokenizer, text),
        TokenizeMode::Chat { user, system, add_generation } => {
            apply_chat_template(
                &gguf,
                &*tokenizer,
                &system,
                &user,
                add_generation,
            )
        }
        TokenizeMode::Compare => run_comparison_tests(&gguf, &*tokenizer),
    }
}

fn show_info(gguf: &GgufFile, meta: &infer_train::model::TokenizerMetadata) {
    println!("=== Tokenizer Info ===");
    println!("Tokenizer type: {:?}", meta.tokenizer_type);
    println!("Vocab size: {}", meta.tokens.len());
    println!("Number of merges: {}", meta.merges.len());
    println!("BOS token ID: {:?}", meta.bos_token_id);
    println!("EOS token ID: {:?}", meta.eos_token_id);
    println!("UNK token ID: {:?}", meta.unk_token_id);
    println!("Add BOS: {:?}", meta.add_bos);
    println!("Add EOS: {:?}", meta.add_eos);
    println!("Byte fallback: {:?}", meta.byte_fallback);
    println!("Pre: {:?}", meta.pre);

    // Show key tokens
    println!("\n=== Key Tokens ===");
    for id in [meta.bos_token_id, meta.eos_token_id, meta.unk_token_id] {
        if let Some(id) = id {
            if let Some(token) = meta.tokens.get(id as usize) {
                println!("  Token {}: '{}'", id, token);
            }
        }
    }

    // Show chat template
    let chat_template = gguf.kv.iter().find_map(|(key, value)| {
        if key.to_lowercase().contains("chat_template") {
            if let infer_train::model::GgufValue::String(s) = value {
                return Some(s.clone());
            }
        }
        None
    });

    if let Some(template) = chat_template {
        println!("\n=== Chat Template ({} chars) ===", template.len());
        for (i, line) in template.lines().enumerate() {
            println!("{:3}: {}", i + 1, line);
        }
    } else {
        println!("\nNo chat template found");
    }
}

fn encode_text(tokenizer: &dyn infer_train::tokenizer::Tokenizer, text: &str) {
    let tokens = tokenizer.encode(text).expect("Failed to encode");
    println!("Text: '{}'", text);
    println!("Tokens ({}): {:?}", tokens.len(), tokens);

    // Show token strings
    let mut token_strs = Vec::new();
    for &id in &tokens {
        let s =
            tokenizer.decode(&[id]).unwrap_or_else(|_| format!("token_{}", id));
        token_strs.push(format!("{}:'{}'", id, s));
    }
    println!("Token strings: {}", token_strs.join(", "));
}

fn decode_tokens(
    tokenizer: &dyn infer_train::tokenizer::Tokenizer,
    tokens_str: &str,
    skip_special: bool,
) {
    let tokens: Vec<u32> =
        tokens_str.split(',').filter_map(|t| t.trim().parse().ok()).collect();

    println!("Tokens: {:?}", tokens);

    let text = if skip_special {
        tokenizer.decode_skip_special(&tokens).expect("Failed to decode")
    } else {
        tokenizer.decode(&tokens).expect("Failed to decode")
    };

    println!("Text: '{}'", text);
}

fn roundtrip_test(
    tokenizer: &dyn infer_train::tokenizer::Tokenizer,
    text: Option<String>,
) {
    let test_texts = if let Some(t) = text {
        vec![t]
    } else {
        vec![
            "Hello".to_string(),
            "Hello, world!".to_string(),
            "Hello World".to_string(),
            "This is a longer sentence for testing.".to_string(),
        ]
    };

    println!("=== Round-trip Test ===");
    let mut all_passed = true;

    for text in &test_texts {
        let tokens = tokenizer.encode(text).expect("Failed to encode");
        let decoded = tokenizer.decode(&tokens).expect("Failed to decode");

        let status = if text == &decoded { "✓" } else { "✗" };
        println!("{} '{}' -> {:?} -> '{}'", status, text, tokens, decoded);

        if text != &decoded {
            all_passed = false;
        }
    }

    if all_passed {
        println!("\n✓ All tests passed!");
    } else {
        println!("\n✗ Some tests failed!");
        std::process::exit(1);
    }
}

fn apply_chat_template(
    gguf: &GgufFile,
    tokenizer: &dyn infer_train::tokenizer::Tokenizer,
    system: &str,
    user: &str,
    add_generation: bool,
) {
    let chat_template_str = gguf.kv.iter().find_map(|(key, value)| {
        if key.to_lowercase().contains("chat_template") {
            if let infer_train::model::GgufValue::String(s) = value {
                return Some(s.clone());
            }
        }
        None
    });

    let Some(template_str) = chat_template_str else {
        eprintln!("No chat template found in model");
        std::process::exit(1);
    };

    let chat_template =
        ChatTemplate::new(&template_str).expect("Failed to parse template");

    let messages = vec![
        ChatMessage { role: "system".to_string(), content: system.to_string() },
        ChatMessage { role: "user".to_string(), content: user.to_string() },
    ];

    let result = if add_generation {
        chat_template
            .apply_with_options(&messages, true)
            .expect("Failed to apply")
    } else {
        chat_template.apply(&messages).expect("Failed to apply")
    };

    println!("=== Chat Template Result ===");
    println!("System: {}", system);
    println!("User: {}", user);
    println!("\nPrompt:");
    println!("{}", result);

    // Tokenize
    let tokens = tokenizer.encode(&result).expect("Failed to encode");
    println!("\nTokens ({}): {:?}", tokens.len(), tokens);
}

fn run_comparison_tests(
    gguf: &GgufFile,
    tokenizer: &dyn infer_train::tokenizer::Tokenizer,
) {
    println!("=== Tokenization Comparison Tests ===\n");

    let meta = gguf.tokenizer_metadata().expect("Failed to get metadata");
    println!("Tokenizer type: {:?}", meta.tokenizer_type);
    println!("Vocab size: {}", meta.tokens.len());

    let mut all_passed = true;

    // Test 1: Simple "Hello"
    println!("\n=== Test 1: Simple 'Hello' ===");
    let tokens = tokenizer.encode("Hello").expect("Failed to encode");
    let status = if tokens == vec![9707] { "✓" } else { "✗" };
    println!("{} 'Hello' -> {:?} (expected [9707])", status, tokens);
    if tokens != vec![9707] {
        all_passed = false;
    }

    // Test 2: Special tokens
    println!("\n=== Test 2: Special Tokens ===");
    let special_tests = [("<|im_start|>", 151644), ("<|im_end|>", 151645)];
    for (text, expected_id) in special_tests {
        let tokens = tokenizer.encode(text).expect("Failed to encode");
        let actual_id = tokens.first().copied().unwrap_or(0);
        let status = if actual_id == expected_id { "✓" } else { "✗" };
        println!(
            "{} '{}' -> {} (expected {})",
            status, text, actual_id, expected_id
        );
        if actual_id != expected_id {
            all_passed = false;
        }
    }

    // Test 3: Round-trip
    println!("\n=== Test 3: Round-trip ===");
    let roundtrip_tests =
        ["Hello", "Hello, world!", "<|im_start|>user\nHello<|im_end|>"];
    for text in roundtrip_tests {
        let tokens = tokenizer.encode(text).expect("Failed to encode");
        let decoded = tokenizer.decode(&tokens).expect("Failed to decode");
        let status = if text == decoded { "✓" } else { "✗" };
        println!("{} '{}' -> {:?} -> '{}'", status, text, tokens, decoded);
        if text != decoded {
            all_passed = false;
        }
    }

    // Test 4: Chat template
    println!("\n=== Test 4: Chat Template ===");
    let chat_template_str = gguf.kv.iter().find_map(|(key, value)| {
        if key.to_lowercase().contains("chat_template") {
            if let infer_train::model::GgufValue::String(s) = value {
                return Some(s.clone());
            }
        }
        None
    });

    if let Some(template_str) = chat_template_str {
        let chat_template =
            ChatTemplate::new(&template_str).expect("Failed to parse template");
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        let result = chat_template
            .apply_with_options(&messages, true)
            .expect("Failed to apply");
        println!("Chat template applied successfully ({} chars)", result.len());

        let full_tokens = tokenizer.encode(&result).expect("Failed to encode");
        println!(
            "Full prompt tokens ({}): {:?}",
            full_tokens.len(),
            full_tokens
        );
    } else {
        println!("No chat template found, skipping");
    }

    if all_passed {
        println!("\n✓ All comparison tests passed!");
    } else {
        println!("\n✗ Some tests failed!");
        std::process::exit(1);
    }
}
