// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Chat Template test with real GGUF file

use infer_train::model::GgufFile;
use infer_train::tokenizer::{ChatMessage, ChatTemplate};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 {
        args[1].as_str()
    } else {
        "Qwen3-0.6B-UD-Q4_K_XL.gguf"
    };

    println!("Loading {}...", path);

    let gguf = GgufFile::load(path).expect("Failed to load GGUF");

    // Extract chat template
    let chat_template_str = gguf
        .kv
        .iter()
        .find(|(key, _)| key.to_lowercase().contains("chat_template"))
        .and_then(|(_, value)| {
            if let infer_train::model::GgufValue::String(s) = value {
                Some(s.clone())
            } else {
                None
            }
        });

    let Some(template_str) = chat_template_str else {
        println!("No chat template found in GGUF!");
        return;
    };

    println!("Found chat template ({} chars)", template_str.len());

    // Create chat template
    let chat_template =
        ChatTemplate::new(&template_str).expect("Failed to parse template");

    // Test messages
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        ChatMessage { role: "user".to_string(), content: "Hello!".to_string() },
    ];

    println!("\nApplying template to messages:");
    for msg in &messages {
        println!("  {}: {}", msg.role, msg.content);
    }

    let result = match chat_template.apply(&messages) {
        Ok(r) => r,
        Err(e) => {
            // Print template lines around the error
            if let Some(line_num_str) = e.split("chat:").nth(1) {
                if let Ok(line_num) =
                    line_num_str.trim_end_matches(')').parse::<usize>()
                {
                    println!("\nTemplate around line {}:", line_num);
                    for (i, line) in template_str.lines().enumerate() {
                        if i >= line_num.saturating_sub(5) && i <= line_num + 5
                        {
                            println!("{:4}: {}", i + 1, line);
                        }
                    }
                }
            }
            panic!("Failed to apply template: {:?}", e);
        }
    };

    println!("\nResult:");
    for (i, line) in result.lines().enumerate() {
        println!("  {}: {}", i, line);
    }

    println!("\n=== Chat Template test passed! ===");
    println!(
        "Note: Complex Jinja2 features like 'strip' may not be fully supported."
    );
}
