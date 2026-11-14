// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Chat Template implementation using Jinja2 templates

use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};

/// A single chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

/// Chat template renderer
pub struct ChatTemplate {
    /// Jinja template string (owned)
    template: String,
}

impl ChatTemplate {
    /// Create a new chat template from a Jinja template string
    pub fn new(template: &str) -> Result<Self, String> {
        // Validate the template by parsing it
        let mut env = Environment::new();
        env.add_template("chat", template)
            .map_err(|e| format!("Failed to parse template: {}", e))?;

        Ok(Self { template: template.to_string() })
    }

    /// Apply the chat template to a list of messages
    pub fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        self.apply_with_options(messages, false)
    }

    /// Apply the chat template with options
    pub fn apply_with_options(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let mut env = Environment::new();
        env.add_template("chat", &self.template)
            .map_err(|e| format!("Failed to add template: {}", e))?;

        let template = env
            .get_template("chat")
            .map_err(|e| format!("Template not found: {}", e))?;

        let bos_token = "";
        let eos_token = "";

        let result = template
            .render(context! {
                messages,
                add_generation_prompt,
                bos_token,
                eos_token,
            })
            .map_err(|e| format!("Template rendering failed: {}", e))?;

        Ok(result)
    }

    /// Get the raw template string
    pub fn template(&self) -> &str {
        &self.template
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_template() {
        // Simple Qwen-style template
        let template = r#"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"#;

        let ct = ChatTemplate::new(template).unwrap();
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
        ];

        let result = ct.apply(&messages).unwrap();
        assert!(result.contains("Hello"));
        assert!(result.contains("Hi there!"));
    }
}
