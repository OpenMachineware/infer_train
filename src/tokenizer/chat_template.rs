// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Chat Template implementation using Jinja2 templates

use minijinja::{Environment, Error, ErrorKind, State, Value, context};
use serde::{Deserialize, Serialize};

/// A single chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

/// Handle unknown method calls for Python string compatibility
fn handle_unknown_method(
    _state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    // Only handle string methods
    let s = match value.as_str() {
        Some(s) => s,
        None => return Err(Error::from(ErrorKind::UnknownMethod)),
    };

    match method {
        "startswith" => {
            if args.len() != 1 {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "startswith() takes exactly 1 argument",
                ));
            }
            let prefix = args[0].as_str().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "startswith() requires a string argument",
                )
            })?;
            Ok(Value::from(s.starts_with(prefix)))
        }
        "endswith" => {
            if args.len() != 1 {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "endswith() takes exactly 1 argument",
                ));
            }
            let suffix = args[0].as_str().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "endswith() requires a string argument",
                )
            })?;
            Ok(Value::from(s.ends_with(suffix)))
        }
        "strip" => {
            if !args.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "strip() takes no arguments",
                ));
            }
            Ok(Value::from(s.trim()))
        }
        "lstrip" => {
            if !args.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "lstrip() takes no arguments",
                ));
            }
            Ok(Value::from(s.trim_start()))
        }
        "rstrip" => {
            if !args.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "rstrip() takes no arguments",
                ));
            }
            Ok(Value::from(s.trim_end()))
        }
        "lower" => {
            if !args.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "lower() takes no arguments",
                ));
            }
            Ok(Value::from(s.to_lowercase()))
        }
        "upper" => {
            if !args.is_empty() {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "upper() takes no arguments",
                ));
            }
            Ok(Value::from(s.to_uppercase()))
        }
        "replace" => {
            if args.len() != 2 {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "replace() takes exactly 2 arguments",
                ));
            }
            let old = args[0].as_str().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "replace() requires string arguments",
                )
            })?;
            let new = args[1].as_str().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "replace() requires string arguments",
                )
            })?;
            Ok(Value::from(s.replace(old, new)))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
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

        // Add Python string method compatibility
        env.set_unknown_method_callback(handle_unknown_method);

        // Add custom functions for Jinja2 compatibility (function call syntax)
        env.add_function("startswith", |s: String, prefix: String| -> bool {
            s.starts_with(&prefix)
        });
        env.add_function("endswith", |s: String, suffix: String| -> bool {
            s.ends_with(&suffix)
        });
        env.add_function("strip", |s: String| -> String {
            s.trim().to_string()
        });

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

        // Add Python string method compatibility
        env.set_unknown_method_callback(handle_unknown_method);

        // Add custom functions for Jinja2 compatibility (function call syntax)
        env.add_function("startswith", |s: String, prefix: String| -> bool {
            s.starts_with(&prefix)
        });
        env.add_function("endswith", |s: String, suffix: String| -> bool {
            s.ends_with(&suffix)
        });
        env.add_function("strip", |s: String| -> String {
            s.trim().to_string()
        });

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

    #[test]
    fn test_startswith_function() {
        let template =
            r#"{% if startswith('hello', 'he') %}yes{% else %}no{% endif %}"#;
        let ct = ChatTemplate::new(template).unwrap();
        let result = ct.apply(&[]).unwrap();
        assert_eq!(result, "yes");
    }

    #[test]
    fn test_startswith_method() {
        // Test Python method call syntax
        let template =
            r#"{% if 'hello'.startswith('he') %}yes{% else %}no{% endif %}"#;
        let ct = ChatTemplate::new(template).unwrap();
        let result = ct.apply(&[]).unwrap();
        assert_eq!(result, "yes");
    }

    #[test]
    fn test_endswith_method() {
        let template =
            r#"{% if 'hello'.endswith('lo') %}yes{% else %}no{% endif %}"#;
        let ct = ChatTemplate::new(template).unwrap();
        let result = ct.apply(&[]).unwrap();
        assert_eq!(result, "yes");
    }

    #[test]
    fn test_strip_method() {
        let template = r#"{{ '  hello  '.strip() }}"#;
        let ct = ChatTemplate::new(template).unwrap();
        let result = ct.apply(&[]).unwrap();
        assert_eq!(result, "hello");
    }
}
