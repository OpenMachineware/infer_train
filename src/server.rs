// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

//! OpenAI/Claude compatible API server

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{Response, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use infer_train::model::GgufFile;
use infer_train::ops::sample_token;
use infer_train::tokenize::BpeTokenizer;

#[derive(Clone)]
struct AppState {
    model_path: Arc<String>,
    tokenizer: Arc<BpeTokenizer>,
}

pub async fn run(
    model_path: &str,
    device_str: &str,
    host: &str,
    port: u16,
    _ctx_size: usize,
) {
    let device = match device_str.to_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        _ => Device::Gpu,
    };

    println!("Loading model from: {}", model_path);

    // Load tokenizer
    let gguf = match GgufFile::open(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open GGUF: {}", e);
            return;
        }
    };
    let tokenizer = match BpeTokenizer::from_gguf(&gguf) {
        Some(t) => t,
        None => {
            eprintln!("Failed to load tokenizer");
            return;
        }
    };

    let ctx = match InferenceContext::from_gguf(model_path, device) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    println!(
        "Model loaded: {} layers, {} hidden",
        ctx.model.layers.len(),
        ctx.config.hidden_size
    );

    let state = AppState {
        model_path: Arc::new(model_path.to_string()),
        tokenizer: Arc::new(tokenizer),
    };

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    println!("Server running at http://{}", addr);

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind: {}", e);
            return;
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("Server error: {}", e);
    }
}

// ============ API Types ============

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

#[derive(Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
}

// ============ Handlers ============

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    // Use model path as the model name
    let model_name =
        state.model_path.rsplit('/').next().unwrap_or("unknown").to_string();

    (
        StatusCode::OK,
        Json(ModelList {
            object: "list".to_string(),
            data: vec![ModelInfo {
                id: model_name,
                object: "model".to_string(),
                created: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                owned_by: "infer_train".to_string(),
            }],
        }),
    )
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response<Body> {
    // Build prompt from messages
    let mut prompt = String::new();
    for msg in &req.messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("System: {}\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("User: {}\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("Assistant: {}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
        }
    }
    prompt.push_str("Assistant:");

    // Tokenize
    let tokens = state.tokenizer.encode(&prompt);
    let prompt_tokens = tokens.len();

    // Load model (temporarily - need proper Send/Sync solution)
    let ctx = match InferenceContext::from_gguf(&state.model_path, Device::Gpu)
    {
        Ok(ctx) => ctx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to load model: {}", e),
                }),
            )
                .into_response();
        }
    };

    // Create KV cache for this request
    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Generate
    let mut generated_text = String::new();
    let mut current_tokens = tokens.clone();
    let mut generated = 0;

    while generated < req.max_tokens {
        let logits = match ctx.model.forward(&current_tokens, &mut kv_cache) {
            Some(l) => l,
            None => break,
        };

        let next_token = sample_token(&logits, req.temperature) as u32;
        current_tokens.clear();
        current_tokens.push(next_token);

        let text = state.tokenizer.decode(&[next_token]);
        generated_text.push_str(&text);
        generated += 1;
    }

    let response = ChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: generated_text,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: generated,
            total_tokens: prompt_tokens + generated,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response<Body> {
    // Tokenize
    let tokens = state.tokenizer.encode(&req.prompt);
    let prompt_tokens = tokens.len();

    // Load model (temporarily - need proper Send/Sync solution)
    let ctx = match InferenceContext::from_gguf(&state.model_path, Device::Gpu)
    {
        Ok(ctx) => ctx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to load model: {}", e),
                }),
            )
                .into_response();
        }
    };

    // Create KV cache
    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Generate
    let mut generated_text = String::new();
    let mut current_tokens = tokens.clone();
    let mut generated = 0;

    while generated < req.max_tokens {
        let logits = match ctx.model.forward(&current_tokens, &mut kv_cache) {
            Some(l) => l,
            None => break,
        };

        let next_token = sample_token(&logits, req.temperature) as u32;
        current_tokens.clear();
        current_tokens.push(next_token);

        let text = state.tokenizer.decode(&[next_token]);
        generated_text.push_str(&text);
        generated += 1;
    }

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text: generated_text,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: generated,
            total_tokens: prompt_tokens + generated,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}
