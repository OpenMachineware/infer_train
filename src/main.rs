// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use clap::{Parser, Subcommand};

mod cli;
mod server;

#[derive(Parser)]
#[command(name = "infer_train")]
#[command(about = "High-performance LLM inference engine", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Quick CLI testing interface
    Cli {
        /// Path to the GGUF model file
        #[arg(short, long)]
        model: String,

        /// Device to use (cpu or gpu)
        #[arg(short, long, default_value = "gpu")]
        device: String,

        /// Maximum tokens to generate
        #[arg(short, long, default_value = "128")]
        max_tokens: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Input prompt
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// Start OpenAI/Claude compatible API server
    Server {
        /// Path to the GGUF model file
        #[arg(short, long)]
        model: String,

        /// Device to use (cpu or gpu)
        #[arg(short, long, default_value = "gpu")]
        device: String,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Context window size
        #[arg(long, default_value = "4096")]
        ctx_size: usize,
    },
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Cli { model, device, max_tokens, temperature, prompt } => {
            cli::run(
                &model,
                &device,
                max_tokens,
                temperature,
                prompt.as_deref(),
            );
        }
        Commands::Server { model, device, host, port, ctx_size } => {
            server::run(&model, &device, &host, port, ctx_size).await;
        }
    }
}
