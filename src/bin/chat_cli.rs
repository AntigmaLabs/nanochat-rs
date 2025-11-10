use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;
use nanochat_rs::check_points::ModelFiles;
use nanochat_rs::engine::{Engine, SamplingParams};
use nanochat_rs::hf;
use nanochat_rs::model::builder::load_model_from_files;
use nanochat_rs::tokenizer::special_tokens;
use tracing_subscriber::filter::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "chat_cli", about = "NanoChat interactive CLI (Rust)")]
struct Args {
    /// Single-turn prompt (non-interactive). If empty, enters interactive REPL.
    #[arg(short = 'p', long = "prompt", default_value = "")]
    prompt: String,

    /// Temperature for sampling
    #[arg(short = 't', long = "temperature", default_value_t = 0.0)]
    temperature: f64,

    /// Top-k sampling (0 disables)
    #[arg(short = 'k', long = "top-k", default_value_t = 50)]
    top_k: usize,

    /// Max new tokens per assistant response
    #[arg(long = "max-tokens", default_value_t = 512)]
    max_tokens: usize,

    /// Source of the model local director or HuggingFace repository.
    /// nano
    #[arg(long = "source", default_value = "hf:Antigma/nanochat-d32")]
    source: String,

    /// RNG seed (optional). If omitted, uses OS RNG.
    #[arg(long = "seed", default_value_t = 42)]
    seed: u64,
}

#[derive(Debug)]
enum ModelSource {
    HuggingFace(String),
    Local(PathBuf),
}

fn parse_source(raw: &str) -> Result<ModelSource> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("--source must not be empty");
    }

    if let Some(repo) = trimmed.strip_prefix("hf:") {
        let repo_id = repo.trim();
        if repo_id.is_empty() {
            bail!("HuggingFace source must use the form 'hf:<repo_id>'");
        }
        Ok(ModelSource::HuggingFace(repo_id.to_owned()))
    } else {
        Ok(ModelSource::Local(PathBuf::from(trimmed)))
    }
}

fn init_tracing() {
    // Respect RUST_LOG when set, otherwise default to info for concise output.
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .try_init();
}

fn main() -> Result<()> {
    init_tracing();

    let args = Args::parse();
    let source = parse_source(&args.source)?;
    let dir = match source {
        ModelSource::HuggingFace(repo_id) => hf::clone(&repo_id)?,
        ModelSource::Local(path) => path,
    };

    let files = ModelFiles::new_from_dir(&dir)?;

    println!("Loading model from files: {:?}", files);
    let (model, tokenizer) = load_model_from_files(&files)?;
    println!("Model loaded: device={:?}", model.device());
    let engine = Engine::new(model);
    // Special tokens
    let bos = tokenizer.encode_special(special_tokens::BOS)?;
    let user_start = tokenizer.encode_special(special_tokens::USER_START)?;
    let user_end = tokenizer.encode_special(special_tokens::USER_END)?;
    let assistant_start = tokenizer.encode_special(special_tokens::ASSISTANT_START)?;
    let assistant_end = tokenizer.encode_special(special_tokens::ASSISTANT_END)?;

    let mut conversation: Vec<u32> = vec![bos];

    let params = SamplingParams::new(
        args.temperature,
        Some(args.top_k),
        Some(args.seed),
        Default::default(),
    );
    let params = params.with_stop_tokens(vec![assistant_end, bos]);

    println!("\nNanoChat Interactive Mode (Rust)");
    println!("{}", "-".repeat(50));
    println!("Type 'quit' or 'exit' to end the conversation");
    println!("Type 'clear' to start a new conversation");
    println!("{}", "-".repeat(50));

    loop {
        let user_input = if !args.prompt.is_empty() {
            args.prompt.clone()
        } else {
            print!("\nUser: ");
            io::stdout().flush().ok();
            let mut buf = String::new();
            let read = io::stdin().read_line(&mut buf);
            match read {
                Ok(0) => {
                    println!("\nGoodbye!");
                    break;
                }
                Ok(_) => buf.trim().to_string(),
                Err(_) => {
                    println!("\nGoodbye!");
                    break;
                }
            }
        };

        let lowered = user_input.to_lowercase();
        if lowered == "quit" || lowered == "exit" {
            println!("Goodbye!");
            break;
        }
        if lowered == "clear" {
            conversation.clear();
            conversation.push(bos);
            println!("Conversation cleared.");
            if !args.prompt.is_empty() {
                break;
            }
            continue;
        }
        if user_input.is_empty() {
            if !args.prompt.is_empty() {
                break;
            }
            continue;
        }

        // Append user message
        conversation.push(user_start);
        conversation.extend(tokenizer.encode(&user_input)?);
        conversation.push(user_end);

        // Begin assistant response
        conversation.push(assistant_start);

        // Streaming generation
        if conversation.is_empty() {
            bail!("Internal error: empty conversation");
        }
        let mut generator = engine.generate(&conversation, 1, &params)?;

        let mut generated: Vec<u32> = Vec::new();
        print!("\nAssistant: ");
        io::stdout().flush().ok();

        // Print first prefill-sampled token
        let mut last = generator.current_tokens()[0];
        generated.push(last);
        print!("{}", tokenizer.decode(&[last])?);
        io::stdout().flush().ok();

        // Decode loop for remaining tokens
        for _ in 1..args.max_tokens {
            if last == assistant_end {
                break;
            }
            generator.decode_step()?;
            if generator.is_completed() {
                break;
            }
            last = generator.current_tokens()[0];
            generated.push(last);
            print!("{}", tokenizer.decode(&[last])?);
            io::stdout().flush().ok();
        }
        println!();

        if *generated.last().unwrap_or(&assistant_end) != assistant_end {
            generated.push(assistant_end);
        }
        conversation.extend(generated);

        // Single-turn mode exits after one response
        if !args.prompt.is_empty() {
            break;
        }
    }

    Ok(())
}
