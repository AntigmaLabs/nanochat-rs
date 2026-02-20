use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use nanochat_rs::check_points::ModelFiles;
use nanochat_rs::engine::{Engine, SamplingParams};
use nanochat_rs::hf;
use nanochat_rs::model::builder::load_model_from_files;
use nanochat_rs::tokenizer::special_tokens;
use nanochat_rs::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, Notify};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

const MAX_MESSAGES_PER_REQUEST: usize = 500;
const MAX_MESSAGE_LENGTH: usize = 8_000;
const MAX_TOTAL_CONVERSATION_LENGTH: usize = 32_000;
const MIN_TEMPERATURE: f64 = 0.0;
const MAX_TEMPERATURE: f64 = 2.0;
const MIN_TOP_K: usize = 1;
const MAX_TOP_K: usize = 200;
const MIN_MAX_TOKENS: usize = 1;
const MAX_MAX_TOKENS: usize = 4_096;

#[derive(Parser, Debug)]
#[command(name = "chat_web", about = "NanoChat Axum web server")]
struct Args {
    /// Number of model replicas to load (one per worker/GPU).
    #[arg(short = 'n', long = "num-workers", default_value_t = 1)]
    num_workers: usize,

    /// Source of the model: hf:<repo_id> or filesystem path.
    #[arg(
        short = 'i',
        long = "source",
        default_value = "hf:Antigma/nanochat-d32"
    )]
    source: String,

    /// Default sampling temperature.
    #[arg(short = 't', long = "temperature", default_value_t = 0.8)]
    temperature: f64,

    /// Default top-k sampling parameter.
    #[arg(short = 'k', long = "top-k", default_value_t = 50)]
    top_k: usize,

    /// Default max new tokens per response.
    #[arg(short = 'm', long = "max-tokens", default_value_t = 512)]
    max_tokens: usize,

    /// Optional RNG seed for deterministic sampling (per worker).
    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Host to bind the HTTP server to.
    #[arg(long = "host", default_value = "0.0.0.0")]
    host: String,

    /// Port to bind the HTTP server to.
    #[arg(long = "port", default_value_t = 8000)]
    port: u16,

    /// Path to chat UI HTML file.
    #[arg(long = "ui-path")]
    ui_path: Option<PathBuf>,

    /// Path to logo SVG served at /logo.svg.
    #[arg(long = "logo-path")]
    logo_path: Option<PathBuf>,
}

#[derive(Clone)]
struct GenerationDefaults {
    temperature: f64,
    top_k: usize,
    max_tokens: usize,
    seed: Option<u64>,
}

#[derive(Clone)]
struct UiAssets {
    html: Option<String>,
    logo: Option<Vec<u8>>,
}

impl UiAssets {
    fn load(ui_path: Option<PathBuf>, logo_path: Option<PathBuf>) -> Self {
        let html = locate_text_asset(
            ui_path,
            &[
                "nanochat/ui.html",
                "reference/nanochat/ui.html",
                "reference/nanochat/nanochat/ui.html",
            ],
        );
        let logo = locate_binary_asset(
            logo_path,
            &[
                "nanochat/logo.svg",
                "reference/nanochat/logo.svg",
                "reference/nanochat/nanochat/logo.svg",
            ],
        );
        Self { html, logo }
    }
}

#[derive(Clone)]
struct AppState {
    worker_pool: Arc<WorkerPool>,
    defaults: Arc<GenerationDefaults>,
    assets: Arc<UiAssets>,
}

struct Worker {
    id: usize,
    engine: Engine,
    tokenizer: Tokenizer,
    device: String,
    bos: u32,
    user_start: u32,
    user_end: u32,
    assistant_start: u32,
    assistant_end: u32,
}

impl Worker {
    fn new(id: usize, engine: Engine, tokenizer: Tokenizer) -> Result<Self> {
        let bos = tokenizer.encode_special(special_tokens::BOS)?;
        let user_start = tokenizer.encode_special(special_tokens::USER_START)?;
        let user_end = tokenizer.encode_special(special_tokens::USER_END)?;
        let assistant_start = tokenizer.encode_special(special_tokens::ASSISTANT_START)?;
        let assistant_end = tokenizer.encode_special(special_tokens::ASSISTANT_END)?;
        let device = format!("{:?}", engine.model().device());
        Ok(Self {
            id,
            engine,
            tokenizer,
            device,
            bos,
            user_start,
            user_end,
            assistant_start,
            assistant_end,
        })
    }

    fn build_conversation(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos];
        for message in messages {
            let (start_token, end_token) = role_boundaries(
                message.role.as_str(),
                self.user_start,
                self.user_end,
                self.assistant_start,
                self.assistant_end,
            )?;
            tokens.push(start_token);
            tokens.extend(self.tokenizer.encode(&message.content)?);
            tokens.push(end_token);
        }
        tokens.push(self.assistant_start);
        Ok(tokens)
    }
}

fn role_boundaries(
    role: &str,
    user_start: u32,
    user_end: u32,
    assistant_start: u32,
    assistant_end: u32,
) -> Result<(u32, u32)> {
    match role {
        // The tokenizer doesn't define dedicated system delimiters, so map
        // system messages to user boundaries for prompt construction.
        "user" | "system" => Ok((user_start, user_end)),
        "assistant" => Ok((assistant_start, assistant_end)),
        other => bail!("Unsupported role during encoding: {}", other),
    }
}

struct WorkerPool {
    queue: Mutex<VecDeque<Arc<Worker>>>,
    notify: Notify,
    workers: Vec<Arc<Worker>>,
}

impl WorkerPool {
    fn new(workers: Vec<Arc<Worker>>) -> Self {
        let queue = Mutex::new(workers.iter().cloned().collect());
        Self {
            queue,
            notify: Notify::new(),
            workers,
        }
    }

    async fn acquire(&self) -> Arc<Worker> {
        loop {
            if let Some(worker) = {
                let mut queue = self.queue.lock().await;
                queue.pop_front()
            } {
                return worker;
            }
            self.notify.notified().await;
        }
    }

    async fn release(&self, worker: Arc<Worker>) {
        {
            let mut queue = self.queue.lock().await;
            queue.push_back(worker);
        }
        self.notify.notify_one();
    }

    async fn snapshot(&self) -> WorkerPoolSnapshot {
        let available = { self.queue.lock().await.len() };
        WorkerPoolSnapshot {
            total_workers: self.workers.len(),
            available_workers: available,
        }
    }

    fn worker_summaries(&self) -> Vec<WorkerSummary> {
        self.workers
            .iter()
            .map(|worker| WorkerSummary {
                worker_id: worker.id,
                device: worker.device.clone(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
struct WorkerPoolSnapshot {
    total_workers: usize,
    available_workers: usize,
}

#[derive(Debug, Clone, Serialize)]
struct WorkerSummary {
    worker_id: usize,
    device: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    temperature: Option<f64>,
    max_tokens: Option<usize>,
    top_k: Option<usize>,
}

#[derive(Debug)]
enum ApiError {
    Validation(String),
    Internal(anyhow::Error),
}

impl ApiError {
    fn validation<T: Into<String>>(msg: T) -> Self {
        Self::Validation(msg.into())
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        Self::Internal(value)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            ApiError::Validation(msg) => (StatusCode::BAD_REQUEST, msg).into_response(),
            ApiError::Internal(err) => {
                error!("internal server error: {:?}", err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error".to_string(),
                )
                    .into_response()
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    if args.num_workers == 0 {
        bail!("--num-workers must be at least 1");
    }
    let model_dir = resolve_model_dir(&args.source)?;
    info!("Loading model files from {}", model_dir.display());
    let files = ModelFiles::new_from_dir(&model_dir)
        .with_context(|| format!("Failed to locate model files in {}", model_dir.display()))?;

    let workers = load_workers(args.num_workers, &files)?;
    info!("Initialized {} worker(s)", workers.len());

    let state = AppState {
        worker_pool: Arc::new(WorkerPool::new(workers)),
        defaults: Arc::new(GenerationDefaults {
            temperature: args.temperature,
            top_k: args.top_k,
            max_tokens: args.max_tokens,
            seed: args.seed,
        }),
        assets: Arc::new(UiAssets::load(args.ui_path, args.logo_path)),
    };

    let router = Router::new()
        .route("/", get(root_handler))
        .route("/logo.svg", get(logo_handler))
        .route("/chat/completions", post(chat_completions))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("invalid host/port combination")?;
    info!("Starting NanoChat web server on http://{}", addr);
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

async fn root_handler(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    if let Some(html) = &state.assets.html {
        let content = html.replace(
            "const API_URL = `http://${window.location.hostname}:8000`;",
            "const API_URL = '';",
        );
        Ok(Html(content))
    } else {
        Ok(Html(
            "<h1>NanoChat</h1><p>UI not configured.</p>".to_string(),
        ))
    }
}

async fn logo_handler(State(state): State<AppState>) -> impl IntoResponse {
    if let Some(bytes) = &state.assets.logo {
        (
            StatusCode::OK,
            [("content-type", "image/svg+xml")],
            bytes.clone(),
        )
            .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let snapshot = state.worker_pool.snapshot().await;
    Json(json!({
        "status": "ok",
        "ready": snapshot.total_workers > 0 && snapshot.available_workers > 0,
        "num_workers": snapshot.total_workers,
        "available_workers": snapshot.available_workers,
    }))
}

async fn stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    let snapshot = state.worker_pool.snapshot().await;
    Json(json!({
        "total_workers": snapshot.total_workers,
        "available_workers": snapshot.available_workers,
        "busy_workers": snapshot.total_workers.saturating_sub(snapshot.available_workers),
        "workers": state.worker_pool.worker_summaries(),
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<impl IntoResponse, ApiError> {
    validate_chat_request(&request)?;

    info!("====================");
    for (idx, message) in request.messages.iter().enumerate() {
        info!(
            "[{}][{}]: {}",
            idx,
            message.role.to_uppercase(),
            message.content
        );
    }
    info!("--------------------");

    let worker = state.worker_pool.acquire().await;
    let worker_for_task = Arc::clone(&worker);
    let worker_for_release = Arc::clone(&worker);
    let defaults = Arc::clone(&state.defaults);
    let (tx, rx) =
        tokio::sync::mpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();
    let req_clone = request.clone();
    let pool = Arc::clone(&state.worker_pool);

    let generation = tokio::task::spawn_blocking(move || {
        if let Err(err) = generate_stream(worker_for_task, req_clone, defaults, tx.clone()) {
            error!("generation error: {:?}", err);
            let payload = json!({ "error": err.to_string() }).to_string();
            let _ = tx.send(Ok(Event::default().data(payload)));
        }
        let _ = tx.send(Ok(Event::default().data(json!({"done": true}).to_string())));
    });

    tokio::spawn(async move {
        if let Err(err) = generation.await {
            error!("generation task join error: {:?}", err);
        }
        pool.release(worker_for_release).await;
        info!("====================");
    });

    let stream = UnboundedReceiverStream::new(rx);
    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)).text("")))
}

fn generate_stream(
    worker: Arc<Worker>,
    request: ChatRequest,
    defaults: Arc<GenerationDefaults>,
    tx: tokio::sync::mpsc::UnboundedSender<Result<Event, std::convert::Infallible>>,
) -> Result<()> {
    let temperature = clamp_f64(
        request.temperature.unwrap_or(defaults.temperature),
        MIN_TEMPERATURE,
        MAX_TEMPERATURE,
    );
    let top_k = clamp_usize(
        request.top_k.unwrap_or(defaults.top_k),
        MIN_TOP_K,
        MAX_TOP_K,
    );
    let mut max_tokens = clamp_usize(
        request.max_tokens.unwrap_or(defaults.max_tokens),
        MIN_MAX_TOKENS,
        MAX_MAX_TOKENS,
    );
    if max_tokens == 0 {
        max_tokens = defaults.max_tokens.max(1);
    }

    let conversation = worker.build_conversation(&request.messages)?;
    let mut sampling =
        SamplingParams::new(temperature, Some(top_k), defaults.seed, Default::default());
    sampling = sampling.with_stop_tokens(vec![worker.assistant_end, worker.bos]);

    let mut generator = worker
        .engine
        .generate(&conversation, 1, &sampling)
        .context("failed to start generator")?;

    let mut generated: Vec<u32> = Vec::new();
    let mut steps = 0usize;

    let mut last = generator.current_tokens()[0];
    generated.push(last);
    emit_token(&worker, last, &tx)?;

    while steps < max_tokens {
        if last == worker.assistant_end {
            break;
        }
        generator.decode_step()?;
        if generator.is_completed() {
            break;
        }
        last = generator.current_tokens()[0];
        generated.push(last);
        emit_token(&worker, last, &tx)?;
        steps += 1;
    }

    let response_text = worker
        .tokenizer
        .decode(&generated)
        .unwrap_or_else(|_| "<decode error>".to_string());
    info!("[ASSISTANT] (worker {}): {}", worker.id, response_text);
    Ok(())
}

fn emit_token(
    worker: &Worker,
    token: u32,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<Event, std::convert::Infallible>>,
) -> Result<()> {
    let text = worker.tokenizer.decode(&[token])?;
    if !text.is_empty() {
        let payload = json!({"token": text, "worker": worker.id}).to_string();
        let _ = tx.send(Ok(Event::default().data(payload)));
    }
    Ok(())
}

fn validate_chat_request(request: &ChatRequest) -> Result<(), ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::validation("At least one message is required"));
    }
    if request.messages.len() > MAX_MESSAGES_PER_REQUEST {
        return Err(ApiError::validation(format!(
            "Too many messages. Maximum {} messages allowed per request",
            MAX_MESSAGES_PER_REQUEST
        )));
    }
    let mut total_length = 0usize;
    for (idx, message) in request.messages.iter().enumerate() {
        if message.content.trim().is_empty() {
            return Err(ApiError::validation(format!(
                "Message {} has empty content",
                idx
            )));
        }
        let len = message.content.chars().count();
        if len > MAX_MESSAGE_LENGTH {
            return Err(ApiError::validation(format!(
                "Message {} is too long. Maximum {} characters allowed per message",
                idx, MAX_MESSAGE_LENGTH
            )));
        }
        total_length += len;
        if !matches!(message.role.as_str(), "user" | "assistant" | "system") {
            return Err(ApiError::validation(format!(
                "Message {} has invalid role. Must be 'user', 'assistant', or 'system'",
                idx
            )));
        }
    }
    if total_length > MAX_TOTAL_CONVERSATION_LENGTH {
        return Err(ApiError::validation(format!(
            "Total conversation is too long. Maximum {} characters allowed",
            MAX_TOTAL_CONVERSATION_LENGTH
        )));
    }
    if let Some(temp) = request.temperature {
        if !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temp) {
            return Err(ApiError::validation(format!(
                "Temperature must be between {} and {}",
                MIN_TEMPERATURE, MAX_TEMPERATURE
            )));
        }
    }
    if let Some(top_k) = request.top_k {
        if !(MIN_TOP_K..=MAX_TOP_K).contains(&top_k) {
            return Err(ApiError::validation(format!(
                "top_k must be between {} and {}",
                MIN_TOP_K, MAX_TOP_K
            )));
        }
    }
    if let Some(max_tokens) = request.max_tokens {
        if !(MIN_MAX_TOKENS..=MAX_MAX_TOKENS).contains(&max_tokens) {
            return Err(ApiError::validation(format!(
                "max_tokens must be between {} and {}",
                MIN_MAX_TOKENS, MAX_MAX_TOKENS
            )));
        }
    }
    Ok(())
}

fn clamp_f64(value: f64, min: f64, max: f64) -> f64 {
    value.clamp(min, max)
}

fn clamp_usize(value: usize, min: usize, max: usize) -> usize {
    value.clamp(min, max)
}

fn resolve_model_dir(source: &str) -> Result<PathBuf> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        bail!("--source must not be empty");
    }
    if let Some(repo) = trimmed.strip_prefix("hf:") {
        let repo_id = repo.trim();
        if repo_id.is_empty() {
            bail!("HuggingFace source must use the form 'hf:<repo_id>'");
        }
        hf::clone(repo_id)
    } else {
        let path = PathBuf::from(trimmed);
        if !path.exists() {
            bail!("Model directory does not exist: {}", path.display());
        }
        Ok(path)
    }
}

fn load_workers(num_workers: usize, files: &ModelFiles) -> Result<Vec<Arc<Worker>>> {
    let mut workers = Vec::with_capacity(num_workers);
    for worker_id in 0..num_workers {
        info!("Loading worker {}", worker_id);
        let (model, tokenizer) = load_model_from_files(files)?;
        let engine = Engine::new(model);
        let worker = Worker::new(worker_id, engine, tokenizer)
            .with_context(|| format!("failed to construct worker {}", worker_id))?;
        let device = worker.device.clone();
        let tokenizer_name = worker.tokenizer.name().to_string();
        info!(
            "Worker {} ready on device {} (tokenizer: {})",
            worker_id, device, tokenizer_name
        );
        workers.push(Arc::new(worker));
    }
    Ok(workers)
}

fn locate_text_asset(path: Option<PathBuf>, fallbacks: &[&str]) -> Option<String> {
    if let Some(path) = path {
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                info!("Loaded UI HTML from {}", path.display());
                return Some(content);
            }
            Err(err) => {
                warn!("Failed to read UI HTML {}: {}", path.display(), err);
            }
        }
    }
    for candidate in fallbacks {
        let path = Path::new(candidate);
        if path.exists() {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    info!("Loaded UI HTML from {}", path.display());
                    return Some(content);
                }
                Err(err) => warn!("Failed to read {}: {}", path.display(), err),
            }
        }
    }
    None
}

fn locate_binary_asset(path: Option<PathBuf>, fallbacks: &[&str]) -> Option<Vec<u8>> {
    if let Some(path) = path {
        match std::fs::read(&path) {
            Ok(bytes) => {
                info!("Loaded logo from {}", path.display());
                return Some(bytes);
            }
            Err(err) => {
                warn!("Failed to read logo {}: {}", path.display(), err);
            }
        }
    }
    for candidate in fallbacks {
        let path = Path::new(candidate);
        if path.exists() {
            match std::fs::read(path) {
                Ok(bytes) => {
                    info!("Loaded logo from {}", path.display());
                    return Some(bytes);
                }
                Err(err) => warn!("Failed to read {}: {}", path.display(), err),
            }
        }
    }
    None
}

fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::{
        role_boundaries, validate_chat_request, ChatMessage, ChatRequest, MAX_MESSAGES_PER_REQUEST,
        MAX_MESSAGE_LENGTH,
    };

    #[test]
    fn role_boundaries_accepts_system_as_user_delimiters() {
        let user_start = 10u32;
        let user_end = 11u32;
        let assistant_start = 20u32;
        let assistant_end = 21u32;

        let got = role_boundaries(
            "system",
            user_start,
            user_end,
            assistant_start,
            assistant_end,
        )
        .expect("system should be encodable");

        assert_eq!(got, (user_start, user_end));
    }

    #[test]
    fn validate_chat_request_accepts_system_role() {
        let req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "be concise".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
            ],
            temperature: Some(0.8),
            max_tokens: Some(64),
            top_k: Some(50),
        };

        validate_chat_request(&req).expect("system role should pass request validation");
    }

    #[test]
    fn validate_chat_request_rejects_too_many_messages() {
        let messages = (0..=MAX_MESSAGES_PER_REQUEST)
            .map(|_| ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            })
            .collect();
        let req = ChatRequest {
            messages,
            temperature: None,
            max_tokens: None,
            top_k: None,
        };

        let err = validate_chat_request(&req).expect_err("should reject message overflow");
        let msg = match err {
            super::ApiError::Validation(msg) => msg,
            super::ApiError::Internal(_) => "internal".to_string(),
        };
        assert!(msg.contains("Too many messages"));
    }

    #[test]
    fn validate_chat_request_rejects_overlong_message() {
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "a".repeat(MAX_MESSAGE_LENGTH + 1),
            }],
            temperature: None,
            max_tokens: None,
            top_k: None,
        };

        let err = validate_chat_request(&req).expect_err("should reject long message");
        let msg = match err {
            super::ApiError::Validation(msg) => msg,
            super::ApiError::Internal(_) => "internal".to_string(),
        };
        assert!(msg.contains("too long"));
    }
}
