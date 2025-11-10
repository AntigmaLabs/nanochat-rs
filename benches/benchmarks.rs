use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nanochat_rs::check_points::ModelFiles;
use nanochat_rs::engine::{Engine, SamplingParams};
use nanochat_rs::hf;
use nanochat_rs::model::builder::load_model_from_files;
use nanochat_rs::tokenizer::{special_tokens, TiktokenEncoding, Tokenizer};
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

fn tokenizer_benchmarks(c: &mut Criterion) {
    // Load tokenizer
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tokenizer.pkl");
    let encoding1 = TiktokenEncoding::from_file(&model_path).expect("Failed to load tokenizer");
    let tokenizer = Tokenizer::from_encoding(encoding1).expect("Failed to create tokenizer");

    // Different text samples for benchmarking
    let short_text = "Hello, world!";
    let medium_text = "The quick brown fox jumps over the lazy dog. This is a test of the tokenizer performance with a medium length sentence that includes various punctuation marks and symbols.";
    let long_text = r#"
In the realm of artificial intelligence and natural language processing, tokenization plays a crucial role
in breaking down text into manageable pieces. The byte-pair encoding (BPE) algorithm, originally developed
for data compression, has found widespread adoption in modern language models. It offers a balance between
vocabulary size and the ability to handle rare or unknown words by breaking them down into subword units.

This benchmark measures the performance of our Rust implementation of the BPE tokenizer, which is inspired
by the tiktoken library from OpenAI. The implementation focuses on both correctness and speed, using efficient
data structures and algorithms to process text quickly while maintaining compatibility with existing models.
"#;

    let mut group = c.benchmark_group("tokenizer_encoding_comparison");
    let samples = vec![
        ("short", short_text),
        ("medium", medium_text),
        ("long", long_text),
    ];

    for (label, text) in &samples {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("standard", *label), text, |b, t| {
            b.iter(|| tokenizer.encode(black_box(t)).unwrap());
        });
    }

    group.finish();

    // Benchmark decoding
    let mut decode_group = c.benchmark_group("tokenizer_decoding_comparison");
    let token_sets: Vec<(&str, Vec<_>)> = samples
        .iter()
        .map(|(label, text)| (*label, tokenizer.encode(text).unwrap()))
        .collect();

    for (label, tokens) in &token_sets {
        decode_group.throughput(Throughput::Elements(tokens.len() as u64));
        decode_group.bench_with_input(BenchmarkId::new("standard", *label), tokens, |b, t| {
            b.iter(|| tokenizer.decode(black_box(t)).unwrap());
        });
    }

    decode_group.finish();
}

fn generation_benchmark(c: &mut Criterion) {
    const MAX_TOKENS: usize = 16;
    let repo_id = "Antigma/nanochat-d32";
    let files = ModelFiles {
        config: hf::download(repo_id, "meta_d20.json").unwrap(),
        model: hf::download(repo_id, "model_d20.safetensors").unwrap(),
        tokenizer: hf::download(repo_id, "tokenizer.pkl").unwrap(),
    };

    let (model, tokenizer) = load_model_from_files(&files).expect("failed to model");
    let engine = Engine::new(model);

    let bos = tokenizer
        .encode_special(special_tokens::BOS)
        .expect("bos token");
    let user_start = tokenizer
        .encode_special(special_tokens::USER_START)
        .expect("user start token");
    let user_end = tokenizer
        .encode_special(special_tokens::USER_END)
        .expect("user end token");
    let assistant_start = tokenizer
        .encode_special(special_tokens::ASSISTANT_START)
        .expect("assistant start token");
    let assistant_end = tokenizer
        .encode_special(special_tokens::ASSISTANT_END)
        .expect("assistant end token");

    let mut conversation = vec![bos, user_start];
    conversation.extend(
        tokenizer
            .encode("Explain why Rust Candle sampling requires deterministic RNG.")
            .expect("encode prompt"),
    );
    conversation.extend([user_end, assistant_start]);

    let params = SamplingParams::new(0.0, None, Some(42), Default::default())
        .with_stop_tokens(vec![assistant_end, bos]);

    let mut group = c.benchmark_group("engine_generation");
    group.throughput(Throughput::Elements(MAX_TOKENS as u64));
    group.bench_function("prefill_plus_decode", |b| {
        b.iter(|| {
            let mut generator = engine
                .generate(black_box(&conversation), 1, &params)
                .expect("generator");

            let mut produced = 1usize;
            while produced < MAX_TOKENS && !generator.is_completed() {
                generator.decode_step().expect("decode");
                produced += 1;
            }
            black_box(produced);
        });
    });
    group.finish();
}

fn custom_criterion() -> Criterion {
    Criterion::default()
        .configure_from_args()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(10)
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = tokenizer_benchmarks, generation_benchmark
}
criterion_main!(benches);
