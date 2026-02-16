use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::display::PrinterOptions;
use candle_core::{DType, Device, Tensor, D};
use datatest_stable::{harness, Result as TestResult};
use nanochat_rs::model::builder::{build_gpt, pick_device, var_builder, MetaConfig};
use nanochat_rs::model::TokenId;
use nanochat_rs::sampling::sample_next_token;
use nanochat_rs::tokenizer::{TiktokenEncoding, Tokenizer};
use pretty_assertions::assert_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;

const FIXTURES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

const TOLERANCE: f32 = 1e-4;

// Run once for each matching JSON file under fixtures/
harness! {
    { test = gpt_logits_test, root = format!("{}/gpt_logits", FIXTURES_DIR), pattern = r"^.*\.json$" },
    { test = tokenizer_test, root = format!("{}/tokenizer", FIXTURES_DIR), pattern = r"^.*\.json$" },
}

fn gpt_logits_test(path: &Path) -> TestResult<()> {
    candle_core::display::set_print_options(PrinterOptions {
        precision: 5,
        threshold: 1000,
        edge_items: 3,
        line_width: 80,
        sci_mode: None,
    });
    // Load model config
    let meta_path = PathBuf::from(format!("{}/meta.json", FIXTURES_DIR));
    let model_path = PathBuf::from(format!("{}/model.pt", FIXTURES_DIR));

    let meta = MetaConfig::from_file(&meta_path).unwrap();

    let device = pick_device(0).unwrap();
    let dtype = DType::F32;
    let vb = var_builder(&model_path, dtype, &device)?;
    let model = build_gpt(vb, &meta.model_config)?;

    // Load test case from the matched file (e.g., gpt_logits_fixtures.json)
    let case: LogitsCase = LogitsCase::from_file(path)?;
    let input = case.input_tensor(&device)?;
    let logits = model.forward(&input, None)?;
    assert_eq_tensor(&logits, &case.logits_tensor(&device)?);

    let mut rng = StdRng::seed_from_u64(case.seed);
    // Sample from the last time step only: logits is (B, T, V) -> take (B, V)
    let (_b, t, _v) = logits.dims3()?;
    let last = logits.narrow(D::Minus2, t - 1, 1)?.squeeze(D::Minus2)?; // (B, V)
    let next_tokens = sample_next_token(
        &last,
        &mut rng,
        case.temperature.unwrap_or(0.0),
        case.top_k,
    )?;
    let next_tokens = next_tokens.to_vec2::<u32>()?;

    assert_eq!(next_tokens[0], case.tokens);

    Ok(())
}

fn tokenizer_test(path: &Path) -> TestResult<()> {
    let encoding = TiktokenEncoding::from_file(Path::new("fixtures/tokenizer.pkl")).unwrap();
    let tokenizer = Tokenizer::from_encoding(encoding).unwrap();
    let case: TokenizerCase = TokenizerCase::from_file(path)?;
    let tokens = tokenizer.encode(&case.text).unwrap();
    assert_eq!(tokens, case.tokens);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, case.text);
    Ok(())
}

#[derive(Debug, Deserialize)]
struct TokenizerCase {
    text: String,
    tokens: Vec<u32>,
}
impl TokenizerCase {
    fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let case: Self = serde_json::from_str(&content)?;
        Ok(case)
    }
}
#[derive(Debug, Deserialize)]
struct LogitsCase {
    input: Vec<Vec<TokenId>>,   // [B, T]
    logits: Vec<Vec<Vec<f32>>>, // [B, T, V]
    seed: u64,
    tokens: Vec<TokenId>, // [B, T]
    temperature: Option<f64>,
    top_k: Option<usize>,
}
impl LogitsCase {
    fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let case: Self = serde_json::from_str(&content)?;
        Ok(case)
    }

    fn input_tensor(&self, device: &Device) -> Result<Tensor> {
        let bsz = self.input.len();
        let seqlen = self.input.first().map(|v| v.len()).unwrap_or(0);
        let flat: Vec<TokenId> = self.input.iter().flatten().copied().collect();
        Tensor::from_vec(flat, (bsz, seqlen), device).map_err(Into::into)
    }

    fn logits_tensor(&self, device: &Device) -> Result<Tensor> {
        let bsz = self.logits.len();
        let seqlen = self.logits.first().map(|v| v.len()).unwrap_or(0);
        let vocab_size = self.logits[0][0].len();
        let flat: Vec<f32> = self.logits.iter().flatten().flatten().copied().collect();
        Tensor::from_vec(flat, (bsz, seqlen, vocab_size), device).map_err(Into::into)
    }
}

fn assert_eq_tensor(got: &Tensor, exp: &Tensor) {
    assert_eq!(
        got.dims(),
        exp.dims(),
        "shape mismatch: got {:?}, expected {:?}",
        got.dims(),
        exp.dims()
    );

    let max = (got - exp)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    if max >= TOLERANCE {
        let got_str = format!("{}", got);
        let exp_str = format!("{}", exp);
        assert_eq!(
            got_str, exp_str,
            "max_abs_diff={max:.6e} (tol={TOLERANCE:.1e})"
        );
    }
}
