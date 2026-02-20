use std::fs;
use std::panic::{catch_unwind, UnwindSafe};
use std::path::Path;

use anyhow::{ensure, Result};
use candle_core::{utils, DType, Device};
use candle_nn::{embedding, linear_no_bias, VarBuilder};
use serde::Deserialize;

use crate::check_points::ModelFiles;
use crate::model::attention::Block;
use crate::model::rope::RoPE;
use crate::model::GPT;
use crate::model::{
    attention::{CausalSelfAttention, Mlp},
    gpt::GPTConfig,
};
use crate::tokenizer::{TiktokenEncoding, Tokenizer};

pub fn build_gpt(vb: VarBuilder, config: &GPTConfig) -> Result<GPT> {
    let lm_head = linear_no_bias(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;
    let vb = vb.pp("transformer");
    let token_embed = embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
    let mut blocks = Vec::with_capacity(config.n_layer);
    for idx in 0..config.n_layer {
        blocks.push(load_block(vb.pp(format!("h.{idx}")), config, idx)?);
    }
    // Prepare RoPE for this chunk on the correct device
    let head_dim = config.n_embd / config.n_head;
    // Create RoPE with sufficient positions (use 10x buffer like Python)
    let max_positions = config.sequence_len * 2;
    let rope = RoPE::new(
        token_embed.embeddings().device(),
        DType::BF16, // use bf16 for python implementation compatibility
        max_positions,
        head_dim,
    )?;
    Ok(GPT::new(config.clone(), token_embed, blocks, lm_head, rope))
}

pub(crate) fn load_block(vb: VarBuilder, config: &GPTConfig, layer_idx: usize) -> Result<Block> {
    const EXPANSION_FACTOR: usize = 4;
    let hidden = config.n_embd * EXPANSION_FACTOR;

    let attn = load_attn(vb.pp("attn"), config, layer_idx)?;
    let vb = vb.pp("mlp");
    let up = linear_no_bias(config.n_embd, hidden, vb.pp("c_fc"))?;
    let down = linear_no_bias(hidden, config.n_embd, vb.pp("c_proj"))?;
    let mlp = Mlp::new(up, down);
    Ok(Block::new(attn, mlp))
}

fn load_attn(vb: VarBuilder, config: &GPTConfig, _layer_idx: usize) -> Result<CausalSelfAttention> {
    let head_dim = config.n_embd / config.n_head;
    let q = linear_no_bias(config.n_embd, config.n_head * head_dim, vb.pp("c_q"))?;
    let k = linear_no_bias(config.n_embd, config.n_kv_head * head_dim, vb.pp("c_k"))?;
    let v = linear_no_bias(config.n_embd, config.n_kv_head * head_dim, vb.pp("c_v"))?;
    let o = linear_no_bias(config.n_head * head_dim, config.n_embd, vb.pp("c_proj"))?;

    Ok(CausalSelfAttention::new(q, k, v, o, config))
}

pub fn var_builder<'a>(path: &'a Path, dtype: DType, device: &'a Device) -> Result<VarBuilder<'a>> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let vb = match ext.as_str() {
        "safetensors" => unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? },
        "pt" | "pth" => VarBuilder::from_pth(path, dtype, device)?,
        _ => anyhow::bail!("unsupported model file extension: {}", ext),
    };
    Ok(vb)
}

#[derive(Debug, Deserialize)]
pub struct MetaConfig {
    pub model_config: GPTConfig,
}

impl MetaConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
}

pub fn load_model_from_files(files: &ModelFiles) -> Result<(GPT, Tokenizer)> {
    let dtype = DType::BF16;
    let device = pick_device(0)?;

    let meta = MetaConfig::from_file(&files.config)?;
    let config = &meta.model_config;
    config.validate()?;

    let vb = var_builder(&files.model, dtype, &device)?;
    let model = build_gpt(vb, config)?;
    let encoding = TiktokenEncoding::from_file(&files.tokenizer)?;
    let tokenizer = Tokenizer::from_encoding(encoding)?;
    ensure!(
        model.config().vocab_size == tokenizer.vocab_size(),
        "Model and tokenizer have different vocab sizes: {} != {}",
        model.config().vocab_size,
        tokenizer.vocab_size()
    );
    Ok((model, tokenizer))
}

pub fn pick_device(local_rank: usize) -> candle_core::Result<Device> {
    if utils::cuda_is_available() {
        // Try CUDA, fall back to CPU if unavailable/unusable.
        if let Some(device) = safe_try_device(|| Device::new_cuda(local_rank)) {
            return Ok(device);
        }
    }
    if utils::metal_is_available() {
        // Candle Metal initialization may panic on misconfigured environments.
        if let Some(device) = safe_try_device(|| Device::new_metal(0)) {
            return Ok(device);
        }
    }
    Ok(Device::Cpu)
}

fn safe_try_device<F>(f: F) -> Option<Device>
where
    F: FnOnce() -> candle_core::Result<Device> + UnwindSafe,
{
    match catch_unwind(f) {
        Ok(Ok(device)) => Some(device),
        Ok(Err(_)) | Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::safe_try_device;
    use candle_core::Device;

    #[test]
    fn safe_try_device_returns_none_on_panic() {
        let got = safe_try_device(|| -> candle_core::Result<Device> {
            panic!("simulated backend panic");
        });
        assert!(got.is_none());
    }

    #[test]
    fn safe_try_device_returns_device_on_success() {
        let got = safe_try_device(|| Ok(Device::Cpu));
        assert!(matches!(got, Some(Device::Cpu)));
    }
}
