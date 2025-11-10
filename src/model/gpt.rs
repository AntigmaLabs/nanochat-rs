use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module};
use serde::{Deserialize, Serialize};

use super::{attention::Block, kv::KVCache, rope::RoPE};
use crate::model::ops::norm;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,

    // max context window
    pub sequence_len: usize,
}

impl GPTConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.n_layer > 0, "config must have at least one layer");
        ensure!(self.n_head > 0, "config.n_head must be > 0");
        ensure!(
            self.n_embd.is_multiple_of(self.n_head),
            "n_embd ({}) must be divisible by n_head ({})",
            self.n_embd,
            self.n_head
        );
        let head_dim = self.n_embd / self.n_head;
        ensure!(head_dim.is_multiple_of(2), "head_dim must be even");

        ensure!(
            self.n_head.is_multiple_of(self.n_kv_head),
            "n_head ({}) must be a multiple of n_kv_head ({})",
            self.n_head,
            self.n_kv_head
        );
        Ok(())
    }
}

/// Complete GPT model that maps token ids to next-token logits.
#[derive(Debug)]
pub struct GPT {
    config: GPTConfig,
    token_embed: Embedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    rope: RoPE,
}

impl GPT {
    pub fn new(
        config: GPTConfig,
        token_embed: Embedding,
        blocks: Vec<Block>,
        lm_head: Linear,
        rope: RoPE,
    ) -> Self {
        Self {
            config,
            token_embed,
            blocks,
            lm_head,
            rope,
        }
    }

    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    pub fn forward(&self, input_ids: &Tensor, mut cache: Option<&mut KVCache>) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        ensure!(seq_len > 0, "forward expects seq_len > 0");

        // Validation happens in insert_layer, no need to duplicate here
        let tokens = self.token_embed.forward(input_ids)?;
        let mut hidden = norm(&tokens)?;

        for (idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &self.rope, cache.as_deref_mut(), idx)?;
        }
        let hidden = norm(&hidden)?;
        let mut logits = self.lm_head.forward(&hidden)?;

        // Apply logits softcap: softcap * tanh(logits / softcap)
        const SOFTCAP: f64 = 15.0;
        logits = ((logits / SOFTCAP)?.tanh()? * SOFTCAP)?;

        // No need to manually commit - insert_layer auto-commits on last layer
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        self.token_embed.embeddings().device()
    }
    pub fn dtype(&self) -> DType {
        self.token_embed.embeddings().dtype()
    }
}
