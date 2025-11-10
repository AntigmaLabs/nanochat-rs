use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

use crate::model::ops::{norm, scaled_dot_product_attention};
use crate::model::rope::RoPE;
use crate::model::{gpt::GPTConfig, kv::KVCache};

/// Transformer block wiring attention and feed-forward sublayers.
#[derive(Debug)]
pub struct Block {
    attn: CausalSelfAttention,
    mlp: Mlp,
}

impl Block {
    pub fn new(attn: CausalSelfAttention, mlp: Mlp) -> Self {
        Self { attn, mlp }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let attn_input = norm(x)?;
        let attn_out = self.attn.forward(&attn_input, rope, cache, layer_idx)?;
        let residual = x.add(&attn_out)?;
        let mlp_input = norm(&residual)?;
        let mlp_out = self.mlp.forward(&mlp_input)?;
        Ok(residual.add(&mlp_out)?)
    }
}

#[derive(Debug)]
pub struct CausalSelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    kv_repeat: usize,
}

impl CausalSelfAttention {
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, config: &GPTConfig) -> Self {
        let head_dim = config.n_embd / config.n_head;
        let kv_repeat = config.n_head / config.n_kv_head;
        Self {
            q,
            k,
            v,
            o,
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            head_dim,
            kv_repeat,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        cache: Option<&mut KVCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        ensure!(seq_len > 0, "attention expects seq_len > 0");

        let mut q = self.q.forward(x)?;
        let mut k = self.k.forward(x)?;
        let mut v = self.v.forward(x)?;

        // Reshape to (B, T, H, D) to mirror the reference; apply RoPE and QK norm
        q = q.reshape((batch, seq_len, self.n_head, self.head_dim))?;
        k = k.reshape((batch, seq_len, self.n_kv_head, self.head_dim))?;
        v = v.reshape((batch, seq_len, self.n_kv_head, self.head_dim))?;

        let past_len = cache.as_ref().map(|c| c.pos()).unwrap_or(0);

        // Apply RoPE first, then QK norm (as per reference implementation)
        q = rope.apply(&q, past_len)?;
        k = rope.apply(&k, past_len)?;

        // Apply QK norm after RoPE
        q = norm(&q)?;
        k = norm(&k)?;

        // Transpose to (B, H, T, D) for attention kernels and make the data contiguous
        // Metal kernels reject matmuls fed with strided (non-contiguous) inputs.
        q = q.transpose(1, 2)?.contiguous()?;
        k = k.transpose(1, 2)?.contiguous()?;
        v = v.transpose(1, 2)?.contiguous()?;

        let (k_full, v_full) = if let Some(cache) = cache {
            // insert_layer writes to buffers and returns references to full buffers
            // We need to narrow to the valid portion for use in attention
            let (k_buf, v_buf) = cache.insert_layer(layer_idx, k, v)?;

            // Calculate total valid length (past_len + current seq_len)
            let total_len = past_len + seq_len;

            // Narrow to valid portion and clone the views (views are lightweight)
            let k_valid = k_buf.narrow(2, 0, total_len)?;
            let v_valid = v_buf.narrow(2, 0, total_len)?;

            (k_valid, v_valid)
        } else {
            // No caching - use k and v directly
            (k, v)
        };

        let total_len = k_full.dim(2)?;
        ensure!(
            total_len == past_len + seq_len,
            "unexpected cache length: expected {}, found {}",
            past_len + seq_len,
            total_len
        );

        let k_attn = self.repeat_kv(&k_full)?;
        let v_attn = self.repeat_kv(&v_full)?;

        // Mask is only needed when processing multiple queries at once (training or chunked inference).
        let mask = if seq_len > 1 {
            Some(
                build_causal_mask(q.device(), past_len, seq_len, total_len)?
                    .reshape((1, 1, seq_len, total_len))?,
            )
        } else {
            None
        };

        let context = scaled_dot_product_attention(&q, &k_attn, &v_attn, mask.as_ref())?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.n_head * self.head_dim))?;

        Ok(self.o.forward(&context)?)
    }

    fn repeat_kv(&self, xs: &Tensor) -> Result<Tensor> {
        if self.kv_repeat == 1 {
            Ok(xs.clone())
        } else {
            let (batch, n_kv_head, seq_len, head_dim) = xs.dims4()?;
            Ok(xs
                .unsqueeze(2)?
                .repeat((1, 1, self.kv_repeat, 1, 1))?
                .reshape((batch, n_kv_head * self.kv_repeat, seq_len, head_dim))?)
        }
    }
}

fn build_causal_mask(
    device: &Device,
    past_len: usize,
    seq_len: usize,
    total_len: usize,
) -> Result<Tensor> {
    // Build explicit (seq_len, total_len) grids to avoid ambiguous broadcasting
    let query_positions = Tensor::arange(past_len as u32, (past_len + seq_len) as u32, device)?
        .reshape((seq_len, 1))?;
    let key_positions = Tensor::arange(0u32, total_len as u32, device)?.reshape((1, total_len))?;

    let q_grid = query_positions.repeat((1, total_len))?; // (seq_len, total_len)
    let k_grid = key_positions.repeat((seq_len, 1))?; // (seq_len, total_len)
    Ok(q_grid.ge(&k_grid)?)
}

#[derive(Debug)]
pub struct Mlp {
    up: Linear,
    down: Linear,
}

impl Mlp {
    pub fn new(up: Linear, down: Linear) -> Self {
        Self { up, down }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.up.forward(x)?.relu()?.sqr()?;
        Ok(self.down.forward(&h)?)
    }
}
