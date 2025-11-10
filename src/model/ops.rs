use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::ops::softmax_last_dim;

pub fn norm(x: &Tensor) -> Result<Tensor> {
    let last_dim = *x
        .dims()
        .last()
        .expect("norm expects tensor with at least 1 dimension");
    let alpha = Tensor::ones(&[last_dim], x.dtype(), x.device())?;
    // Use PyTorch's rms_norm default when eps=None, which is 1e-6
    let eps = 1e-6;
    Ok(candle_nn::ops::rms_norm(x, &alpha, eps)?)
}

/// Scaled dot-product attention.
///
/// Expects `q`, `k`, `v` shaped `[batch, heads, q_len, head_dim]`,
/// `[batch, heads, k_len, head_dim]`, `[batch, heads, k_len, head_dim]`.
/// If `mask` is provided, it should be a boolean/binary tensor broadcastable
/// to `[batch, heads, q_len, k_len]` where `true` means keep and `false` means mask.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let (_, _, _, hd) = q.dims4()?;

    let scale = (hd as f64).sqrt();
    // candle_nn::ops::sdpa(q, k, v, scale, 0.0f32) still has issue on metal, so use explicit matmul + softmax.
    let q_f = q.to_dtype(DType::F32)?.contiguous()?;
    let k_t = k.to_dtype(DType::F32)?.transpose(2, 3)?.contiguous()?;
    let v_f = v.to_dtype(DType::F32)?.contiguous()?;

    let mut scores = q_f.matmul(&k_t)?;

    scores = (scores / scale)?;

    if let Some(m) = mask {
        let m = m.broadcast_as(scores.shape())?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.dims().to_vec(), scores.device())?;
        scores = m.where_cond(&scores, &neg_inf)?;
    }

    let attn = softmax_last_dim(&scores)?;
    let ctx = attn.matmul(&v_f)?;
    Ok(ctx.to_dtype(in_dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    const EPS: f32 = 1e-6;

    #[test]
    fn norm_1d_matches_manual_rmsnorm() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![3f32, 4f32], (2,), &device)?;

        let y = norm(&x)?;

        let y0 = y.i(0)?.to_scalar::<f32>()?;
        let y1 = y.i(1)?.to_scalar::<f32>()?;

        let mean_sq = (3f32 * 3f32 + 4f32 * 4f32) / 2.0;
        let denom = (mean_sq + EPS).sqrt();
        let exp0 = 3f32 / denom;
        let exp1 = 4f32 / denom;

        assert!((y0 - exp0).abs() < 1e-5, "y0={y0} exp0={exp0}");
        assert!((y1 - exp1).abs() < 1e-5, "y1={y1} exp1={exp1}");
        Ok(())
    }

    #[test]
    fn norm_2d_zero_row_remains_zero() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![3f32, 4f32, 0f32, 0f32], (2, 2), &device)?;

        let y = norm(&x)?;

        let y10 = y.i((1, 0))?.to_scalar::<f32>()?;
        let y11 = y.i((1, 1))?.to_scalar::<f32>()?;

        assert!(y10.abs() < 1e-7);
        assert!(y11.abs() < 1e-7);
        Ok(())
    }
}
