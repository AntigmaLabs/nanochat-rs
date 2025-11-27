use anyhow::Result;
use candle_core::{DType, Tensor, D};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossReduction {
    Mean,
    None,
}

/// Cross entropy loss for logits and integer class targets.
///
/// - logits: (B, T, V)
/// - targets: (B, T) with class indices in [0, V) or ignore_index
/// - ignore_index: targets equal to this value are ignored in reduction/averaging
/// - reduction:
///   - Mean: returns scalar ()
///   - None: returns per-position loss with shape (B, T)
pub fn cross_entropy_loss(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: i64,
    reduction: LossReduction,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let (_bt_b, _bt_t) = targets.dims2()?; // sanity, but don't bind
                                           // Cast logits to f32 for stable math (reference does logits.float() before CE)
    let logits = logits.to_dtype(DType::F32)?;
    // Flatten to (N, V) and (N,)
    let n = b * t;
    let logits2d = logits.reshape((n, v))?;
    // Prepare targets
    // valid mask = targets != ignore_index
    let ignore_t =
        Tensor::full(ignore_index, targets.dims(), targets.device())?.to_dtype(targets.dtype())?;
    let valid_mask = targets.ne(&ignore_t)?.to_dtype(DType::U8)?; // 1 = valid, 0 = ignore
                                                                  // Make a safe target tensor where invalid indices are set to 0 to avoid OOB during gather
    let safe_targets = targets.broadcast_mul(&valid_mask.to_dtype(DType::I64)?)?;
    // Compute logsumexp across last dim: lse = max + log(sum(exp(x - max)))
    let max_logits = logits2d.max(D::Minus1)?; // (N,)
    let shifted = logits2d.broadcast_sub(&max_logits.reshape((n, 1))?)?; // (N,V)
    let exp = shifted.exp()?; // (N,V)
    let sum_exp = exp.sum(D::Minus1)?; // (N,)
    let lse = max_logits.broadcast_add(&sum_exp.log()?)?; // (N,)
                                                          // Gather logit of target class
    let safe_targets_u32 = safe_targets.to_dtype(DType::U32)?;
    let gathered = logits2d.gather(&safe_targets_u32.reshape((n, 1))?, D::Minus1)?; // (N,1)
    let gathered = gathered.squeeze(D::Minus1)?; // (N,)
                                                 // NLL = logsumexp - logit[target]
    let nll = lse.broadcast_sub(&gathered)?; // (N,)
                                             // Zero out ignored positions (so they don't contribute if we later average with count)
    let nll_masked = nll.broadcast_mul(&valid_mask.to_dtype(DType::F32)?)?; // (N,)
    match reduction {
        LossReduction::None => {
            // Reshape back to (B, T)
            Ok(nll_masked.reshape((b, t))?)
        }
        LossReduction::Mean => {
            // Average over number of valid positions
            let denom = valid_mask
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            if denom == 0.0 {
                // If no valid tokens, return +inf like Python bpb path does in degenerate case
                Ok(Tensor::full(f32::INFINITY, (), logits.device())?)
            } else {
                let total = nll_masked.sum_all()?; // (1,)
                let denom_t = Tensor::full(denom, (), logits.device())?;
                let mean = total.broadcast_div(&denom_t)?;
                Ok(mean.reshape(())?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result as AnyResult;
    use candle_core::Device;

    #[test]
    fn test_cross_entropy_simple() -> AnyResult<()> {
        // Two positions, vocab=3. Targets: [1, 2]
        // logits are log-probabilities so that softmax probs are [0.1, 0.7, 0.2] and [0.2, 0.3, 0.5]
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                0.1f32.ln(),
                0.7f32.ln(),
                0.2f32.ln(), // pos 0
                0.2f32.ln(),
                0.3f32.ln(),
                0.5f32.ln(), // pos 1
            ],
            (1, 2, 3),
            &device,
        )?;
        let targets = Tensor::from_vec(vec![1i64, 2], (1, 2), &device)?;
        let loss = cross_entropy_loss(&logits, &targets, -1, LossReduction::Mean)?;
        let val = loss.to_scalar::<f32>()?;
        // expected ~= -ln(0.7) + -ln(0.5) / 2 = (0.356675 + 0.693147) / 2 = 0.524911
        assert!((val - 0.5249).abs() < 1e-3, "got {val}");
        Ok(())
    }
}
