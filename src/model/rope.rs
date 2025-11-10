/// Simplified and optimized RoPE implementation
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, D};

const BASE_THETA: f32 = 10_000.0;

/// Optimized RoPE implementation with better caching and simpler API
#[derive(Debug)]
pub struct RoPE {
    /// Pre-computed cos/sin tables with shape (max_positions, head_dim/2)
    cos: Tensor,
    sin: Tensor,
    /// Maximum sequence length we can handle
    max_positions: usize,
    /// Dimension of each attention head
    head_dim: usize,
}

impl RoPE {
    /// Create a new RoPE instance with pre-computed tables
    pub fn new(
        device: &Device,
        dtype: DType,
        max_positions: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| BASE_THETA.powf(-((2 * i) as f32 / head_dim as f32)))
            .collect();

        // Build position indices
        let positions = Tensor::arange(0u32, max_positions as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_positions, 1))?;

        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;

        // Compute angles = positions @ inv_freq
        let angles = positions.matmul(&inv_freq)?;

        // Pre-compute and store cos/sin
        let cos = angles.cos()?.to_dtype(dtype)?;
        let sin = angles.sin()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            max_positions,
            head_dim,
        })
    }

    /// Apply rotary embeddings to Q or K tensors
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, heads, head_dim]
    /// * `position_offset` - Starting position (for KV cache scenarios)
    ///
    /// Returns the rotated tensor (not in-place for simplicity)
    pub fn apply(&self, x: &Tensor, position_offset: usize) -> Result<Tensor> {
        let (_batch, seq_len, _heads, head_dim) = x.dims4()?;

        ensure!(
            head_dim == self.head_dim,
            "head_dim mismatch: expected {}, got {}",
            self.head_dim,
            head_dim
        );

        ensure!(
            position_offset + seq_len <= self.max_positions,
            "position range [{}, {}) exceeds max {}",
            position_offset,
            position_offset + seq_len,
            self.max_positions
        );

        // Compute in the input dtype: cast small cos/sin slices after slicing.
        let in_dtype = x.dtype();

        // Get the relevant slice of cos/sin tables, then cast to input dtype (cheaper than casting x)
        let cos = self
            .cos
            .narrow(0, position_offset, seq_len)?
            .to_dtype(in_dtype)?
            .unsqueeze(0)? // Add batch dimension
            .unsqueeze(2)?; // Add head dimension

        let sin = self
            .sin
            .narrow(0, position_offset, seq_len)?
            .to_dtype(in_dtype)?
            .unsqueeze(0)?
            .unsqueeze(2)?;

        // Split input into two halves
        let half = head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;

        // Apply rotation: standard RoPE formula
        // y1 = x1 * cos + x2 * sin
        // y2 = -x1 * sin + x2 * cos
        let y1 = (x1.broadcast_mul(&cos))?.broadcast_add(&x2.broadcast_mul(&sin)?)?;
        let y2 = (x2.broadcast_mul(&cos))?.broadcast_sub(&x1.broadcast_mul(&sin)?)?;

        // Concatenate the rotated halves (already in input dtype)
        Ok(Tensor::cat(&[&y1, &y2], D::Minus1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_v2_basic() -> Result<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(&device, DType::F32, 2048, 64)?;

        // Test with batch of sequences
        let x = Tensor::randn(0.0f32, 1.0, (2, 16, 8, 64), &device)?;
        let rotated = rope.apply(&x, 0)?;

        assert_eq!(rotated.dims(), x.dims());

        // Test with position offset (KV cache scenario)
        let rotated_offset = rope.apply(&x, 100)?;
        assert_eq!(rotated_offset.dims(), x.dims());

        Ok(())
    }
}
