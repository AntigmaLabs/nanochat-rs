use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};

/// Per-layer key/value cache with pre-allocated buffers.
/// Each layer stores k,v in fixed-size buffers, avoiding concatenation overhead.
#[derive(Debug)]
pub struct KVCache {
    /// Pre-allocated (K, V) buffers per layer, each shaped (B, H, max_pos, D)
    layers: Vec<(Tensor, Tensor)>,

    pos: usize,
}

impl KVCache {
    /// Create a new KVCache with pre-allocated buffers.
    ///
    /// # Arguments
    /// * `num_layers` - Number of layers in the model
    /// * `shape` - Shape of the cache tensors (B, H, max_pos, D)
    /// * `device` - Device to allocate tensors on
    /// * `dtype` - Data type for cache tensors
    pub fn new(
        num_layers: usize,
        shape: (usize, usize, usize, usize),
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        ensure!(num_layers > 0, "num_layers must be > 0");
        // Pre-allocate buffers for all layers
        let mut kv_buffers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            // Allocate zero-initialized buffers: (B, H, max_pos, D)
            let k = Tensor::zeros(shape, dtype, device)?;
            let v = Tensor::zeros(shape, dtype, device)?;
            kv_buffers.push((k, v));
        }

        Ok(Self {
            layers: kv_buffers,
            pos: 0,
        })
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn max_pos(&self) -> usize {
        self.layers[0].0.dim(2).expect("max_pos must be > 0")
    }

    /// Ensure the cache has at least `min_capacity` positions along the sequence dimension.
    /// If needed, grows the underlying K/V buffers for all layers and preserves existing data.
    pub fn ensure_capacity(&mut self, min_capacity: usize) -> Result<()> {
        let (b, h, old_max, d) = self.layers[0].0.dims4()?;
        if min_capacity <= old_max {
            return Ok(());
        }

        // Grow capacity geometrically to amortize reallocation cost.
        let mut new_max = old_max.max(1);
        while new_max < min_capacity {
            new_max *= 2;
        }

        let dtype = self.layers[0].0.dtype();
        let device = self.layers[0].0.device().clone();

        for layer_idx in 0..self.layers.len() {
            let mut new_k = Tensor::zeros((b, h, new_max, d), dtype, &device)?;
            let mut new_v = Tensor::zeros((b, h, new_max, d), dtype, &device)?;

            let valid_len = self.pos;
            if valid_len > 0 {
                let k_src = self.layers[layer_idx].0.narrow(2, 0, valid_len)?;
                let v_src = self.layers[layer_idx].1.narrow(2, 0, valid_len)?;
                new_k = new_k.slice_scatter(&k_src, 2, 0)?;
                new_v = new_v.slice_scatter(&v_src, 2, 0)?;
            }

            self.layers[layer_idx].0 = new_k;
            self.layers[layer_idx].1 = new_v;
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Insert keys and values into the cache using pre-allocated buffers.
    /// Returns references to the valid portion of cached K,V tensors (avoids cloning).
    ///
    /// This method writes into pre-allocated buffers, avoiding concatenation overhead.
    pub fn insert_layer(
        &mut self,
        layer_idx: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<(&Tensor, &Tensor)> {
        // Validate layer index
        debug_assert!(layer_idx < self.layers.len(), "layer index out of range");

        let seq_len = k.dim(2)?;
        ensure!(seq_len > 0, "sequence length must be non-zero");

        let required = self.pos + seq_len;

        // Ensure capacity up-front for this write
        self.ensure_capacity(required)?;

        // Write k,v into the pre-allocated buffers at the current position
        // Use slice_scatter to write into buffer: buffer[.., .., pos:pos+seq_len, ..] = k/v
        self.layers[layer_idx].0 = self.layers[layer_idx].0.slice_scatter(&k, 2, self.pos)?;
        self.layers[layer_idx].1 = self.layers[layer_idx].1.slice_scatter(&v, 2, self.pos)?;

        // Auto-commit: advance position after last layer (like Python reference)
        if layer_idx == self.layers.len() - 1 {
            self.pos += seq_len;
        }

        // Return references to buffers (caller will narrow as needed)
        Ok((&self.layers[layer_idx].0, &self.layers[layer_idx].1))
    }

    /// Expand this cache's batch dimension in-place.
    ///
    /// If current batch size is 1, this will repeat the valid prefix across the
    /// batch dimension to reach `new_batch`. The underlying buffers are
    /// reallocated to the new shape (new_batch, H, max_pos, D). The `pos` is
    /// preserved. If `new_batch` equals the current batch size, this is a no-op.
    pub fn expand_batch(&mut self, new_batch: usize) -> Result<()> {
        ensure!(new_batch > 0, "new_batch must be > 0");

        let (cur_b, h, max_pos, d) = self.layers[0].0.dims4()?;
        if new_batch == cur_b {
            return Ok(());
        }

        // Only support expanding from batch size 1 -> new_batch for now
        ensure!(
            cur_b == 1,
            "unsupported batch size change: current {} -> new {} (only 1->B supported)",
            cur_b,
            new_batch
        );

        let dtype = self.layers[0].0.dtype();
        let device = self.layers[0].0.device().clone();

        for layer_idx in 0..self.layers.len() {
            // Allocate new buffers
            let mut new_k = Tensor::zeros((new_batch, h, max_pos, d), dtype, &device)?;
            let mut new_v = Tensor::zeros((new_batch, h, max_pos, d), dtype, &device)?;

            let valid_len = self.pos;
            if valid_len > 0 {
                // Take the valid prefix from existing buffers
                let k_src = self.layers[layer_idx].0.narrow(2, 0, valid_len)?;
                let v_src = self.layers[layer_idx].1.narrow(2, 0, valid_len)?;

                // Repeat along batch dimension
                let k_ready = k_src.repeat((new_batch, 1, 1, 1))?;
                let v_ready = v_src.repeat((new_batch, 1, 1, 1))?;

                // Scatter into new buffers at position 0
                new_k = new_k.slice_scatter(&k_ready, 2, 0)?;
                new_v = new_v.slice_scatter(&v_ready, 2, 0)?;
            }

            self.layers[layer_idx].0 = new_k;
            self.layers[layer_idx].1 = new_v;
        }

        Ok(())
    }
}
