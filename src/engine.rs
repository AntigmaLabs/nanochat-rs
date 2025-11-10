use ahash::HashSet;
use anyhow::{bail, Result};
use candle_core::{Tensor, D};
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;

use crate::model::{KVCache, TokenId, GPT};
use crate::sampling::sample_next_token;

pub struct Engine {
    model: Arc<GPT>,
}

#[derive(Clone, Debug)]
pub struct SamplingParams {
    temperature: f64,
    top_k: Option<usize>,
    seed: Option<u64>,
    stop_tokens: HashSet<TokenId>,
}

impl SamplingParams {
    pub fn new(
        temperature: f64,
        top_k: Option<usize>,
        seed: Option<u64>,
        stop_tokens: HashSet<TokenId>,
    ) -> Self {
        Self {
            temperature,
            top_k,
            seed,
            stop_tokens,
        }
    }
    pub fn with_stop_tokens<T: IntoIterator<Item = TokenId>>(mut self, stop_tokens: T) -> Self {
        self.stop_tokens = stop_tokens.into_iter().collect();
        self
    }

    pub fn rng(&self) -> StdRng {
        match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut trng = rand::rng();
                StdRng::from_rng(&mut trng)
            }
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::new(0.0, None, None, Default::default())
    }
}

// iterator over generated sequences
pub struct Generator {
    model: Arc<GPT>,
    kv_cache: KVCache,
    sequences: Vec<RowState>,
    current_tokens: Vec<TokenId>,
    rng: StdRng,

    sampling_params: SamplingParams,
}

// state for each batch
#[derive(Clone, Default)]
struct RowState {
    tokens: Vec<TokenId>,
    completed: bool,
}
impl RowState {
    pub fn new(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
            completed: false,
        }
    }
}

impl Generator {
    /// Creates a new context state and performs the prefill phase
    fn new(
        model: Arc<GPT>,
        prefill: &[TokenId],
        num_samples: usize,
        params: &SamplingParams,
    ) -> Result<Self> {
        if prefill.is_empty() {
            bail!("prompt must be non-empty");
        }
        if num_samples == 0 {
            bail!("num_samples must be > 0");
        }
        let ctx_limit = model.config().sequence_len;

        // Initialize sequences
        let mut sequences: Vec<RowState> = vec![RowState::new(prefill.to_vec()); num_samples];

        let device = model.device();
        let dtype = model.dtype();

        // 1) Prefill: process prompt once with batch_size=1
        let mut kv_cache = KVCache::new(
            model.config().n_layer,
            (
                1,
                model.config().n_head,
                ctx_limit,
                model.config().n_embd / model.config().n_head,
            ),
            device,
            dtype,
        )?;
        let ids_prefill = Tensor::from_vec(prefill.to_vec(), (1, prefill.len()), device)?;
        let logits = model.forward(&ids_prefill, Some(&mut kv_cache))?;
        let (_b, t, _v) = logits.dims3()?;
        let last = logits.narrow(D::Minus2, t - 1, 1)?.squeeze(D::Minus2)?;

        // Sample first token
        let mut rng = params.rng();
        let first = sample_next_token(&last, &mut rng, params.temperature, params.top_k)?;
        let first_id = first.to_vec2::<u32>()?[0][0];

        // 2) Expand cache in-place for batch processing
        if num_samples > 1 {
            kv_cache.expand_batch(num_samples)?;
        }

        // Append first token to all sequences
        for row in &mut sequences {
            row.tokens.push(first_id);
        }

        let current_tokens = vec![first_id; num_samples];

        Ok(Self {
            model,
            sequences,
            current_tokens,
            kv_cache,
            rng,
            sampling_params: params.clone(),
        })
    }

    pub fn current_tokens(&self) -> &[TokenId] {
        &self.current_tokens
    }

    /// Performs one decode step: forward pass + sampling
    pub fn decode_step(&mut self) -> Result<()> {
        let device = self.model.device();
        let num_samples = self.sequences.len();

        // Forward with single token per sequence
        let ids = Tensor::from_vec(self.current_tokens.clone(), (num_samples, 1), device)?;
        let logits = self.model.forward(&ids, Some(&mut self.kv_cache))?;
        let last = logits.squeeze(D::Minus2)?;

        // Sample next tokens
        let next = sample_next_token(
            &last,
            &mut self.rng,
            self.sampling_params.temperature,
            self.sampling_params.top_k,
        )?;
        let next_ids = next.to_vec2::<u32>()?;

        // Update sequences and current tokens
        for (i, row) in self.sequences.iter_mut().enumerate() {
            let token = next_ids[i][0];
            self.current_tokens[i] = token;
            if row.completed {
                continue;
            }
            row.tokens.push(token);
            if self.sampling_params.stop_tokens.contains(&token) {
                row.completed = true;
            }
        }

        Ok(())
    }

    pub fn is_completed(&self) -> bool {
        self.sequences
            .iter()
            .all(|row| row.completed || row.tokens.len() >= self.model.config().sequence_len)
    }

    pub fn sequences(&self) -> Vec<Vec<TokenId>> {
        self.sequences
            .iter()
            .map(|row| row.tokens.clone())
            .collect()
    }
}

impl Engine {
    pub fn new(model: GPT) -> Self {
        Self {
            model: Arc::new(model),
        }
    }

    pub fn model(&self) -> &Arc<GPT> {
        &self.model
    }

    /// Autoregressive token generation with KV caching for efficiency.
    ///
    /// - prompt: initial token ids (shared across all samples)
    /// - num_samples: how many sequences to generate in parallel
    /// - max_new_tokens: how many tokens to generate
    pub fn generate(
        &self,
        prompt: &[TokenId],
        num_samples: usize,
        params: &SamplingParams,
    ) -> Result<Generator> {
        Generator::new(Arc::clone(&self.model), prompt, num_samples, params)
    }
}
