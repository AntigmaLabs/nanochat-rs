use std::path::Path;
use std::{fs, path::PathBuf};

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelFiles {
    pub fn new(step: Option<String>) -> Result<Self> {
        let name_suffix = if let Some(tag) = step {
            format!("_{}", tag)
        } else {
            String::new()
        };
        Ok(Self {
            config: PathBuf::from(format!("meta{}.json", name_suffix)),
            model: PathBuf::from(format!("model{}.pt", name_suffix)),
            tokenizer: PathBuf::from("tokenizer.pkl"),
        })
    }

    pub fn new_from_dir(dir: &Path) -> Result<Self> {
        let step = find_last_step(dir);
        let step_str = step.map(|s| format!("{:06}", s));
        let mut files = Self::new(step_str)?;
        files.set_model_base(dir);
        files.set_tokenizer_base(dir);
        Ok(files)
    }

    pub fn set_model_base(&mut self, base_dir: &Path) {
        self.config = base_dir.join(self.config.clone());
        self.model = base_dir.join(self.model.clone());
    }
    pub fn set_tokenizer_base(&mut self, base_dir: &Path) {
        self.tokenizer = base_dir.join(self.tokenizer.clone());
    }

    pub fn config_path(&self) -> &str {
        self.config.to_str().unwrap_or_default()
    }

    pub fn model_path(&self) -> &str {
        self.model.to_str().unwrap_or_default()
    }

    pub fn tokenizer_path(&self) -> &str {
        self.tokenizer.to_str().unwrap_or_default()
    }
}

fn extract_step_from_filename(file_name: &str) -> Option<u64> {
    const PREFIX: &str = "model_";
    const SUFFIX: &str = ".pt";
    if !file_name.starts_with(PREFIX) || !file_name.ends_with(SUFFIX) {
        return None;
    }
    let step_str = &file_name[PREFIX.len()..file_name.len() - SUFFIX.len()];
    step_str.parse::<u64>().ok()
}

/// Find the highest step number among files named like `model_<step>.pt` in `checkpoint_dir`.
/// Mirrors reference/nanochat/nanochat/checkpoint_manager.py::find_last_step.
pub fn find_last_step<P: AsRef<Path>>(checkpoint_dir: P) -> Option<u64> {
    let checkpoint_dir = checkpoint_dir.as_ref();
    let entries = fs::read_dir(checkpoint_dir)
        .with_context(|| {
            format!(
                "failed to read checkpoint directory {}",
                checkpoint_dir.display()
            )
        })
        .ok()?;

    let mut last_step: Option<u64> = None;
    for entry in entries {
        let entry = entry
            .with_context(|| "failed to read directory entry")
            .ok()?;

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(step) = extract_step_from_filename(&name) {
            last_step = Some(last_step.map_or(step, |curr| curr.max(step)));
        }
    }
    last_step
}
