use std::path::PathBuf;

use anyhow::Result;
use hf_hub::api::sync::Api;
use tracing::debug;

pub fn clone(repo_id: &str) -> Result<std::path::PathBuf> {
    let repo = Api::new()?.model(repo_id.to_string());
    let info = repo.info()?;
    debug!("cloning model from HuggingFace: {:?}", info);
    let mut repo_root: Option<PathBuf> = None;
    for file in info.siblings {
        // Downloads to local HF cache (or returns cached path)
        let path = repo.get(&file.rfilename)?;
        let parent = path.parent().unwrap();
        if repo_root.is_none()
            || repo_root.as_ref().unwrap().to_str().unwrap().len() > parent.to_str().unwrap().len()
        {
            repo_root = Some(parent.to_path_buf()); // assume the repo is a single directory
        }
    }
    repo_root.ok_or_else(|| anyhow::anyhow!("No files found in HuggingFace repository"))
}

pub fn download(repo_id: &str, filename: &str) -> Result<std::path::PathBuf> {
    let repo = Api::new()?.model(repo_id.to_string());
    let path = repo.get(filename)?;
    Ok(path)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore = "requires internet access"]
    fn test_hf_download() {
        let repo_id = "Antigma/nanochat-d32";
        let path = clone(repo_id).unwrap();
        assert!(path.is_dir());
    }
}
