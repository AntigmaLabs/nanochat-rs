use ahash::AHashMap;
use anyhow::{ensure, Result};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use serde_pickle::de::DeOptions;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::{collections::HashMap, fs::File, io::BufReader, path::Path};
use tracing::debug;

use crate::model::TokenId;

// token id and rank are the same thing
pub type Rank = TokenId;

type FastMap<K, V> = AHashMap<K, V>;

/// A tokenizer that uses a BPE encoding to convert text into token IDs
pub struct Tokenizer {
    name: String,
    pattern: Regex,
    mergeable_ranks: FastMap<Vec<u8>, Rank>,
    special_tokens: FastMap<String, Rank>,
    decoder: FastMap<Rank, Vec<u8>>,
    special_tokens_decoder: FastMap<Rank, String>,
}

pub mod special_tokens {
    pub const BOS: &str = "<|bos|>";
    pub const USER_START: &str = "<|user_start|>";
    pub const USER_END: &str = "<|user_end|>";
    pub const ASSISTANT_START: &str = "<|assistant_start|>";
    pub const ASSISTANT_END: &str = "<|assistant_end|>";
}

impl Tokenizer {
    pub fn from_encoding(encoding: TiktokenEncoding) -> Result<Self> {
        let mut mergeable_ranks: FastMap<Vec<u8>, Rank> = FastMap::default();
        for (k, v) in encoding.mergeable_ranks.into_iter() {
            mergeable_ranks.insert(k.into_vec(), v);
        }

        let mut special_tokens: FastMap<String, Rank> = FastMap::default();
        for (k, v) in encoding.special_tokens.into_iter() {
            special_tokens.insert(k, v);
        }

        // Build reverse mappings for decoding
        let mut decoder: FastMap<Rank, Vec<u8>> = FastMap::default();
        for (bytes, token) in mergeable_ranks.iter() {
            decoder.insert(*token, bytes.clone());
        }

        let mut special_tokens_decoder: FastMap<Rank, String> = FastMap::default();
        for (text, token) in special_tokens.iter() {
            special_tokens_decoder.insert(*token, text.clone());
        }

        Ok(Self {
            name: encoding.name,
            pattern: Regex::new(&encoding.pat_str)?,
            mergeable_ranks,
            special_tokens,
            decoder,
            special_tokens_decoder,
        })
    }

    /// Core BPE encoding using a heap of candidate merges (tiktoken-style "merge-free" BPE)
    /// Returns the sequence of token IDs after applying all possible merges
    fn encode_bytes(&self, piece: &[u8]) -> Vec<Rank> {
        let n = piece.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![piece[0] as Rank];
        }

        // Doubly-linked list over indices 0..n-1
        let mut left: Vec<Option<usize>> = (0..n)
            .map(|i| if i == 0 { None } else { Some(i - 1) })
            .collect();
        let mut right: Vec<Option<usize>> = (0..n)
            .map(|i| if i + 1 < n { Some(i + 1) } else { None })
            .collect();
        let start: Vec<usize> = (0..n).collect();
        let mut alive: Vec<bool> = vec![true; n];
        let mut token: Vec<Rank> = (0..n).map(|i| piece[i] as Rank).collect();

        let mut heap: BinaryHeap<Reverse<Part>> = BinaryHeap::new(); // min-heap by rank

        let merged_rank =
            |i: usize, start: &Vec<usize>, right: &Vec<Option<usize>>| -> Option<Rank> {
                let j = right[i]?;
                let end_next = match right[j] {
                    Some(k) => start[k],
                    None => n,
                };
                let bytes = &piece[start[i]..end_next];
                self.mergeable_ranks.get(bytes).copied()
            };

        for i in 0..n {
            if right[i].is_some() {
                if let Some(rank) = merged_rank(i, &start, &right) {
                    heap.push(Reverse(Part { start: i, rank }));
                }
            }
        }

        while let Some(Reverse(Part { rank, start: i })) = heap.pop() {
            if !alive[i] {
                continue;
            }
            let j = match right[i] {
                Some(j) if alive[j] => j,
                _ => continue,
            };
            // Validate that this candidate is still current
            if merged_rank(i, &start, &right) != Some(rank) {
                continue; // stale
            }

            // Merge i and j: assign new token id (rank) to i, remove j
            token[i] = rank;
            // splice out j
            let r = right[j];
            right[i] = r;
            if let Some(rk) = r {
                left[rk] = Some(i);
            }
            alive[j] = false;

            // Update candidates around i
            if let Some(li) = left[i] {
                if alive[li] {
                    if let Some(rk) = merged_rank(li, &start, &right) {
                        heap.push(Reverse(Part {
                            start: li,
                            rank: rk,
                        }));
                    }
                }
            }
            if right[i].is_some() {
                if let Some(rk) = merged_rank(i, &start, &right) {
                    heap.push(Reverse(Part { start: i, rank: rk }));
                }
            }
        }

        // Collect tokens by walking from head
        let mut out = Vec::with_capacity(n);
        // find head
        let idx = (0..n).find(|&i| alive[i] && left[i].is_none());
        if let Some(mut cur) = idx {
            loop {
                out.push(token[cur]);
                match right[cur] {
                    Some(next) if alive[next] => cur = next,
                    _ => break,
                }
            }
        }
        out
    }

    /// Encode text into a sequence of token IDs
    /// Handles special tokens and applies BPE to regular text chunks
    pub fn encode(&self, text: &str) -> Result<Vec<Rank>> {
        self.encode_with_special_tokens(
            text,
            &self
                .special_tokens
                .keys()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>(),
        )
    }

    pub fn encode_special(&self, text: &str) -> Result<u32> {
        self.special_tokens
            .get(text)
            .copied()
            .ok_or(anyhow::anyhow!("Special token not found"))
    }

    /// Encode text with allowed special tokens
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        allowed_special: &Vec<&str>,
    ) -> Result<Vec<Rank>> {
        let mut tokens = Vec::new();
        let mut start = 0;

        while start < text.len() {
            // Check for special tokens at current position
            let mut found_special = false;
            if !allowed_special.is_empty() {
                for &special in allowed_special {
                    if text[start..].starts_with(special) {
                        if let Some(&token) = self.special_tokens.get(special) {
                            tokens.push(token);
                            start += special.len();
                            found_special = true;
                            break;
                        }
                    }
                }
            }

            if found_special {
                continue;
            }

            // Find the next chunk using the pattern
            if let Ok(Some(m)) = self.pattern.find(&text[start..]) {
                let chunk_start = start + m.start();
                let chunk_end = start + m.end();
                let chunk = &text[chunk_start..chunk_end];

                // Encode this chunk with BPE
                let chunk_tokens = self.encode_bytes(chunk.as_bytes());
                tokens.extend(chunk_tokens);

                start = chunk_end;
            } else {
                // No match, shouldn't happen with proper regex but handle gracefully
                break;
            }
        }

        Ok(tokens)
    }

    /// Decode a sequence of token IDs back into text
    pub fn decode(&self, tokens: &[Rank]) -> Result<String> {
        let mut bytes = Vec::new();
        let mut out = String::new();

        for &tok in tokens {
            if let Some(special_text) = self.special_tokens_decoder.get(&tok) {
                if !bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&bytes));
                    bytes.clear();
                }
                out.push_str(special_text);
            } else if let Some(token_bytes) = self.decoder.get(&tok) {
                bytes.extend(token_bytes);
            } else {
                debug!("Unknown token: {}", tok);
            }
        }

        if !bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&bytes));
        }

        Ok(out)
    }

    /// Get the tokenizer name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of tokens in the vocabulary
    pub fn vocab_size(&self) -> usize {
        self.mergeable_ranks.len() + self.special_tokens.len()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TiktokenEncoding {
    pub name: String,
    pub pat_str: String,
    pub mergeable_ranks: HashMap<ByteBuf, u32>,
    pub special_tokens: HashMap<String, u32>,
}

impl TiktokenEncoding {
    pub fn from_file(path: &Path) -> Result<Self> {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        ensure!(
            ext == "pkl",
            "File {path:?} must be a pickle file with extension .pkl"
        );
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let encoding: Self = serde_pickle::from_reader(reader, DeOptions::new())?;
        Ok(encoding)
    }
}

struct Part {
    start: usize,
    rank: Rank,
}

impl Ord for Part {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank
            .cmp(&other.rank)
            .then_with(|| self.start.cmp(&other.start))
    }
}

impl PartialOrd for Part {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Part {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.start == other.start
    }
}

impl Eq for Part {}

#[cfg(test)]
mod tests {
    use super::*;
    const TOKENIZER_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/tokenizer.pkl");
    #[test]
    fn test_from_pickle() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        println!("{:?}", encoding);
    }

    #[test]
    fn test_tokenizer_creation() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();
        assert!(!tokenizer.name().is_empty());
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_basic_encoding() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();

        // Test simple text
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text).unwrap();
        assert!(!tokens.is_empty(), "Tokens should not be empty");
        println!(
            "Encoded '{}' to {} tokens: {:?}",
            text,
            tokens.len(),
            tokens
        );
    }

    #[test]
    fn test_empty_string() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();

        let tokens = tokenizer.encode("").unwrap();
        assert_eq!(tokens.len(), 0, "Empty string should produce no tokens");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();

        let texts = vec![
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Testing 123",
            "Unicode: 你好世界 🌍",
            "Code: fn main() { println!(\"Hello\"); }",
        ];

        for text in texts {
            let tokens = tokenizer.encode(text).unwrap();
            let decoded = tokenizer.decode(&tokens).unwrap();
            println!("Original: {}", text);
            println!("Tokens: {:?}", tokens);
            println!("Decoded: {}", decoded);
            assert_eq!(
                text, decoded,
                "Round-trip encoding/decoding should preserve text"
            );
        }
    }

    #[test]
    fn test_single_byte() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();

        let text = "a";
        let tokens = tokenizer.encode(text).unwrap();
        assert_eq!(tokens.len(), 1, "Single character should be one token");

        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_decode_with_unknown_token() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();

        // Test with an extremely high token ID that likely doesn't exist
        let tokens = vec![999999];
        let decoded = tokenizer.decode(&tokens).unwrap();
        // Should gracefully handle unknown tokens (skip them)
        println!("Decoded unknown token: '{}'", decoded);
    }

    #[test]
    fn test_multi_line_text() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();

        let text = "Line 1\nLine 2\nLine 3";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(text, decoded, "Multi-line text should round-trip correctly");
    }

    #[test]
    fn test_numbers_and_symbols() {
        let encoding = TiktokenEncoding::from_file(Path::new(TOKENIZER_PATH)).unwrap();
        let tokenizer = Tokenizer::from_encoding(encoding).unwrap();

        let text = "123 + 456 = 579! @#$%^&*()";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(
            text, decoded,
            "Numbers and symbols should round-trip correctly"
        );
    }
}
