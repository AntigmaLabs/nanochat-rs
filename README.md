# Nanochat-rs: Tiny GPT-style cognitive core in pure Rust
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/CbAsUR434B)
[![Website](https://img.shields.io/badge/Website-antigma.ai-orange?logo=google-chrome&logoColor=white)](https://antigma.ai)
[![Twitter](https://img.shields.io/twitter/follow/antigma_labs?style=social)](https://twitter.com/antigma_labs)
[![🤗 Hugging Face](https://img.shields.io/badge/HuggingFace-Antigma-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/Antigma)


This is a rust implementation of [karpathy/nanochat](https://github.com/karpathy/nanochat). 
Built on [candle](https://github.com/huggingface/candle), focused on clarity and parity with
the reference while keeping the code minimal.


> [!WARNING]
> Experimental; focused on inference for now.


## Features
- Native Rust
- Hugging Face integration
- Centralized model loader resilient to tensor name changes
- Minimal surface area to keep cognitive load low (not production-grade; performance–simplicity trade-off)
- Compatible with tiktoken `.pkl` tokenizer configs

### Differences from the referenced nanochat
- Tokenizer encoding/decoding is production-ready and unified (no `tiktoken` dependency at runtime)
- Removed the embedded Python interpreter in the engine
- Performance and ergonomics improvements in the generation logic
- More emphasis on post-training


## Quick start
We have a 32‑layer version trained with about \$1000 budget on Hugging Face: https://huggingface.co/Antigma/nanochat-d32  
There is also a smaller 20‑layer version (d20) used for benchmarks and testing within the same Hugging Face repo.

run on Apple With GPU
```
cargo run --release --features metal -- -p "write 100 words"
```

with CUDA
```
cargo run --release --features cuda -- -p "write 100 words"
```

We also provide a ChatGPT-like web UI. To launch it, run:

```
cargo run --bin chat_web
```

After starting the server, visit the URL displayed in the terminal (by default, [http://localhost:8000/](http://localhost:8000/), or use your host's public IP/port if running on a remote machine).

You can now interact with your local LLM just like ChatGPT—try asking creative questions, writing stories, or exploring model behavior.

> [!NOTE]
> The web UI assets (`ui.html`, `logo.svg`, etc.) live inside the `reference/nanochat` submodule. Make sure you initialize submodules before running `chat_web`:
> ```
> git submodule update --init --recursive
> ```
> The web server loads a model from Hugging Face by default (`hf:Antigma/nanochat-d32`). If you already have the weights locally, point the server to them with `--source /path/to/model_dir` to avoid unnecessary downloads.

### Server flags
```
cargo run --release --bin chat_web -- \
  [--num-workers N] \
  [--source hf:<repo_id>|/path/to/model_dir] \
  [--temperature FLOAT] [--top-k INT] [--max-tokens INT] [--seed INT] \
  [--host HOST] [--port PORT] \
  [--ui-path /path/to/ui.html] [--logo-path /path/to/logo.svg]
```
- `--num-workers`: number of model replicas to load (e.g., one per GPU)
- `--source`: `hf:<repo_id>` or a local directory with model files
- `--temperature`, `--top-k`, `--max-tokens`, `--seed`: sampling defaults
- `--host`, `--port`: bind address (defaults to `0.0.0.0:8000`)
- `--ui-path`, `--logo-path`: override embedded/fallback UI assets

### REST API
- Health:
```
GET /health
```
- Stats:
```
GET /stats
```
- Chat completions (SSE stream via POST):
```
POST /chat/completions
Content-Type: application/json
{
  "messages": [
    { "role": "user", "content": "Hello!" }
  ],
  "temperature": 0.8,
  "top_k": 50,
  "max_tokens": 256
}
```
Example:
```
curl -N -H "Content-Type: application/json" \
  -X POST \
  --data '{"messages":[{"role":"user","content":"Write a haiku about Rust."}]}' \
  http://localhost:8000/chat/completions
```
Server-side constraints:
- Roles: `user`, `assistant`, `system`
- Max messages/request: 500
- Max chars/message: 8,000
- Max total conversation chars: 32,000
- `temperature` ∈ [0.0, 2.0], `top_k` ∈ [1, 200], `max_tokens` ∈ [1, 4096]

## Demo
<video src="demo1.mp4" controls width="720"></video>

Direct link: [demo1.mp4](./demo1.mp4)

Build:
```
cargo build --release
```

Run tests (validates parity against reference fixtures):
```
cargo test -q
```

Run benchmarks (Criterion):
```
cargo bench --features metal
```
This benchmark uses the d20 version of the model
Reports will be written under `target/criterion/**/report/index.html`.

## Upstream tracking and correctness
The reference implementation lives under `reference/nanochat` (tracked from
`karpathy/nanochat`). Parity is tested via auto-generated fixtures under
`fixtures/`. Numerical tolerance for logits parity is 1e-4.

If you cloned with submodules, you can update them with:
```
git submodule update --init --remote --recursive
```

### Regenerate fixtures
This uses the Python reference via `uv` and writes JSON fixtures into `fixtures/`.
```
./gen-fixtures.sh
```

Tokenizer tests expect a tiktoken pickle at `reference/tokenizer.pkl` (a small
one is provided in the repo).

## Troubleshooting
- Slow or failing downloads: pre-download from HF and pass `--source /path/to/model_dir`
- UI not shown: ensure `reference/nanochat/ui.html` exists or pass `--ui-path`
- CUDA errors: verify driver/runtime and rebuild with `--features cuda`
- Apple GPU: prefer `--features metal` (default on macOS targets)

## High level roadmap
- SFT and RL(requires backward pass)
- (maybe) bring back the embedded python interpreter (or other context-free language like Lua) 
- Additional tensor backend other than Candle, encountered many candle kernel issues while building this.
- Pretraining (low priority, likely limited utility to do full training in Rust)

## License
MIT or Apache-2.0, at your option.

## Acknowledgements
- Andrej Karpathy for the original nanochat
- Hugging Face Candle team for the lightweight Rust tensor/NN stack
