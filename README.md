Tiny GPT-style core reimplemented in pure Rust, tracking
[karpathy/nanochat](https://github.com/karpathy/nanochat). Built on
[candle](https://github.com/huggingface/candle), focused on clarity and parity with
the reference while keeping the code minimal.


## Features
- Native rust
- Integration with HuggingFace
- Centralized model loader resilient to tensor name changes
- Minimal surface area to keep cognitive load low (not production-grade, performance-readability trade-off)
- Compatible with tiktoken `.pkl` tokenizer configs

### Main difference with the referenced nanochat
- Tokenizer encoding/decoding is production ready, so it is unified without dependency on tiktoken
- Removed the embeded python interpreter in engine
- Some performance and ergonomic improvement on the generation logic. 
- More emphasis on post training 


## Quick start
we have a 32 layer version trained with about $1000 budget at hugging face: https://huggingface.co/Antigma/nanochat-d32
there is also a smaller 20 layer version d20 used for benchmark and testing within the same HuggingFace repo

run on Apple With GPU
```
cargo run --release --features metal -- -p "write 100 words"
```

with Cuda
```
cargo run --release --features cuda -- -p "write 100 words"

```

## Demo

<video src="./demo1.mov" controls playsinline muted loop width="720"></video>

Direct link: [demo1.mov](./demo1.mov)

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

## High level roadmap
- A webserver to match original pythong reference
- SFT and RL(requires backward pass)
- (maybe) bring back the embedded python(or any context free language like Lua) 
- Additional tensor backend other than Candle
- Pretraining (low priority, likely limited utility to do full training in Rust)

## License
MIT or Apache-2.0, at your option.

## Acknowledgements
- Andrej Karpathy for the original nanochat
- Hugging Face Candle team for the lightweight Rust tensor/NN stack
