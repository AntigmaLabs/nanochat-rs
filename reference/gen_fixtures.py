"""
Generate test fixtures for cross-validation between Python and Rust implementations.

This script creates a small GPT model, runs a forward pass with sample inputs,
and saves the configuration, model weights, input tokens, and expected logits
to fixture files that can be used to test the Rust implementation.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "nanochat"))
from nanochat.gpt import GPT, GPTConfig

import json
import random
import torch
from util import print_tree
from dataclasses import dataclass, asdict
from typing import Any, Optional

@dataclass
class MetaConfig:
    model_config: dict[str, Any]
    tensors: list[str]

@dataclass
class TestCase:
    """Simple container for a test case.

    """

    name: str = "default"
    description: Optional[str] = None
    seed: int = 42
    input: Any = None
    temperature: float = 0.0
    top_k: Optional[int] = None



def create_tiny_config():
    """Create a tiny GPT configuration for testing."""
    return GPTConfig(
        sequence_len=64,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,  # Multi-Query Attention
        n_embd=64,
    )


def initialize_model_deterministically(model):
    """Initialize model weights with a deterministic seed (simple random init)."""
    # Use simple random initialization instead of init_weights() to get non-zero logits
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.uniform_(param, -0.1, 0.1)
    return model


def save_model_weights(model, output_dir):
    """Save model weights in .pt format."""
    state_dict = model.state_dict()

    # Save as .pt
    pt_path = output_dir / "model.pt"
    torch.save(state_dict, pt_path)
    print(f"Saved model weights (pt) to {pt_path}")



def generate_random_token_input_cases(
    random_seed: int,
    num_cases: int = 10,
    batch_size: int = 1,
    vocab_size: int = 32,
    len_range: tuple[int, int] = (1, 8),
) -> list[TestCase]:
    rng = random.Random(random_seed)

    cases: list[TestCase] = []
    for i in range(num_cases):
        length = rng.randint(len_range[0], min(len_range[1], vocab_size))
        batch_inputs = []
        for _ in range(batch_size):
            seq = [rng.randrange(vocab_size) for _ in range(length)]
            batch_inputs.append(seq)
        cases.append(
            TestCase(
                name=f"random_{i + 1:02d}_seed{random_seed}_len{length}_batch{batch_size}",
                seed=random_seed,
                input=batch_inputs,
            )
        )
    return cases


def gpt_logits(model, test_case: TestCase, dir: Path):
    """Run the model on input_ids and return a serialized fixture dict.

    Args:
        model: Initialized GPT model in eval mode.
        test_case: The test case to run.

    Returns:
        dict with keys: name, input_ids, expected_logits
    """
    input_tensor = torch.tensor(test_case.input, dtype=torch.long)

    print(f"\nGenerating fixtures: {dir.name}/{test_case.name}")
    print(f"  Input shape: {input_tensor.shape}")

    with torch.no_grad():
        logits = model.forward(input_tensor, targets=None, kv_cache=None)
        start_tokens = test_case.input[0]
        tokens = list(
            model.generate(
                start_tokens,
                max_tokens=1,
                temperature=test_case.temperature,
                top_k=test_case.top_k,
                seed=test_case.seed,
            )
        )

    print(f"  Output shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

    path = dir / f"{test_case.name}.json"
    fixture = {
        "name": test_case.name,
        "input": test_case.input,
        "logits": logits.tolist(),
        "seed": test_case.seed,
        "tokens": tokens,
        "temperature": test_case.temperature,
        "top_k": test_case.top_k,
    }
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)


def generate_fixtures(output_dir: str = "fixtures", seed: int = 42):
    """Generate test fixtures for input_tokens -> logits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create and initialize model
    config = create_tiny_config()
    print("Creating model")
    model = GPT(config)
    model = initialize_model_deterministically(model)
    model.eval()

    meta_config = MetaConfig(
        model_config=asdict(config),
        tensors=[name for name, _ in model.state_dict().items()],
    )
    print("Created meta config:\n" + json.dumps(asdict(meta_config), indent=2))

    # Persist meta and model weights
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(asdict(meta_config), f, indent=2)
    print(f"Saved meta to {meta_path}")

    save_model_weights(model, output_dir)

    logits_dir = output_dir / "gpt_logits"
    logits_dir.mkdir(exist_ok=True)
    # Generate deterministic test cases from provided seed
    cases = generate_random_token_input_cases(seed, num_cases=10, batch_size=1)
    # Upstream generate() treats top_k=0 as "disable filtering".
    # Keep one fixture on this code path to catch regressions.
    if cases:
        base = cases[0]
        cases.append(
            TestCase(
                name=f"{base.name}_topk0",
                seed=base.seed,
                input=base.input,
                temperature=0.0,
                top_k=0,
            )
        )

    for test_case in cases:
        gpt_logits(model, test_case, logits_dir)
    # Print directory tree for quick inspection
    print("\nOutput directory tree:")
    print_tree(output_dir)



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test fixtures for GPT model testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../fixtures",
        help="Output directory for fixtures (default: ../fixtures)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic weight initialization (default: 42)",
    )

    args = parser.parse_args()

    # Set global seed for reproducibility
    torch.manual_seed(args.seed)

    generate_fixtures(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
