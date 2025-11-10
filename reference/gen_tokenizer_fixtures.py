"""
Generate tokenizer fixtures for cross-validation between Python and Rust implementations.

This script loads the tokenizer encoding (tiktoken.Encoding from tokenizer.pkl), and produces
individual TokenTest JSON files under <output-dir>/tokenizer/ with fields:
- name: str
- text: str
- tokens: list[int]
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any
import random
import string

import tiktoken

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from util import print_tree  # noqa: E402


def load_encoding(tokenizer_pkl: str) -> tiktoken.Encoding:
    pkl_path = Path(tokenizer_pkl)
    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)
    if not isinstance(enc, tiktoken.Encoding):
        raise TypeError(f"Expected tiktoken.Encoding in {pkl_path}, got {type(enc)}")
    return enc


def default_cases() -> list[str]:
    return [
        "",
        "hello",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Sphinx of black quartz, judge my vow!",
        "  leading and trailing  ",
        "Newlines\nand\ttabs\t",
        "Quotes: \"' backslashes \\",
        "Symbols: !@#$%^&*()_+-=[]{}|;:,.<>/?",
        "URL: https://example.com/path?x=1&y=2",
        "email: test@example.com",
        "Numbers: 0 1 2 3 10 42 007 3.1415",
        "🐍 Python 3.10",
        "こんにちは",
        "Mixed 中英 text",
    ]


def generate_random_text_cases(
    seed: int,
    num_cases: int,
    len_range: tuple[int, int] = (0, 64),
) -> list[str]:
    rng = random.Random(seed)
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "quick", "brown", "fox", "lazy", "dog",
        "rust", "python", "token", "encode", "decode", "model", "nano", "chat", "gpt", "kv",
        "apple", "banana", "carrot", "delta", "echo", "foxtrot",
    ]
    extras = ["🚀", "✨", "🔥", "💡", "汉字", "Καλημέρα", "مرحبا"]
    punct = list(".,;:!?-—()[]{}'\"")

    cases: list[str] = []
    for _ in range(num_cases):
        length = rng.randint(len_range[0], len_range[1])
        tokens: list[str] = []
        for i in range(length):
            choice = rng.random()
            if choice < 0.70:
                tokens.append(rng.choice(words))
            elif choice < 0.80:
                n = rng.randint(0, 9999)
                tokens.append(str(n))
            elif choice < 0.90:
                tokens.append(rng.choice(extras))
            else:
                # short random ascii chunk
                k = rng.randint(1, 6)
                tokens.append(''.join(rng.choice(string.ascii_letters) for _ in range(k)))
            if rng.random() < 0.2:
                tokens.append(rng.choice(punct))
        # Join tokens with spaces, then randomly insert some punctuation tight to words
        text = ' '.join(tokens)
        if rng.random() < 0.3:
            text = text.replace(" ", "")
        if rng.random() < 0.3:
            text = text.strip()
        cases.append(text)
    return cases


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def generate_fixtures(
    output_dir: str = "../fixtures",
    tokenizer_pkl: str | None = None,
    random_cases: int = 10,
    seed: int = 42,
) -> None:
    out_root = Path(output_dir)
    tok_out = out_root / "tokenizer"
    tok_out.mkdir(parents=True, exist_ok=True)

    enc = load_encoding(Path(tokenizer_pkl) if tokenizer_pkl else None)

    # Emit one JSON per case: TokenTest{name, text, tokens}
    known = default_cases()
    for i, text in enumerate(known, start=1):
        ids = [int(t) for t in enc.encode_ordinary(text)]
        case = {"name": f"known_{i:02d}", "text": text, "tokens": ids}
        write_json(tok_out / f"known_{i:02d}.json", case)

    random_texts = generate_random_text_cases(seed=seed, num_cases=random_cases)
    for i, text in enumerate(random_texts, start=1):
        ids = [int(t) for t in enc.encode_ordinary(text)]
        case = {"name": f"random_{i:02d}_seed{seed}", "text": text, "tokens": ids}
        write_json(tok_out / f"random_{i:02d}_seed{seed}.json", case)

    print("\nTokenizer fixtures written to:")
    print_tree(tok_out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate tokenizer fixtures (tiktoken)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../fixtures",
        help="Output directory for fixtures (default: ../fixtures)",
    )
    parser.add_argument(
        "--tokenizer-pkl",
        type=str,
        default="../fixtures/tokenizer.pkl",
        help="Path to tokenizer.pkl (defaults to reference/tokenizer.pkl)",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=10,
        help="Number of random text cases to generate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generating varied texts (default: 42)",
    )

    args = parser.parse_args()
    generate_fixtures(
        output_dir=args.output_dir,
        tokenizer_pkl=args.tokenizer_pkl,
        random_cases=args.random,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
