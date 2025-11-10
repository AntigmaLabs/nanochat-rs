import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch


def _is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, (dict, OrderedDict)):
        return False
    if len(obj) == 0:
        return True
    return all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items())


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    # Direct state_dict
    if _is_state_dict_like(obj):
        return dict(obj)  # make a plain dict

    # torch.nn.Module
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()

    # Common checkpoint layouts
    if isinstance(obj, dict):
        # Try common keys that may hold the state dict or a module
        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "weights",
            "params",
            "model",
            "module",
            "gpt",
            "network",
        ]
        for key in candidate_keys:
            if key in obj:
                sub = obj[key]
                if _is_state_dict_like(sub):
                    return dict(sub)
                if isinstance(sub, torch.nn.Module):
                    return sub.state_dict()
                # Some frameworks store an inner dict under the candidate key
                if isinstance(sub, dict):
                    # Heuristic: look one more level deep for a state-dict-like mapping
                    for _, maybe in sub.items():
                        if _is_state_dict_like(maybe):
                            return dict(maybe)

    raise ValueError(
        "Could not extract a state dict from the provided checkpoint. "
        "Expected a torch.nn.Module or a dict-like mapping of str -> Tensor."
    )


def _move_to_cpu_contiguous(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State dict entry '{name}' is not a Tensor: {type(tensor)}")
        # Ensure cpu and contiguous memory layout for saving
        cpu_tensor = tensor.detach().to("cpu").contiguous()
        out[name] = cpu_tensor
    return out


def convert_pt_to_safetensors(input_path: Path, output_path: Path, strict: bool = True) -> None:
    try:
        from safetensors.torch import save_file  # local import for clearer error if missing
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "The 'safetensors' package is required. Install with: pip install safetensors"
        ) from e

    checkpoint = torch.load(str(input_path), map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _move_to_cpu_contiguous(state_dict)

    # Optional sanity check: values must be tensors on CPU
    if strict:
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Key '{k}' has non-tensor value of type {type(v)}")
            if v.device.type != "cpu":
                raise AssertionError(f"Key '{k}' is not on CPU: {v.device}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Add minimal metadata
    metadata = {
        "format": "pt->safetensors",
        "framework": "pytorch",
    }
    save_file(state_dict, str(output_path), metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .pt checkpoint to .safetensors"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input .pt file (state_dict or checkpoint).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        help="Path to output .safetensors file. Defaults to input path with .safetensors extension.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict validation checks before saving.",
    )

    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_suffix(".safetensors")

    convert_pt_to_safetensors(input_path, output_path, strict=not args.no_strict)
    print(f"Converted '{input_path}' -> '{output_path}'")


if __name__ == "__main__":
    main()


