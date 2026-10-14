"""Shared device selection for training, evaluation and inference."""
import torch


def resolve_device(device="auto"):
    if str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Supported devices: auto, cpu, mps, cuda or cuda:N") from error
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu, mps or auto")
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {selected.index} is unavailable")
    elif selected.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Apple MPS is unavailable; use --device cpu or auto")
    elif selected.type != "cpu":
        raise ValueError("Supported devices: auto, cpu, mps, cuda or cuda:N")
    return selected


def empty_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
