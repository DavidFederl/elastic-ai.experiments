import random
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Capture the state of all random number generators.

    The numpy state is converted to plain lists, so ``torch.load`` can read the
    result with ``weights_only=True``.

    Returns:
        dict[str, Any]: State of the python, numpy, torch and cuda generators.
    """
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state(legacy=True)
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (
            name,
            np.asarray(keys).tolist(),
            position,
            has_gauss,
            cached_gaussian,
        ),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the state captured by `capture_rng_state`."""
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state(
        (name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gaussian)
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
