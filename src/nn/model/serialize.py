import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_from_json(path: Path) -> dict:
    def _dtype_from_string(dtype_name: str) -> torch.dtype:
        normalized = dtype_name.replace("torch.", "", 1)
        dtype = getattr(torch, normalized, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported dtype '{dtype_name}'")
        return dtype

    state_dict: dict = {}
    try:
        with open(path, "r") as file:
            json_state_dict = json.load(file)
        for name, payload in json_state_dict.items():
            dtype = _dtype_from_string(payload["dtype"])
            values = payload["values"]
            shape = payload["shape"]
            state_dict[name] = torch.tensor(values, dtype=dtype).reshape(shape)
    except FileNotFoundError:
        logger.error(f"Could not load parameters from {path}: File not found!")
    except PermissionError:
        logger.error(f"Could not load parameters from {path}: Permission denied!")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error(f"Could not load parameters from {path}: {exc}")

    return state_dict


def save_as_json(state_dict: dict, path: Path) -> None:
    def _state_dict_to_json(state_dict: dict) -> dict:
        json_dict: dict = {}

        for name, tensor in state_dict.items():
            json_dict[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "values": tensor.detach().cpu().tolist(),
            }

        return json_dict

    try:
        with open(path, "w") as file:
            json_state_dict: dict = _state_dict_to_json(state_dict)
            json.dump(json_state_dict, file)
    except FileNotFoundError:
        logger.error(f"Could not store parameters to {path}: File not found!")
    except PermissionError:
        logger.error(f"Could not store parameters to {path}: Permission denied!")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error(f"Could not store parameters to {path}: {exc}")
