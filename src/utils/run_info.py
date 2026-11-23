import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml

PACKAGES = ("elasticai-creator", "torch", "torchvision", "numpy", "h5py")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    """Version of a package, with the commit for packages installed from git."""
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    version = distribution.version
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is not None:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if commit is not None:
            version = f"{version} ({commit})"
    return version


def get_versions() -> dict[str, Any]:
    """Collect what is needed to reproduce a run.

    Returns:
        dict[str, Any]: Git commit, python, package and hardware versions.
    """
    # untracked files (for example Slurm output files) do not change the code
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "cuda": torch.version.cuda,
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
    }


def write_run_info(log_dir: Path, config: dict[str, Any]) -> None:
    """Write the config and the versions of a run into the log directory.

    Args:
        log_dir (Path): Log directory of the run.
        config (dict[str, Any]): Settings of the run, with all variables expanded.
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    with log_dir.joinpath("config.yaml").open("w") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    with log_dir.joinpath("versions.json").open("w") as versions_file:
        json.dump(get_versions(), versions_file, indent=2)
