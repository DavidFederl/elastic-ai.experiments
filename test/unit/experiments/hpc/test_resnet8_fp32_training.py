import re
from pathlib import Path

import pytest

from src.utils import load_config

JOB_DIR = Path(__file__).parents[4] / "experiments" / "hpc" / "resnet8_fp32_training"


def _cpus_per_task(script: Path) -> int:
    match = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", script.read_text(), re.M)
    assert match is not None, f"no --cpus-per-task in {script}"
    return int(match.group(1))


@pytest.mark.parametrize("name", ["config.yaml", "config_test.yaml"])
def test_num_workers_equals_cpus_per_task(name):
    """Each data loading worker needs its own CPU of the Slurm job."""
    config = load_config(JOB_DIR / name)

    assert isinstance(config, dict)
    assert config["num_workers"] == _cpus_per_task(JOB_DIR / "run_training.sh")


def test_job_array_has_one_unique_seed_per_task():
    script = (JOB_DIR / "run_training.sh").read_text()

    seeds = re.search(r"^SEEDS=\(([\d ]+)\)$", script, re.M)
    array = re.search(r"^#SBATCH --array=0-(\d+)$", script, re.M)

    assert seeds is not None and array is not None
    values = seeds.group(1).split()
    assert len(values) == int(array.group(1)) + 1
    assert len(set(values)) == len(values)
