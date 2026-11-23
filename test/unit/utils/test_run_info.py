import json

import yaml

from src.utils import get_versions, write_run_info


def test_get_versions_reports_packages_and_python():
    versions = get_versions()

    assert versions["packages"]["torch"] is not None
    assert versions["python"].count(".") == 2
    assert set(versions) >= {"git_commit", "git_dirty", "cuda", "gpus", "hostname"}


def test_write_run_info_writes_config_and_versions(tmp_path):
    config = {"dataset": "/ws/cifar10.h5", "seed": 42}

    write_run_info(tmp_path / "run", config)

    assert yaml.safe_load((tmp_path / "run" / "config.yaml").read_text()) == config
    versions = json.loads((tmp_path / "run" / "versions.json").read_text())
    assert "packages" in versions
