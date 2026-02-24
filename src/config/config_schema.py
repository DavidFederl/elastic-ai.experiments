from schema import And, Optional, Schema


def get_config_schema() -> Schema:
    """Return the schema for the configuration file to perform validation."""
    return Schema(
        schema={
            "dataset": {
                "type": str,
                Optional("parameter"): {
                    Optional("storage_path"): str,
                    Optional("batch_size"): int,
                },
            },
            "model": {
                "type": And(str, lambda t: t in ["linear_v1"]),
                Optional("parameter"): {
                    Optional("fixed_point_total_bits"): int,
                    Optional("fixed_point_fraction_bits"): int,
                },
            },
            "training": {
                Optional("seed"): And(int, lambda s: s > 0, lambda s: s < 2**32),
                Optional("loss"): str,
                Optional("optimizer"): str,
                Optional("device"): And(str, lambda d: d in ["cpu", "mps", "cuda"]),
                "epochs": int,
                Optional("store_only_last"): bool,
            },
            "experiments": [
                {
                    "type": str,
                    Optional("parameter"): {
                        Optional("fixed_point_total_bits"): int,
                        Optional("fixed_point_fraction_bits"): int,
                    },
                }
            ],
        },
        ignore_extra_keys=True,
    )
