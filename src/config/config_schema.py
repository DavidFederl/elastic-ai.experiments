from schema import And, Optional, Schema


def get_config_schema() -> Schema:
    """Return the schema for the configuration file to perform validation."""
    return Schema(
        {
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
                Optional("loss"): str,
                Optional("optimizer"): str,
                Optional("device"): And(str, lambda d: d in ["cpu", "mps", "cuda"]),
                "epochs": int,
                Optional("store_only_last"): bool,
            },
        },
        ignore_extra_keys=True,
    )
