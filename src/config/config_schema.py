from schema import Optional, Schema


def get_config_schema() -> Schema:
    """Return the schema for the configuration file to perform validation."""
    return Schema(
        {
            "dataset": {
                "type": str,
                Optional("storage_path"): str,
                Optional("batch_size"): int,
            },
        },
        ignore_extra_keys=True,
    )
