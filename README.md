# IES Delta-Compression Experiments

> [!NOTE]
> This repository contains the code for the experiments regarding
> "Delta-Compression" by [David P. Federl](mailto:david-peter.federl@uni-due.de).

## Setup

The project is bootstrapped with the tools [devenv](https://devenv.sh/) and [astral-uv](https://astral.sh/uv/).

All required dependencies will be installed automatically by devenv itself
utilizing astral-uv for Python package management.

> [!TIP]
> A more detailed information can be found in the [setup guide](./SETUP.md)!

## Run Experiment

The experiment runner provides a CLI user interfaces based on [click](https://click.palletsprojects.com/en/stable/).

```bash
$ uv run main.py --help

 Usage: main.py [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config                     PATH  Configuration file defining the experiment. [required] │
│    --resume     --no-resume           Resume after aborted execution. [default: resume]      │
│    --log-dir                    PATH  Log directory. [default: logs/1773433149541]           │
│    --verbose    --no-verbose          Enable verbose output. [default: no-verbose]           │
│    --help                             Show this message and exit.                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Configuration File

The application expects a configuration based on the YAML notation.
This configuration file is used to define the dataset, the model, and the
experiment to perform.

**Minimal Schema:**

> [!WARNING]
> Minimal required fields.
> If these fields are not present the experiment execution will fail!

```yaml
dataset:
  type: <dataset-name>

model:
  type: <model-name>

training:
  epochs: <epochs>

experiment:
  - type: <expression-name>
```

> [!TIP]
> Valid entries for the `type` fields can be retrieved from
> the [config_schema.py](./src/config/config_schema.py)

<details>

<summary>Max Schema</summary>

> [!IMPORTANT]
> The fields not in the minimal schema are optional and can be omitted.

```yaml
dataset:
  type: <datset-name>
  parameter:
    storage_path: <local-path>
    batch_size: <batch-size>

model:
  type: <model-name>
  parameter:
    fixed_point_total_bits: <total-bitwidth>
    fixed_point_fraction_bits: <fraction-bits>

training:
  epochs: <training-epochs>
  device: <cpu|cude|mps>
  loss: <cse|mse|...>
  optimizer: <adam|sgd|...>
  seed: <seed>
  store_only_last: <true|false>

experiment:
  type: <experiment-name>
  parameter:
    model_fixed_point_total_bits: <delta-bitwidth>
    model_fixed_point_fraction_bits: <delta-fraction-bits>
    delta_bit_width: <delta-bitwidth>
```

</details>

## Contributing

Have a look at the [CONTRIBUTING.md](./CONTRIBUTING.md) file!
