# IES Delta-Compression Experiments

> [!NOTE]
> This repository contains the code for the experiments regarding
> "Delta-Compression" by [David P. Federl](mailto:david-peter.federl@uni-due.de).

## Setup

The project is based on [devenv](https://devenv.sh/) and [astral-uv](https://astral.sh/uv/).

All required dependencies will installed automatically by devenv itself
utilizing astral-uv for Python packages.

## Run Experiment

The experiment runner provides a CLI user interfaces based on [click](https://click.palletsprojects.com/en/stable/).

```bash
$ uv run main.py --help

Usage: main.py [OPTIONS]

Options:
  --config PATH             Configuration file defining the experiment.
                            [required]
  --debug / --no-debug
  --verbose / --no-verbose
  --log-dir TEXT
  --help                    Show this message and exit.
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
```

**Max Schema:**

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
    fixed_point_total_bits: <bitwidth>
    fixed_point_fraction_bits: <fraction-bits>

training:
  loss: <cse|mse|...>
  optimizer: <adam|sgd|...>
  device: <cpu|cude|mps>
  epochs: <training-epochs>
  store_only_last: <true|false>
```
