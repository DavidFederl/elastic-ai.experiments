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
> A more detailed information can be found in [configuration README](./src/config/README.md)!

## Contributing

Have a look at the [CONTRIBUTING.md](./CONTRIBUTING.md) file!
