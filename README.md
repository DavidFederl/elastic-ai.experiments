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

Experiments can be found in the [experiments folder](./experiments/).
Each file has different inputs that can be determined via the `--help` flag

**EXAMPLE:**

```bash
$ uv run src/experiments/delta_compression/dc_experiment_01.py --help

 Usage: dc_experiment_01.py [OPTIONS]

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --log-dir                                      PATH                  Log directory. [default: logs/17781732390079]     │
│ --verbose                      --no-verbose                          Enable verbose output. [default: no-verbose]      │
│ --epochs                                       INTEGER RANGE [x>=1]  Epochs for Training. [default: 100]               │
│ --fixed-point-total-bits                       INTEGER RANGE [x>=1]  [default: 16]                                     │
│ --fixed-point-fraction-bits                    INTEGER RANGE [x>=1]  [default: 8]                                      │
│ --install-completion                                                 Install completion for the current shell.         │
│ --show-completion                                                    Show completion for the current shell,            │
│                                                                       to copy it or customize the installation.        │
│ --help                                                               Show this message and exit.                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Contributing

Have a look at the [CONTRIBUTING.md](./CONTRIBUTING.md) file!
