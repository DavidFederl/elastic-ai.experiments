# Setup Guide

## Prerequisites

- Bash
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/es-ude/experiment-runner.git
cd experiment-runner
```

### 2. Set Up Environment Using Devenv

This project includes a `.devenv.nix` configuration for easy
development environment setup using [devenv.sh](https://devenv.sh/).

#### Setup with Devenv

```bash
# Install devenv.sh if you don't have it
curl -fsSL https://devenv.sh/install.sh | sh

# Enter the development environment
devenv shell
```

> [!WARNING]
> The first time you run `devenv shell`, it may take a few minutes to download
> and set up the Docker container.

> [!INFORMATION]
> Devenv will automatically install [astral-uv](https://astral.sh/uv/)
> and setup a virtual environment providing python and its required packages.

#### Devenv Features

- **Automatic dependency installation**: All dependencies are installed automatically
- **Isolated environment**: Clean development environment
- **Pre-configured tools**: Python, UV, and other development tools
- **Easy cleanup**: Exit the shell to leave the environment

#### Devenv Commands

```bash
# Start a shell in the development environment
devenv shell

# Print information about the current environment
devenv info

# Update installed dependencies
devenv update
```

## Project Structure

```
experiment-runner/
├── src/                  # Main source code
│   ├── nn/               # Neural network components
│   ├── experiments/      # Experiment implementations
│   ├── tools/            # Utility tools (e.g., graph generation)
│   └── config/           # Configuration management
├── tests/                # Test files
├── pyproject.toml        # Project configuration
├── main.py              # Main entry point
├── SETUP.md             # This setup guide
└── README.md             # Project overview
```

### Getting Help

For additional help, check the project documentation or open an issue on GitHub.
