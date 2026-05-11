{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  packages = [
    pkgs.git
    pkgs.cocogitto
  ];

  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    uv.enable = true;
    uv.sync.enable = true;
    uv.sync.allGroups = true;
  };

  scripts = {
    fp_training = {
      exec = ''
        SCRIPT_DIR=$(dirname "$(realpath "$0")")
        EXPERIMENT=''${DEVENV_ROOT}/experiments/training/mlp_fp_fashionmnist.py

        for run in {0..99}; do
          uv run "''${EXPERIMENT}" \
            --log-dir "logs/$1/fp_bs512/''${run}/" \
            --verbose \
            --epochs 100 \
            --batch-size 512
        done
      '';
      binary = "bash";
      description = "Run fixed-point training (100 times; 100 epochs), Input: log-prefix";
    };
    fxp_training = {
      exec = ''
        EXPERIMENT=''${DEVENV_ROOT}/experiments/training/mlp_fxp_fashionmnist_adam.py

        echo "Running: $EXPERIMENT"

        for run in {0..99}; do
          uv run "''${EXPERIMENT}" \
            --log-dir "logs/$2/fxp_8bit_bs512/$run" \
            --verbose \
            --epochs 100 \
            --batch-size 512 \
            --total-fixed-point-bits $1
        done
      '';
      binary = "bash";
      description = "Run fixed-point training (100 times; 100 epochs). Input: total-fixed-point-bits, log-prefix";
    };
    consecutive_delta_training = {
      exec = ''
        EXPERIMENT=''${DEVENV_ROOT}/experiments/training/mlp_delta_fashionmnist_adam.py

        echo "Running: $EXPERIMENT"

        for run in {0..99}; do
          uv run "''${EXPERIMENT}" \
            --log-dir "logs/$5/runs/q$(( $1 - $2 - 1 )).$2_d$3o$4_consecutive_bs512/$run" \
            --verbose \
            --epochs 100 \
            --batch-size 512 \
            --total-fixed-point-bits $1 \
            --fraction-bits $2 \
            --delta-bits $3 \
            --delta-offset $4 \
            --delta-type consecutive
        done
      '';
      binary = "bash";
      description = "Run delta aware training for given fraction-bits (100 times; 100 epochs). Input: total-fixed-point-bits, fraction-bits, delta-bits, delta-offset, log-prefix";
    };
    fixed_delta_training = {
      exec = ''
        EXPERIMENT=''${DEVENV_ROOT}/experiments/training/mlp_delta_fashionmnist_adam.py

        echo "Running: $EXPERIMENT"

        for run in {0..99}; do
          uv run "''${EXPERIMENT}" \
            --log-dir "logs/$5/runs/q$(( $1 - $2 - 1 )).$2_d$3o$4_fixed_bs512/$run" \
            --verbose \
            --epochs 100 \
            --batch-size 512 \
            --total-fixed-point-bits $1 \
            --fraction-bits $2 \
            --delta-bits $3 \
            --delta-offset $4 \
            --delta-type fixed-reference
        done
      '';
      binary = "bash";
      description = "Run delta aware training for given fraction-bits (100 times; 100 epochs). Input: total-fixed-point-bits, fraction-bits, delta-bits, delta-offset, log-prefix";
    };
  };

  tasks = {
    "check:conventional-commit" = {
      exec = ''
        if [ -n "$CI" ]; then
          ${pkgs.cocogitto}/bin/cog check ..$GITHUB_SOURCE_REF
        else
          ${pkgs.cocogitto}/bin/cog check main..
        fi
      '';
    };
    "check:python-lint" = {
      exec = ''
        uv run ruff format --check
      '';
      before = [ "check:code-lint" ];
    };
    "check:types" = {
      exec = ''
        uv run ty check src/
      '';
      before = [ "check:code-lint" ];
    };
    "check:code-lint" = {
      after = [
        "check:python-lint"
        "check:types"
      ];
    };
    "check:pytest" = {
      exec = ''
        uv run pytest
      '';
    };
    "check:unit-tests" = {
      after = [
        "check:pytest"
      ];
    };
  };

  # See full reference at https://devenv.sh/reference/options/
}
