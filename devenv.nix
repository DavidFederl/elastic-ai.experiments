{ pkgs, lib, config, inputs, ... }:

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

  scripts.run_experiments = {
    exec = ''
      for config_file in "$1"/*.yaml; do
        $DEVENV_ROOT/scripts/run_experiment.sh $config_file
      done
    '';
  };

  tasks ={
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
      before = ["check:code-lint"];
    };
    "check:types" = {
      exec = ''
        uv run ty check src/
      '';
      before = ["check:code-lint"];
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
