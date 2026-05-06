{ pkgs, lib, config, inputs, ... }:

{
  packages = [
    pkgs.git
  ];

  languages.python = {
    enable = true;
    version = "3.12";
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
    "tests:unit-tests" = {
       exec = ''
         uv run pytest
       '';
    };
  };

  # See full reference at https://devenv.sh/reference/options/
}
