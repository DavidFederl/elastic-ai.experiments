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
  };

  scripts.run_experiments = {
    exec = ''
      for config_file in "$1"/*.yaml; do
        $DEVENV_ROOT/scripts/run_experiment.sh $config_file
      done
    '';
  };

  # See full reference at https://devenv.sh/reference/options/
}
