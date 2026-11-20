from pathlib import Path

import typer

from experiments.training.smartable_adam import Device, fixed_point, floating_point

app = typer.Typer()


@app.command()
def fp32() -> None:
    for experiment_id in range(0, 100):
        log_dir = Path(f"logs/aiot26/fp32/{experiment_id}")
        floating_point(
            log_dir=log_dir,
            verbose=True,
            epochs=100,
            batch_size=128,
            seed=experiment_id,
            device=Device.MPS,
            early_stopping=(10, 0.01),
        )


@app.command()
def fxp() -> None:
    for fractional_bits in range(0, 8):
        for experiment_id in range(0, 10):
            log_dir = Path(
                f"logs/aiot26/fxp/Q{8 - 1 - fractional_bits}.{fractional_bits}/{experiment_id}"
            )
            fixed_point(
                log_dir=log_dir,
                verbose=True,
                epochs=50,
                batch_size=128,
                seed=experiment_id,
                device=Device.MPS,
                early_stopping=(10, 0.01),
            )


if __name__ == "__main__":
    app()
