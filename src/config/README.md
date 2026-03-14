# Configuration

The application expects a configuration based on the YAML notation.
This configuration file is used to define the dataset, the model, and the
experiment to perform.

## Minimal Schema

The minimal schema describes the minimal necessary fields for the configuration.
These fields have to be present at all time!

```yaml
dataset:
  type: <str>

model:
  type: <str>

training:
  epochs: <unsinged int>

experiment:
  - type: <str>
```

## Extended Schema

In addition to the minimal schema,
the application supports more configuration options.
These options however have defaults and are therefore not required to be present.

```yaml
dataset:
  type: <str>
  parameter:
    storage_path: <str>
    batch_size: <unsinged int>

model:
  type: <str>
  parameter:
    fixed_point_total_bits: <unsinged int>
    fixed_point_fraction_bits: <unsinged int>

training:
  epochs: <unsinged int>
  device: <str>
  loss: <str>
  optimizer: <str>
  seed: <unsinged int>
  store_only_last: <bool>

experiment:
  type: <str>
  parameter:
    model_fixed_point_total_bits: <unsinged int>
    model_fixed_point_fraction_bits: <unsinged int>
    delta_bit_width: <unsinged int>
```

## Configuration Fields

### Dataset Configuration

The dataset block defines the dataset to use for the experiment.
The application currently supports the `FashionMNIST` dataset.

In addition to these a custom path to store the downloaded dataset can be
provided with the `storage_path` parameter.

The batch size can be configured with the `batch_size` parameter and is used
for the Dataloader instances provided by the dataset class.

### Model Configuration

The model block defines the neural network model to use for the experiment.
The application currently provides

- the `linear_v1_eai` model based on the elastic-AI.creator Fixed-Point layers and
- the `linear_v1_torch` model based on standard PyTorch layers.

For elastic-AI.creator model additional parameters are supported.
Under the `parameter` block the following parameters can be configured:

- `fixed_point_total_bits`: The total number of bits for the fixed-point representation.
- `fixed_point_fraction_bits`: The number of bits for the fraction part of the
  fixed-point representation.

### Training Configuration

For the training configuration the following parameters can be configured:

- `epochs`: The number of training epochs.
- `device`: The device (`cpu` | `cuda` | `mps`) to use for training.
- `loss`: The loss function to use for training (`cse` | `mse`).
- `optimizer`: The optimizer to use for training (`adam` | `sgd`).
- `seed`: The seed to use for training.
- `store_only_last`: If `true` only the last batch of the training data is stored.

### Experiment Configuration

For the experiment configuration the following parameters can be configured:

- `type`: The type of the experiment to perform.
- `parameter`: Additional parameters for the experiment.
  - `model_fixed_point_total_bits`: The total number of bits for the fixed-point
    representation of the model.
  - `model_fixed_point_fraction_bits`: The number of bits for the fraction part
    of the fixed-point representation of the model.
  - `delta_bit_width`: The number of bits for the delta representation.

> [!TIP]
> With the `*<key>` syntax in YAML a different key can be referenced.
> This key needs to be defined with the `&<key> <value>` syntax.
> With this feature the configuration parameter from the model section can be referenced:
>
> ```yaml
> model:
>   parameter:
>     fixed_point_total_bits: &model_total_bits 16
>     fixed_point_fraction_bits: &model_fraction_bits 8
>
> experiment:
>   parameter:
>     model_fixed_point_total_bits: *model_total_bits
>     model_fixed_point_fraction_bits: *model_fraction_bits
> ```

## Troubleshooting

A more detailed information alongside parameter constraints can be found in the
[config_schema.py](./src/config/config_schema.py) file.
