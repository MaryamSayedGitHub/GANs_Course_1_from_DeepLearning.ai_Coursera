# Basic GAN on MNIST

A simple fully-connected Generative Adversarial Network (GAN) trained on the MNIST dataset, based on DeepLearning.ai's "Build a Basic GAN" course.

## What this does

The script trains two competing neural networks:

- **Generator** — takes a random noise vector and tries to turn it into a realistic-looking handwritten digit image.
- **Discriminator** — looks at an image (real or generated) and tries to tell whether it's real or fake.

Both networks improve together: the Generator gets better at fooling the Discriminator, and the Discriminator gets better at catching fakes.

## Requirements

```bash
pip install torch torchvision tqdm matplotlib
```

## Usage

```bash
python GAN.py
```

or, if using `uv`:

```bash
uv run GAN.py
```

The MNIST dataset will be downloaded automatically on first run.

## Output

While training, the script creates a `training_results/` folder:

```
training_results/
├── training_log.csv          # loss values logged every 50 epochs
└── sample_images/
    ├── epoch_0.png            # generated image samples every 50 epochs
    ├── epoch_50.png
    ├── epoch_100.png
    └── ...
```

### `training_log.csv` columns

| Column               | Meaning                                    |
| -------------------- | ------------------------------------------ |
| `epoch`              | Current training epoch                     |
| `step`               | Cumulative batch step across all epochs    |
| `generator_loss`     | Average Generator loss at logging time     |
| `discriminator_loss` | Average Discriminator loss at logging time |
| `timestamp`          | When this row was logged                   |

### Reading the results

```python
import pandas as pd
df = pd.read_csv("training_results/training_log.csv")
df.plot(x="epoch", y=["generator_loss", "discriminator_loss"])
```

**How to interpret the losses:**

- Both losses should hover around a similar value (roughly 0.6–0.7) without drifting too far apart.
- If the discriminator loss drops close to 0, it's winning too easily — the generator isn't learning fast enough.
- If the discriminator loss rises a lot instead, it's being fooled too easily — training may be unstable.
- The sample images in `sample_images/` are the more intuitive signal: digits should look progressively more recognizable as training progresses.

## Configuration

Key hyperparameters (top of `GAN.py`):

| Variable             | Default | Description                                      |
| -------------------- | ------- | ------------------------------------------------ |
| `n_epochs`           | 200     | Number of training epochs                        |
| `z_dim`              | 64      | Size of the noise vector                         |
| `batch_size`         | 128     | Images per training batch                        |
| `lr`                 | 0.00001 | Learning rate for both optimizers                |
| `display_step`       | 500     | How often to print live progress to console      |
| `LOG_EVERY_N_EPOCHS` | 50      | How often to save results to `training_results/` |

Runs on GPU automatically if `torch.cuda.is_available()` is `True`, otherwise falls back to CPU.
