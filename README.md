# Fraud Detection Using Variational AutoEncoders

Unsupervised credit-card fraud detection using reconstruction-error anomaly
detection. Autoencoders and Variational Autoencoders (VAEs) are trained **only on
legitimate transactions**, so that fraudulent transactions — never seen during
training — produce a high reconstruction error and can be flagged by thresholding.

## Contents

| Notebook | Models |
| --- | --- |
| `AutoEncoder.ipynb` | Two plain autoencoders (shallow and deeper) |
| `VariationalAutoEncoder.ipynb` | Two variational autoencoders (shallow and deeper) |

## Dataset

The [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
dataset (ULB). 284,807 transactions with 492 frauds (~0.17%), 30 numeric
features: `Time`, `Amount`, and 28 PCA components `V1`–`V28`, plus the `Class`
label (0 = legitimate, 1 = fraud).

The notebooks load it from `/content/drive/MyDrive/creditcard.csv` (Google Colab).
Update that path to point at your local copy of `creditcard.csv`.

## Approach

### Preprocessing
- Separate features `X` from the label `y` (`Class`).
- Log-scale `Time` and `Amount` with `log1p`.
- Scale all features to `[0, 1]` with `MinMaxScaler`.
- **Train set:** non-fraud transactions only (with a tiny 0.1% validation split,
  also non-fraud).
- **Test set:** all remaining non-fraud transactions + all fraud transactions.

### Training
- Models are trained to reconstruct their input (`X → X`).
- Optimizer: Adam. Epochs: 100. Batch size: 32.
- Plain AEs use MSE loss; VAEs use MSE reconstruction loss + KL divergence,
  with the reparameterization ("sampling") trick.

### Detection
- Compute per-transaction reconstruction error (mean squared error) on the
  training data.
- Threshold = `mean(train_errors) + k * std(train_errors)`
  (`k = 2.5` for the plain AEs, `k = 1.5` for the VAEs).
- A test transaction is flagged as fraud when its reconstruction error exceeds
  the threshold.
- `AutoEncoder.ipynb` also plots the reconstruction-error distribution against
  the threshold.

## Model architectures

All layers are `Dense`. Input dimension = 30.

**Autoencoder 1 (shallow):** `30 → 14 (tanh) → 30 (tanh)`

**Autoencoder 2 (deeper):** `30 → 14 (tanh) → 10 (tanh) → 14 (tanh) → 30 (tanh)`

**VAE 1 (shallow):** encoder `30 → 64 (relu) → z_mean/z_log_var (14)`;
decoder `14 → 64 (relu) → 30 (sigmoid)`

**VAE 2 (deeper):** encoder `30 → 20 (relu) → 12 (relu) → z_mean/z_log_var (12)`;
decoder `12 → 12 (relu) → 20 (relu) → 30 (sigmoid)`

## Results (test set)

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Autoencoder 1 | 0.990 | 0.793 | 0.880 |
| Autoencoder 2 | 0.993 | 0.829 | 0.904 |
| VAE 1 | 0.969 | 0.829 | 0.894 |
| VAE 2 | 0.969 | 0.829 | 0.894 |

The deeper plain autoencoder gave the best F1 in these experiments.

## Requirements

- Python 3
- `tensorflow` / `keras`
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Usage

1. Download `creditcard.csv` and update the `pd.read_csv(...)` path in each
   notebook.
2. Open a notebook in Jupyter or Colab and run all cells.
3. Each notebook trains its two models and prints precision, recall, and F1 on
   the test set.

## Team

- Vadapalli Sai Sravan (CS24MTECH02007)
- Supreet Shukla (CS24MTECH02004)
- Tarun Jangir (CS24MTECH02005)
- Taufique Ramzan Shaikh (CS24MTECH02006)
- Afzaal Ahmad (CS24MTECH02002)
