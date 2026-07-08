# ACIDS Spectral MLP (1-Channel, Int8)

Self-contained C inference package for the W8-quantized **spectral MLP** (ACIDS vehicle classification, **ch0 continuous path**, **no mel**).

This package runs end-to-end in C: **1600-sample audio → spectral features → StandardScaler → W8 MLP → class logits**.

## Requirements

- `gcc`, `make`, standard C library (`-lm`)
- Python is **not** required for inference

## Build And Run

```bash
cd ARL_deploy/ACIDS_spectral_mlp_int8_ch0
make
./spectral_infer
```

Running `./spectral_infer` with no arguments runs all 15 packaged samples in `samples/`.

To run one explicit sample:

```bash
./spectral_infer samples/Gv3c1090_96.bin
```

Convert a human-readable `.txt` sample to `.bin` (no Python needed):

```bash
./spectral_txt_to_bin samples_txt/Gv1c2020_66.txt /tmp/Gv1c2020_66.bin
./spectral_infer /tmp/Gv1c2020_66.bin
```

## End-To-End Summary

| Stage | Function | Input shape | Output shape | Notes |
|-------|----------|-------------|--------------|-------|
| 1. Load sample | `main.c` | — | `(1600,)` float32 | Raw ch0 stream @ 1600 Hz |
| 2. Spectral frontend | `acids_spectral_preprocess()` | `(1600,)` | `(83,)` float32 | RFFT on first 160 samples; no mel |
| 3. StandardScaler | `spectral_apply_standard_scaler()` | `(83,)` raw | `(83,)` scaled | Stats in `spectral_scaler.h`; not folded into weights |
| 4. W8 MLP | `model_forward()` | `(83,)` scaled | `(10,)` logits | Linear/ReLU; int8 weight-only linear layers |
| 5. Decision | argmax in `main.c` | `(10,)` logits | class id | Maps to name via `acids_class_names.h` |

## Input Format

Each sample is raw float32 binary: a **flat 1600-sample ch0 audio stream** @ 1600 Hz (continuous truncate path).

| Field | Value |
|-------|-------|
| Dtype | `float32` |
| Layout | 1-D stream |
| Total values | 1600 |
| File size | 6400 bytes (1600 × 4) |

Your upstream pipeline must deliver **audio channel 0** already decimated to 1600 Hz as a contiguous 1-second clip. The C code does **not** perform downsampling or multi-mic selection.

## Clip Duration

This model sees a **1-second ACIDS continuous window** (1600 samples @ 1600 Hz).

| Stage | Shape | Sample rate | Duration |
|-------|-------|-------------|----------|
| Raw input | (1600,) | 1600 Hz | 1.0 s |
| RFFT window (used) | (160,) | 1600 Hz | 0.1 s |
| Spectral features (raw) | (83,) | — | centroid + mean energy + PSD |
| Scaled features | (83,) | — | after StandardScaler |
| Model output | (10,) | — | class logits |

**Important:** Only `input[0..159]` is used for the RFFT. Samples 160–1599 are present for format compatibility but ignored (matches Python training in `train_ch0_classifiers.py`).

### Human-readable `.txt` format

The same data is available in `samples_txt/` (one float per line, with a short header):

| Line | Content | Example |
|------|---------|---------|
| 1 | Sample name | `Gv1c2020_66` |
| 2 | Label id (use `-1` if unknown) | `0` |
| 3 | Layout tag (fixed) | `RAW_AUDIO_STREAM` |
| 4 | Count (fixed for this model) | `1600` |
| 5–1604 | Stream values | one float per line |

Convert `.txt` → `.bin` with `./spectral_txt_to_bin` (built by `make`).

## Data Flow

```text
sample .bin  [1600] @ 1600 Hz
  -> main.c
  -> acids_spectral_preprocess()
       take input[0..159]
       RFFT (n_fft=160) -> power spectrum (81 bins)
       centroid + mean energy + normalized PSD -> 83 raw features
  -> spectral_apply_standard_scaler()
  -> model_forward()   (W8 MLP: 83 -> 128 -> 128 -> 10)
  -> argmax logits
```

## Main Files

| File | Role |
|------|------|
| `main.c` | Inference runner; loads 1600-float `.bin`, runs full pipeline, prints prediction |
| `spectral_txt_to_bin.c` | Converts human-readable `.txt` samples to `.bin` |
| `acids_spectral_preprocess.c/.h` | Spectral frontend (RFFT + centroid / mean energy / PSD) |
| `acids_spectral_rfft_tables.h` | Precomputed RFFT basis (`torch.fft.rfft`, n_fft=160) and frequency bins |
| `spectral_scaler.c/.h` | StandardScaler: `(x - mean) / scale` on 83-dim features before the MLP |
| `model.c/.h`, `weights.h` | Generated W8 MLP graph and quantized weights |
| `nn_ops_float.h`, `nn_ops_int8.h` | Runtime operators used by the generated model |
| `acids_samples.h` | Packaged sample manifest (names, paths, expected labels) |
| `acids_class_names.h` | Maps output indices to class names |

`main.c`:

- Opens one or more raw sample binaries (1600 float32 each).
- If no input path is provided, loops through packaged samples from `acids_samples.h`.
- Calls `acids_spectral_preprocess()` → `spectral_apply_standard_scaler()` → `model_forward()`.
- Prints expected class, predicted class, logits, and match status.

## Preprocessing Details

### Spectral frontend

Entry point:

```c
int acids_spectral_preprocess(const float *input_1600, float *out_features_83);
```

Input:

```text
input_1600: [1600] float32, ch0 continuous stream @ 1600 Hz
```

Output:

```text
out_features_83: [83] float32, unscaled features:
  [0]     spectral centroid (Hz)
  [1]     mean spectral energy
  [2:82]  normalized PSD (81 bins)
```

Spectral parameters (matches training):

```text
n_fft: 160
sample_rate: 1600 Hz
use_hann: false
epsilon: 1e-12
feature layout: [centroid, mean_energy, psd[0..80]]
```

Pipeline:

```text
1600-sample stream @ 1600 Hz
  -> use first 160 samples only
  -> RFFT, 81 bins
  -> power = |fft|^2
  -> centroid = sum(power * freq) / sum(power)
  -> mean_energy = mean(power)
  -> psd = power / sum(power)
  -> concat -> 83 raw features
```

Lookup tables:

- `acids_spectral_rfft_tables.h`: RFFT basis matching `torch.fft.rfft` with n_fft=160.
- `ACIDS_SPECTRAL_FREQS[81]`: frequency axis for centroid (k * 10 Hz).

### StandardScaler

Entry point:

```c
void spectral_apply_standard_scaler(const float *raw, float *scaled, int dim);
```

Applied **in C before** `model_forward()`. Mean and scale arrays are in `spectral_scaler.h` (from training checkpoint). They are **not** folded into the first Linear layer weights.

## Model Input And Output

Model input (after spectral frontend + scaler):

```text
[83] float32
```

Model output:

```text
logits[10]
```

The runner uses argmax over these 10 logits.

Class mapping:

```text
0 -> background
1 -> 1
2 -> 2
3 -> 3
4 -> 4
5 -> 5
6 -> 6
7 -> 7
8 -> 8
9 -> 9
```

## Model Source

| Field | Value |
|-------|-------|
| Training script | `src2/data_analysis/train_ch0_classifiers.py` |
| Architecture | MLP(83 → 128 → 128 → 10), ReLU, Dropout (eval identity) |
| Features | spectral centroid + mean energy + PSD (`n_fft=160`, ch0, no mel) |
| Classes | 10 (background, 1–9) |
| Float val accuracy | 83.2% (spectral MLP, ch0 continuous path) |
| Parameters | 28,554 |
| Quantization | Linear weight-only int8 per-channel |
| Scaler | StandardScaler in C (explicit, not folded) |

Training path: ch0 audio → continuous truncate to 1600 samples → spectral features → StandardScaler → MLP.

## Training Methodology

Training is performed offline in Python (`src2/data_analysis/train_ch0_classifiers.py`). The C deploy package contains only the exported weights and preprocessing constants from that run.

### Dataset and task

| Field | Value |
|-------|-------|
| Dataset | ACIDS vehicle classification |
| Task | 10-class vehicle type (`vehicle_type` label subkey) |
| Classes | background, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Train / val split | ACIDS official index files in `src2/data/ACIDS.yaml` |
| Dataloader | `acids_vehicle` (single_label_only) |

### Audio preprocessing (before feature extraction)

```text
ACIDS .pt sample (3 ch × 7 segments × 256 samples @ 16 kHz)
  -> select audio channel 0
  -> AcidsContinuousTruncator: flatten segments, truncate to 1600 samples @ 1600 Hz
  -> 1600-sample 1-D stream per example
```

| Parameter | Value |
|-----------|-------|
| Channel | 0 (mic ch0 only) |
| Sample rate | 1600 Hz (post-decimation) |
| Stream length | 1600 samples (1.0 s) |
| Segments | 7 × 256 padded, truncated to 1600 |

### Feature extraction (no mel)

From each 1600 stream, spectral features are computed in Python (`analyze_mel_spectral_features.py`):

```text
stream[0:160]  (first 160 samples only)
  -> np.fft.rfft(n=160), no Hann window
  -> power spectrum (81 bins)
  -> centroid + mean energy + normalized PSD
  -> 83-dim feature vector
```

| Parameter | Value |
|-----------|-------|
| n_fft | 160 |
| Hann window | false (default) |
| Feature dim | 83 (2 stats + 81 PSD bins) |

### MLP training

| Field | Value |
|-------|-------|
| Model | `SpectralMLP`: Linear(83→128) → ReLU → Dropout(0.3) → Linear(128→128) → ReLU → Dropout(0.3) → Linear(128→10) |
| Input normalization | `StandardScaler` fit on train features only |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Epochs | 50 |
| Batch size | 256 |
| Checkpoint selection | Best validation accuracy across epochs |
| Train command | `python train_ch0_classifiers.py --yaml_path ../data/ACIDS.yaml` |

Training applies StandardScaler **before** the MLP in Python. The same scaler mean/scale are exported to `spectral_scaler.h` and applied in C at inference (not folded into layer weights).

### Training results (float32 MLP)

| Split | Accuracy | Macro F1 |
|-------|----------|----------|
| Train | 95.8% | 0.958 |
| Val | 83.2% | 0.704 |

Checkpoint used for C export: `src2/tmp/ch0_classifiers/models/spectral_mlp.pth` (includes `model_state_dict`, `scaler_mean`, `scaler_scale`).

### Export to C (offline, not in this deploy folder)

The float MLP was compiled and quantized to int8 weight-only Linear layers via Tiny-NN-in-C. Conv calibration is not used (MLP only). Packaged samples were exported from ACIDS test `.pt` files as 1600-sample ch0 streams.

## Quantization

This package uses:

```text
Linear layers: weight-only int8 per-channel (per-output-column scales in weights.h)
Activations: float32 throughout (including MLP inputs after scaler)
No conv layers
No activation calibration required (weight scales computed from float weights)
```

Expected W8 vs float32 argmax agreement on full val set: ~99.7% (7 borderline flips on close logits).

## Expected Output

For each packaged sample, the runner prints:

```text
sample=<sample name>
input=<sample path>
expected_id=<label id>
expected_name=<label name>
pred_id=<predicted id>
pred_name=<predicted name>
logits=<comma-separated 10 logits>
match=yes/no
```

Example:

```text
sample=Gv6d1090_125
input=samples/Gv6d1090_125.bin
expected_id=6
expected_name=6
pred_id=6
pred_name=6
logits=-4.88128,-5.05032,-6.89247,-15.0356,-1.22536,-1.64581,12.1884,-25.8881,1.10446,0.470395
match=yes
```

When running a single file without expected labels, `expected_*` and `match` lines are omitted.

## Package Layout

```text
ACIDS_spectral_mlp_int8_ch0/
  README.md
  Makefile
  main.c                      # inference runner
  spectral_txt_to_bin.c       # .txt -> .bin converter
  acids_spectral_preprocess.c/.h
  acids_spectral_rfft_tables.h
  spectral_scaler.c/.h
  acids_class_names.h
  acids_samples.h
  model.c / model.h
  weights.h
  nn_ops_float.h / nn_ops_int8.h
  samples/*.bin               # 15 binary inputs (6400 bytes each)
  samples_txt/*.txt           # same samples, human-readable
```
