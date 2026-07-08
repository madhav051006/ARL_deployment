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

When running all packaged samples via `./spectral_infer` (no arguments), each line also includes `sample=`, `expected_id=`, `expected_name=`, and `match=yes/no`.

## Txt Sample Format

Human-readable samples live in `samples_txt/`. Each file starts with a short header, then one float per line:

```text
Gv1c2020_66          <-- sample name
0                    <-- label id (use -1 if unknown)
RAW_AUDIO_STREAM     <-- layout tag (fixed)
1600                 <-- stream length (fixed)
186                  <-- stream[0]
118                  <-- stream[1]
76                   <-- stream[2]
60                   <-- stream[3]
...
```

Lines 5–1604 are the 1600 audio samples (`stream[0]` … `stream[1599]`).

Convert a `.txt` sample to `.bin` (no Python needed):

```bash
./spectral_txt_to_bin samples_txt/Gv1c2020_66.txt /tmp/Gv1c2020_66.bin
./spectral_infer /tmp/Gv1c2020_66.bin
```

## From ACIDS `(3, 7, 256)` to `(1600)`

1. ACIDS stores each clip’s mic audio as `(3, 7, 256)` — 3 channels, 7 time segments, 256 samples per segment — captured at **16 kHz** native rate.
2. Looking at a singular channel this takes the shape `(1, 7, 256)` with 1792 total samples out of which we keep 1600 samples from a 1 second event effectively downsampling to **1600 Hz**.

```python
# ACIDS .pt mic tensor (16 kHz native capture)
audio = sample["data"]["shake"]["audio"]          # (3, 7, 256)

# ch0 only -> flatten 7 segments -> keep first 1600 samples
ch0 = audio[0]                                    # (7, 256)
flat = ch0.reshape(7 * 256)                       # (1792,)  = 7 * 256
stream_1600 = flat[:1600]                         # (1600,)  1.0 s @ 1600 Hz
```

## Preprocessing: `.txt` → Model Input

Once you have a `1600`-line stream in a `.txt` file (or an equivalent `.bin`), inference runs through four stages.

```text
samples_txt/*.txt
  -> spectral_txt_to_bin          
  -> samples/*.bin  [1600] float32
  -> main.c                       (load)
  -> acids_spectral_preprocess()  (1600 -> 83 raw spectral features)
  -> spectral_apply_standard_scaler()  (83 raw -> 83 scaled)
  -> model_forward()              (83 scaled -> 10 logits)
```

### Step 0 — `.txt` → `.bin` (`spectral_txt_to_bin`)

Skips the 4-line header (name, label, `RAW_AUDIO_STREAM`, count). Parses the next **1600** lines as `float32` values and writes them verbatim to a raw binary file (6400 bytes). 

```text
input:  header + 1600 text floats
output: float32[1600] raw audio stream
```

### Step 1 — Load audio (`main.c`)

Reads exactly **1600 × 4 = 6400 bytes** from the `.bin` file into `audio[1600]`. 

```text
input:  float32[1600]  (ch0 continuous stream @ 1600 Hz)
output: float32[1600]  (same buffer, in memory)
```

### Step 2 — Spectral frontend (`acids_spectral_preprocess`)

Converts the time-domain stream into **83 unscaled spectral features**. Only the **first 160 samples** (`audio[0..159]`) are used; `audio[160..1599]` is ignored (matches training).

Sub-steps:

1. **Window** — copy `audio[0..159]` into a 160-sample buffer.
2. **RFFT** — real FFT with `n_fft=160` via **kissfft** at runtime (matches `np.fft.rfft`). Produces **81** frequency bins.
3. **Power spectrum** — `power[k] = real(k)² + imag(k)²` for `k = 0..80`.
4. **Spectral centroid** — `centroid = Σ(power[k] × freq[k]) / (Σ power[k] + ε)` where `freq[k] = k × 10 Hz` at 1600 Hz sample rate.
5. **Mean spectral energy** — `mean_energy = Σ power[k] / 81`.
6. **Normalized PSD** — `psd[k] = power[k] / (Σ power[k] + ε)` (sums to ~1.0 across bins).

Feature vector layout:

```text
raw[0]       spectral centroid (Hz)
raw[1]       mean spectral energy
raw[2:82]    normalized PSD, 81 bins (0 Hz .. 800 Hz)
```

```text
input:  float32[1600]
output: float32[83]   (unscaled)
```

### Step 3 — StandardScaler (`spectral_apply_standard_scaler`)

Applies the training-time scaler **in C before the MLP** (not folded into layer weights). Constants are in `spectral_scaler.h`, exported from the training checkpoint.

```c
scaled[i] = (raw[i] - SPECTRAL_SCALER_MEAN[i]) / SPECTRAL_SCALER_SCALE[i];
```

```text
input:  float32[83]   (raw spectral features)
output: float32[83]   (scaled — this is the MLP input)
```

### Step 4 — Model input

`model_forward(scaled, logits)` expects exactly this **83-dimensional scaled vector**. The W8 MLP runs `Linear(83→128) → ReLU → Linear(128→128) → ReLU → Linear(128→10)` and writes 10 class logits. `main.c` takes the argmax to produce the predicted class.

```text
model input:  float32[83]   (after StandardScaler)
model output: float32[10]   (logits; argmax -> class id)
```

### Shape summary

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 0 (optional) | `spectral_txt_to_bin` | `.txt` header + 1600 lines | `float32[1600]` `.bin` |
| 1 | `main.c` fread | `float32[1600]` file | `float32[1600]` buffer |
| 2 | `acids_spectral_preprocess` | `float32[1600]` | `float32[83]` raw |
| 3 | `spectral_apply_standard_scaler` | `float32[83]` raw | `float32[83]` scaled |
| 4 | `model_forward` | `float32[83]` scaled | `float32[10]` logits |

## End-To-End Summary

| Stage | Function | Input shape | Output shape | Notes |
|-------|----------|-------------|--------------|-------|
| 1. Load sample | `main.c` | — | `(1600,)` float32 | Raw ch0 stream @ 1600 Hz |
| 2. Spectral frontend | `acids_spectral_preprocess()` | `(1600,)` | `(83,)` float32 | RFFT on first 160 samples; no mel |
| 3. StandardScaler | `spectral_apply_standard_scaler()` | `(83,)` raw | `(83,)` scaled | Stats in `spectral_scaler.h`; not folded into weights |
| 4. W8 MLP | `model_forward()` | `(83,)` scaled | `(10,)` logits | Linear/ReLU; int8 weight-only linear layers |
| 5. Decision | argmax in `main.c` | `(10,)` logits | class id | Maps to name via `acids_class_names.h` |


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
| `acids_spectral_preprocess.c/.h` | Spectral frontend (kissfft RFFT + centroid / mean energy / PSD) |
| `kissfft/` | Vendored kissfft real FFT (`kiss_fftr`, n_fft=160); ~2 KB static cfg, no 100 KB lookup tables |
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


### Dataset and task

| Field | Value |
|-------|-------|
| Dataset | ACIDS vehicle classification |
| Task | 10-class vehicle type (`vehicle_type` label subkey) |
| Classes | background, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Train / val split | ACIDS official index files in `src2/data/ACIDS.yaml` |
| Dataloader | `acids_vehicle` (single_label_only) |


### Training results (float32 MLP)

| Split | Accuracy | Macro F1 |
|-------|----------|----------|
| Train | 95.8% | 0.958 |
| Val | 83.2% | 0.704 |



## Quantization

This package uses:

```text
Linear layers: weight-only int8 per-channel (per-output-column scales in weights.h)
Activations: float32 throughout (including MLP inputs after scaler)
No conv layers
No activation calibration required (weight scales computed from float weights)
```

Expected W8 vs float32 argmax agreement on full val set: ~99.7% (7 borderline flips on close logits).



## Package Layout

```text
ACIDS_spectral_mlp_int8_ch0/
  README.md
  Makefile
  main.c                      # inference runner
  spectral_txt_to_bin.c       # .txt -> .bin converter
  acids_spectral_preprocess.c/.h
  kissfft/                    # vendored kissfft (BSD-3-Clause)
  spectral_scaler.c/.h
  acids_class_names.h
  acids_samples.h
  model.c / model.h
  weights.h
  nn_ops_float.h / nn_ops_int8.h
  samples/*.bin               # 15 binary inputs (6400 bytes each)
  samples_txt/*.txt           # same samples, human-readable
```
