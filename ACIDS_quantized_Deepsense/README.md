# ACIDS Quantized DeepSense

Self-contained C inference package for the W8-quantized DeepSense audio model (ACIDS vehicle classification, **`n_fft=25`** mel).

Remote: `git@github.com:madhav051006/ARL_deployment.git`

## Repository Layout

```text
ARL_deployment/
└── ACIDS_quantized_Deepsense/
    ├── Makefile
    ├── main.c
    ├── samples/
    └── ...
```

## Requirements

- `gcc`, `make`, standard C library (`-lm`)
- Python is **not** required for inference

## Build And Run

```bash
git clone git@github.com:madhav051006/ARL_deployment.git
cd ARL_deployment/ACIDS_quantized_Deepsense
make
./acids_infer
```

Running `./acids_infer` with no arguments runs all 15 packaged samples in `samples/`.

To run one explicit sample:

```bash
./acids_infer samples/Gv3c1090_96.bin
```

Convert a human-readable `.txt` sample to `.bin` (no Python needed):

```bash
./acids_txt_to_bin samples_txt/Gv3c1090_96.txt /tmp/Gv3c1090_96.bin
./acids_infer /tmp/Gv3c1090_96.bin
```

## Input Format

Each sample is raw float32 binary with this flattened shape:

```text
[C=3][segments=7][T=25]
```

This is **decimated audio at 1600 Hz** (25 samples per segment):

```text
audio: (3, 7, 25) @ 1600 Hz per segment
```

| Field | Value |
|-------|-------|
| Dtype | `float32` |
| Layout | CHW when flattened |
| Total values | 525 |
| File size | 2,100 bytes (525 × 4) |

## Clip Duration

This model sees a **short ACIDS event window (~110 ms)**.

| Stage | Shape | Sample rate | Duration |
|-------|-------|-------------|----------|
| Raw input | (3, 7, 25) | 1600 Hz | ~109 ms |
| After mel | (1, 7, 80) | — | 7 time bins |

Flatten order (channel-major):

```text
index(c, seg, t) = c * (7 * 25) + seg * 25 + t
```

The C runner uses only **audio channel 0**. Channels 1 and 2 are ignored.

Your upstream pipeline must deliver audio already decimated to 1600 Hz with 25 samples per segment. The C code does **not** perform downsampling.

### Human-readable `.txt` format

The same data is available in `samples_txt/` (one float per line, with a short header):

| Line | Content | Example |
|------|---------|---------|
| 1 | Sample name | `Gv3c1090_96` |
| 2 | Label id (use `-1` if unknown) | `0` |
| 3 | Layout tag (fixed) | `RAW_AUDIO_CHW` |
| 4 | Shape (fixed for this model) | `3,7,25` |
| 5–529 | Flattened CHW values | one float per line |

Convert `.txt` → `.bin` with `./acids_txt_to_bin` (built by `make`).

### Example Python writer

```python
import numpy as np

NUM_CHANNELS = 3
NUM_SEGMENTS = 7
SAMPLES_PER_SEGMENT = 25


def write_sample_bin(path: str, audio_chw: np.ndarray) -> None:
    if audio_chw.shape != (NUM_CHANNELS, NUM_SEGMENTS, SAMPLES_PER_SEGMENT):
        raise ValueError(f"expected (3, 7, 25), got {audio_chw.shape}")
    audio_chw.astype(np.float32).tofile(path)


def write_sample_txt(path: str, sample_name: str, label_id: int, audio_chw: np.ndarray) -> None:
    flat = audio_chw.astype(np.float32).reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{sample_name}\n{int(label_id)}\nRAW_AUDIO_CHW\n3,7,25\n")
        for value in flat:
            f.write(f"{float(value):.9g}\n")
```

## Data Flow

```text
sample .bin  [3][7][25] @ 1600 Hz
  -> main.c
  -> acids_audio_preprocess_ch0_segments_chw()
       select channel 0
       log-mel per segment (n_fft=25)
  -> model_forward()
  -> argmax logits
```

## Main Files

| File | Role |
|------|------|
| `main.c` | Inference runner |
| `acids_txt_to_bin.c` | `.txt` → `.bin` converter |
| `acids_mel_preprocess.c/.h` | Log-mel preprocessing (`n_fft=25`) |
| `acids_mel_tables.h`, `acids_rfft_tables.h` | Mel filterbank and RFFT tables |
| `model.c/.h`, `weights.h` | Generated W8 model graph and weights |
| `nn_ops_*.h` | Runtime operators used by the generated model |
| `acids_samples.h` | Packaged sample manifest |
| `acids_class_names.h` | Output class names |

Class mapping:

```text
0 -> background
1 -> 3
2 -> 6
3 -> 8
4 -> 9
```

## Preprocessing Details

Entry point:

```c
int acids_audio_preprocess_ch0_segments_chw(const float *input_chw, float *out_chw);
```

Input:

```text
input_chw: [3][7][25] float32, flattened CHW @ 1600 Hz per segment
```

Output:

```text
out_chw: [1][7][80] float32, flattened CHW
```

Mel parameters:

```text
n_fft: 25
mel_bins: 80
fmin: 20.0 Hz
fmax: 800.0 Hz
sample_rate: 1600 Hz
log epsilon: 1e-6
```

Pipeline per segment:

```text
25 samples @ 1600 Hz
  -> RFFT, 13 bins
  -> power spectrum
  -> mel filterbank [80][13]
  -> log(mel + 1e-6)
```

## Model Input And Output

Model input after preprocessing:

```text
[1][7][80] float32
```

Model output:

```text
logits[5]
```

The runner uses argmax over these 5 logits.

## Model Source

| Field | Value |
|-------|-------|
| Experiment | `finetune_audio_deepsense_dw_large_mel_5class_nfft25` |
| Architecture | `DeepSenseDepthwiseBackbone` (GELU, 1-channel deploy) |
| Classes | 5 (background, 3, 6, 8, 9) |
| Val accuracy | 69.1% (float32, epoch 8) |
| Quantization | Conv static int8 per-channel; linear weight-only int8 per-channel |

Training used 3-channel mel; C deployment uses channel 0 only.

## Quantization

```text
Convolution layers: static int8 per-channel (weights + activation scales)
Linear/projection layers: weight-only int8 per-channel
Activations around Linear layers: float32
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
logits=<comma-separated logits>
match=yes/no
```

## Package Layout

```text
ACIDS_quantized_Deepsense/
  README.md
  Makefile
  main.c
  acids_txt_to_bin.c
  model.c / model.h
  weights.h
  nn_ops_*.h
  acids_mel_preprocess.c/.h
  acids_mel_tables.h
  acids_rfft_tables.h
  acids_class_names.h
  acids_samples.h
  samples/*.bin          # 15 binary inputs (2,100 bytes each)
  samples_txt/*.txt      # same samples, human-readable
```
