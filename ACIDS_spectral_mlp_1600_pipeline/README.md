# Spectral MLP 1600 Pipeline (Welch 83-dim)

Self-contained train → W8 quantize → **C inference** package for ACIDS vehicle classification.

- **Input:** 1 s audio as `float32[1600]` @ 1600 Hz (ch0 continuous truncate from ACIDS `.pt`)
- **Frontend:** Welch mean periodogram (`n_fft=160`, `hop=80`) → **83** features
- **Model:** MLP `83 → 128 → 128 → 10`, StandardScaler, weight-only **int8**
- **Inference:** entirely in C (`deploy/spectral_infer`)

## Latest full-ACIDS results

| Split | N | Accuracy | Macro F1 |
|-------|---|----------|----------|
| Train | 21631 | 98.1% | 0.963 |
| Val | 2190 | 90.7% | 0.778 |
| **Test (C infer)** | **2190** | **91.0%** | **0.780** |

Test scoring calls `./spectral_infer` on every test `.bin` (Welch + scaler + W8 MLP in C).

## Setup (conda env for this folder only)

```bash
cd ARL_deploy/ACIDS_spectral_mlp_1600_pipeline
conda env create -f environment.yml
conda activate spectral_mlp_1600
```

Needs `gcc` / `make`. **Tiny-NN-in-C is vendored** in `Tiny-NN-in-C/` (used only for W8 export).

## Point at ACIDS train / val / test

Edit [`paths.yaml`](paths.yaml) (defaults match `src2/data/ACIDS.yaml` `vehicle_classification`):

```yaml
train_index_file: /data/misra8/ACIDS/random_partition_index_vehicle_classification/train_index.txt
val_index_file:   /data/misra8/ACIDS/random_partition_index_vehicle_classification/val_index.txt
test_index_file:  /data/misra8/ACIDS/random_partition_index_vehicle_classification/test_index.txt
```

Or pass `--train_index` / `--val_index` / `--test_index` on the CLI.

## One-shot pipeline

```bash
python3 run_acids_pipeline.py --paths_yaml paths.yaml --epochs 50 --gpu 0
```

Quick smoke (cap each split):

```bash
python3 run_acids_pipeline.py --max_export_samples 200 --epochs 10 --gpu -1
```

### Steps

1. **`export_acids_indices.py`** — ACIDS `.pt` → ch0 `[:1600]` → `exported_acids/{train,val,test}_X.npy` + `test_bin/*.bin`
2. **`train_from_1600.py`** — Welch 83-dim → StandardScaler → MLP → `artifacts/spectral_mlp.pth`
3. **`export_w8_spectral_mlp.py`** — Tiny-NN W8 → `deploy/model.*`, `weights.h`, `spectral_scaler.h`
4. **`make -C deploy`** — builds `spectral_infer` (Welch C + kissfft + scaler + W8)
5. **`run_c_test_infer.py`** — C inference on all test bins → `artifacts/c_test_report.json`

## Run C inference on samples

```bash
cd deploy
make

# packaged smoke samples
./spectral_infer

# any raw 1600-float32 .bin (6400 bytes, no header)
./spectral_infer /path/to/clip.bin

# from samples_txt-style .txt
./spectral_txt_to_bin clip.txt /tmp/clip.bin
./spectral_infer /tmp/clip.bin
```

`.txt` layout:

```text
sample_name
<label_id 0-9>       # vehicle_type: 0=background, 1..9
RAW_AUDIO_STREAM
1600
<1600 floats>
```

## Layout

```text
environment.yml           # conda env: spectral_mlp_1600
paths.yaml                # ACIDS index paths (edit these)
export_acids_indices.py
train_from_1600.py
export_w8_spectral_mlp.py
package_deploy_samples.py
run_c_test_infer.py
run_acids_pipeline.py     # full ACIDS flow
run_pipeline.py           # optional: already-flat txt train/val only
welch_features.py
load_1600_txt.py
Tiny-NN-in-C/             # vendored W8 compiler (export only)
exported_acids/           # after export (gitignored)
artifacts/                # checkpoint + reports (*.pth gitignored)
deploy/                   # COMPLETE C inference packet (handoff)
```

## Handoff

Zip/copy **`deploy/`** alone. Recipients need only `gcc`/`make` — no Python, no Tiny-NN, no ACIDS `.pt` data.

## Notes

- Welch uses the **full** 1600 samples (not first-160 snapshot). Old snapshot checkpoints are incompatible — retrain + re-export.
- Scaler stays in C (`spectral_scaler.h`), not folded into layer weights.
- Feature dim stays **83**; MLP flash footprint matches the snapshot architecture.
