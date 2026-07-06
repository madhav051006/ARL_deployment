#!/usr/bin/env python3
"""Compare float32 PyTorch vs C W8 inference on all packaged ACIDS samples."""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parent.parent if (PKG.parent.parent / "src2").is_dir() else PKG.parent
SRC2 = REPO / "src2"
sys.path.insert(0, str(SRC2))

from data_augmenter.mel_preprocess import MelPreprocessor  # noqa: E402
from models.DeepSenseDepthwise import DeepSenseDepthwiseBackbone  # noqa: E402

CKPT = SRC2 / (
    "experiments/20260706_183051_finetune_finetune_audio_deepsense_dw_large_mel_5class_nfft25/"
    "models/best_model.pth"
)

NUM_SEGMENTS = 7
SAMPLES_PER_SEGMENT = 25
MEL_BINS = 80
RAW_SHAPE = (3, NUM_SEGMENTS, SAMPLES_PER_SEGMENT)
MEL_SIZE = NUM_SEGMENTS * MEL_BINS

SAMPLE_NAMES = [
    "Gv3c1090_96",
    "Gv6d1090_125",
    "Gv3c1090_231",
    "Gv1c2020_66",
    "Gv8c1220_99",
    "Gv1c2020_70",
    "Gv7c1056_4",
    "Gv3c1090_114",
    "Gv7c1056_178",
    "Gv9d1126_58",
    "Gv8c1220_180",
    "Gv9d1126_126",
    "Gv6d1090_6",
    "Gv3c1090_362",
    "Gv3c1090_215",
]

GT_LABEL_IDS = [0, 2, 0, 0, 3, 0, 0, 0, 0, 4, 3, 4, 0, 0, 0]


def compile_mel_so() -> Path:
    lib_path = PKG / "libacids_mel_preprocess.so"
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c99",
        str(PKG / "acids_mel_preprocess.c"),
        "-o",
        str(lib_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return lib_path


def compile_model_so() -> Path:
    lib_path = PKG / "libmodel_w8.so"
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c99",
        str(PKG / "model.c"),
        "-o",
        str(lib_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return lib_path


def c_mel_from_raw(raw_chw: np.ndarray, lib_path: Path) -> np.ndarray:
    lib = ctypes.CDLL(str(lib_path))
    lib.acids_audio_preprocess_ch0_segments_chw.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.acids_audio_preprocess_ch0_segments_chw.restype = ctypes.c_int
    flat_in = np.ascontiguousarray(raw_chw.astype(np.float32).reshape(-1))
    out_arr = np.zeros(MEL_SIZE, dtype=np.float32)
    rc = lib.acids_audio_preprocess_ch0_segments_chw(
        flat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"acids_audio_preprocess_ch0_segments_chw returned {rc}")
    return out_arr.reshape(1, NUM_SEGMENTS, MEL_BINS)


def w8_logits(mel_1ch: np.ndarray, lib_path: Path, num_classes: int = 5) -> np.ndarray:
    lib = ctypes.CDLL(str(lib_path))
    lib.model_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.model_forward.restype = None
    flat_in = np.ascontiguousarray(mel_1ch.astype(np.float32).reshape(-1))
    out = np.zeros(num_classes, dtype=np.float32)
    lib.model_forward(
        flat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def make_mel_preprocessor() -> MelPreprocessor:
    return MelPreprocessor(
        n_fft=25, n_mel=80, fmin=20.0, fmax=800.0, sample_rate=1600, device="cpu"
    )


def pytorch_mel_1ch(raw_chw: np.ndarray, mel_pp: MelPreprocessor) -> np.ndarray:
    x = torch.from_numpy(raw_chw[0:1].astype(np.float32)).unsqueeze(0)
    out = mel_pp.preprocess({"shake": {"audio": x}})
    return out["shake"]["audio"].squeeze(0).numpy()


def pytorch_mel_3ch(raw_chw: np.ndarray, mel_pp: MelPreprocessor) -> np.ndarray:
    parts = []
    for c in range(3):
        x = torch.from_numpy(raw_chw[c : c + 1].astype(np.float32)).unsqueeze(0)
        out = mel_pp.preprocess({"shake": {"audio": x}})
        parts.append(out["shake"]["audio"])
    mel = torch.cat(parts, dim=1).squeeze(0)
    return mel.numpy()


def build_backbone(model_cfg: dict, num_classes: int, in_channels: int) -> DeepSenseDepthwiseBackbone:
    return DeepSenseDepthwiseBackbone(
        in_channels=in_channels,
        in_spectrum_len=model_cfg["in_spectrum_len"],
        num_classes=num_classes,
        channels_freq=model_cfg["channels_freq"],
        kernel_sizes_freq=model_cfg["kernel_sizes_freq"],
        strides_freq=model_cfg["strides_freq"],
        temporal_channels=model_cfg["temporal_channels"],
        num_temporal_layers=model_cfg["num_temporal_layers"],
        temporal_kernel=model_cfg["temporal_kernel"],
        fc_dim=model_cfg["fc_dim"],
        dropout_ratio=0.0,
        pretrain_mode=False,
        proj_hidden_dim=model_cfg["proj_hidden_dim"],
        proj_out_dim=model_cfg["proj_out_dim"],
    )


def load_backbone_1ch(backbone: DeepSenseDepthwiseBackbone, ckpt_path: Path, channel_index: int = 0) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    mapped = {}
    for key, value in sd.items():
        if not key.startswith("backbone."):
            continue
        local_key = key[len("backbone.") :]
        if local_key == "freq_stack.0.depthwise.weight" and value.shape[0] > 1:
            mapped[local_key] = value[channel_index : channel_index + 1].clone()
        elif (
            local_key == "freq_stack.0.pointwise.weight"
            and value.ndim == 4
            and value.shape[1] > 1
        ):
            mapped[local_key] = value[:, channel_index : channel_index + 1, :, :].clone()
        else:
            mapped[local_key] = value
    missing, unexpected = backbone.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading 1ch backbone: {unexpected}")
    allowed_missing = {
        "projection_head.0.weight",
        "projection_head.0.bias",
        "projection_head.2.weight",
        "projection_head.2.bias",
    }
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Missing 1ch backbone keys: {bad_missing}")


def load_backbone_3ch(backbone: DeepSenseDepthwiseBackbone, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    mapped = {key[len("backbone.") :]: value for key, value in sd.items() if key.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading 3ch backbone: {unexpected}")
    allowed_missing = {
        "projection_head.0.weight",
        "projection_head.0.bias",
        "projection_head.2.weight",
        "projection_head.2.bias",
    }
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Missing 3ch backbone keys: {bad_missing}")


@torch.no_grad()
def float_logits(backbone: DeepSenseDepthwiseBackbone, mel: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)
    out = backbone(x)
    return out["logits"].squeeze(0).numpy()


def logit_metrics(ref: np.ndarray, other: np.ndarray) -> tuple[float, float, float]:
    diff = ref - other
    max_abs = float(np.max(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))
    denom = float(np.linalg.norm(ref) * np.linalg.norm(other))
    cos = float(np.dot(ref, other) / denom) if denom > 0 else 1.0
    return max_abs, l2, cos


def argmax_logits(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def parse_acids_infer_block(text: str, sample_name: str) -> np.ndarray | None:
    pattern = rf"sample={re.escape(sample_name)}\n.*?logits=([0-9eE+\-\.,]+)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    return np.array([float(x) for x in m.group(1).split(",")], dtype=np.float32)


def main() -> None:
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    config_path = CKPT.parent.parent / "config.yaml"
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment_name = config["experiment_name"]
    experiment_config = config["experiments"][experiment_name]
    model_cfg = config["models"][experiment_config["model"]]
    num_classes = config[config["task_name"]]["num_classes"]

    mel_lib = compile_mel_so()
    model_lib = compile_model_so()

    mel_pp = make_mel_preprocessor()

    bb1 = build_backbone(model_cfg, num_classes, in_channels=1)
    load_backbone_1ch(bb1, CKPT, channel_index=0)
    bb1.eval()

    bb3 = build_backbone(model_cfg, num_classes, in_channels=3)
    load_backbone_3ch(bb3, CKPT)
    bb3.eval()

    infer_exe = PKG / "acids_infer"
    infer_out = subprocess.run(
        [str(infer_exe)],
        cwd=str(PKG),
        capture_output=True,
        text=True,
        check=False,
    )
    infer_text = infer_out.stdout

    rows = []
    max_abs_1ch_vs_w8 = []
    max_abs_3ch_vs_w8 = []
    max_abs_1ch_cmel_vs_w8 = []

    for i, name in enumerate(SAMPLE_NAMES):
        bin_path = PKG / "samples" / f"{name}.bin"
        raw = np.fromfile(bin_path, dtype=np.float32).reshape(RAW_SHAPE)

        mel_c = c_mel_from_raw(raw, mel_lib)
        mel_pt_1 = pytorch_mel_1ch(raw, mel_pp)
        mel_pt_3 = pytorch_mel_3ch(raw, mel_pp)

        mel_c_vs_pt = float(np.max(np.abs(mel_c - mel_pt_1)))

        w8 = w8_logits(mel_c, model_lib, num_classes)
        infer_w8 = parse_acids_infer_block(infer_text, name)
        if infer_w8 is not None:
            infer_diff = float(np.max(np.abs(w8 - infer_w8)))
            if infer_diff > 1e-5:
                print(f"WARN {name}: ctypes W8 vs acids_infer max diff {infer_diff:.3e}")

        fl1 = float_logits(bb1, mel_pt_1)
        fl3 = float_logits(bb3, mel_pt_3)
        fl1_cmel = float_logits(bb1, mel_c)

        m1 = logit_metrics(fl1, w8)
        m3 = logit_metrics(fl3, w8)
        m1c = logit_metrics(fl1_cmel, w8)
        max_abs_1ch_vs_w8.append(m1[0])
        max_abs_3ch_vs_w8.append(m3[0])
        max_abs_1ch_cmel_vs_w8.append(m1c[0])

        gt = GT_LABEL_IDS[i]
        rows.append(
            {
                "sample": name,
                "mel_c_vs_pt": mel_c_vs_pt,
                "max_abs_1ch": m1[0],
                "l2_1ch": m1[1],
                "cos_1ch": m1[2],
                "max_abs_3ch": m3[0],
                "l2_3ch": m3[1],
                "cos_3ch": m3[2],
                "max_abs_1ch_cmel": m1c[0],
                "pred_f1": argmax_logits(fl1),
                "pred_f3": argmax_logits(fl3),
                "pred_f1_cmel": argmax_logits(fl1_cmel),
                "pred_w8": argmax_logits(w8),
                "gt": gt,
                "match_1ch_w8": argmax_logits(fl1) == argmax_logits(w8),
                "match_3ch_w8": argmax_logits(fl3) == argmax_logits(w8),
                "match_1ch_cmel_w8": argmax_logits(fl1_cmel) == argmax_logits(w8),
            }
        )

    arr_1 = np.array(max_abs_1ch_vs_w8)
    arr_3 = np.array(max_abs_3ch_vs_w8)
    arr_1c = np.array(max_abs_1ch_cmel_vs_w8)
    export_match = "1ch (matches C export)" if arr_1.mean() <= arr_3.mean() else "3ch"
    if arr_1.mean() < arr_3.mean():
        export_match = "1ch (matches C W8 export; channel-0 slice)"
    elif arr_3.mean() < arr_1.mean():
        export_match = "3ch full checkpoint"
    else:
        export_match = "tie (1ch export path; both similar)"

    agree_f1_w8 = sum(r["match_1ch_w8"] for r in rows) / len(rows)
    agree_f3_w8 = sum(r["match_3ch_w8"] for r in rows) / len(rows)
    agree_f1c_w8 = sum(r["match_1ch_cmel_w8"] for r in rows) / len(rows)
    agree_f1_gt = sum(r["pred_f1"] == r["gt"] for r in rows) / len(rows)
    agree_f3_gt = sum(r["pred_f3"] == r["gt"] for r in rows) / len(rows)
    agree_w8_gt = sum(r["pred_w8"] == r["gt"] for r in rows) / len(rows)

    print("Per-sample float32 vs W8 (using PyTorch mel ch0; W8 on C mel)")
    print(
        f"{'sample':<16} {'mel|C-PT|':>10} {'max|F1-W8|':>11} {'L2 F1-W8':>10} "
        f"{'cos':>7} {'predF1':>6} {'predW8':>6} {'gt':>3} {'F1=W8':>6}"
    )
    for r in rows:
        print(
            f"{r['sample']:<16} {r['mel_c_vs_pt']:10.3e} {r['max_abs_1ch']:11.3e} "
            f"{r['l2_1ch']:10.3e} {r['cos_1ch']:7.5f} {r['pred_f1']:6d} {r['pred_w8']:6d} "
            f"{r['gt']:3d} {'yes' if r['match_1ch_w8'] else 'no':>6}"
        )

    print()
    print("Aggregate max |logit diff| (float vs W8), per sample")
    print(f"{'variant':<22} {'mean':>12} {'median':>12} {'max':>12} {'pred agree w/ W8':>18}")
    print(
        f"{'1ch PT mel':<22} {arr_1.mean():12.3e} {np.median(arr_1):12.3e} {arr_1.max():12.3e} "
        f"{agree_f1_w8:17.1%}"
    )
    print(
        f"{'1ch C mel (export)':<22} {arr_1c.mean():12.3e} {np.median(arr_1c):12.3e} {arr_1c.max():12.3e} "
        f"{agree_f1c_w8:17.1%}"
    )
    print(
        f"{'3ch PT mel':<22} {arr_3.mean():12.3e} {np.median(arr_3):12.3e} {arr_3.max():12.3e} "
        f"{agree_f3_w8:17.1%}"
    )

    print()
    print("Prediction agreement with ground truth (15 samples)")
    print(f"  float 1ch vs GT: {agree_f1_gt:.1%} ({sum(r['pred_f1']==r['gt'] for r in rows)}/15)")
    print(f"  float 3ch vs GT: {agree_f3_gt:.1%} ({sum(r['pred_f3']==r['gt'] for r in rows)}/15)")
    print(f"  W8 vs GT:        {agree_w8_gt:.1%} ({sum(r['pred_w8']==r['gt'] for r in rows)}/15)")

    print()
    print(f"Export alignment: {export_match}")
    print(f"  (1ch mean max logit diff vs W8: {arr_1.mean():.3e}; 3ch: {arr_3.mean():.3e})")


if __name__ == "__main__":
    main()
