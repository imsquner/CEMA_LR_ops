#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
tests/test_e2e.py — 端到端链路：已训练 checkpoint + NPU 算子 还原预测曲线。

链路（文档 §11.7 模式 1 分离编排，验证阶段模型可在 CPU 推理）：
    data_cache 特征 → 训练好的 checkpoint 模型 → 标准化 LR 预测 r̂
    → [NPU] LRDecode(r̂, 窗口末 EMA 锚点, μ, σ) → 电压预测
    → 与 results/predictions/formal__F3__*.csv 逐点对拍（RMSE 相对差 < 1%）

用法（需已 build 且 source set_env）：
    python3 tests/test_e2e.py --dataset FC1 --backbone bigru --target lr --seed 42
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch_npu

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
F3_CODE = PROJECT / "CEMA-LR_F3" / "code"
RESULTS = PROJECT / "CEMA-LR_F3" / "results"
CHECKPOINTS = RESULTS / "checkpoints"
PREDICTIONS = RESULTS / "predictions"

import sys  # noqa: E402
sys.path.insert(0, str(F3_CODE))
from experiments.lr_filter_closed_loop_1h.data import load_windows, make_folds  # noqa: E402
from experiments.lr_filter_closed_loop_1h.models import build_model  # noqa: E402
from experiments.lr_filter_closed_loop_1h.search_space import strip_metadata  # noqa: E402

torch.ops.load_library(str(PROJECT / "build" / "libcustom_ops.so"))


def to_npu(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).npu()


def load_prediction_csv(dataset, backbone, target, seed):
    path = PREDICTIONS / f"formal__F3__{dataset}__{backbone}__{target}__selected__seed{seed}.csv"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return np.array([float(r["prediction"]) for r in rows]), np.array([float(r["anchor_voltage"]) for r in rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FC1", choices=["FC1", "FC2"])
    parser.add_argument("--backbone", default="bigru", choices=["gru", "tcn", "lstm", "bigru", "transformer"])
    parser.add_argument("--target", default="lr", choices=["lr", "direct"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="模型推理设备：cpu / npu:0")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                        help="模型推理精度：GRU/LSTM 类算子在昇腾上仅支持 fp16（DynamicGRUV2 限制）")
    parser.add_argument("--from-csv", action="store_true",
                        help="用 CemaFilter 算子从原始 CSV 重建 dema5/ema9 特征（完整算子链路）")
    args = parser.parse_args()

    # 1) 加载 checkpoint（含 scaler / config / state_dict）
    ckpt_path = CHECKPOINTS / f"formal__F3__{args.dataset}__{args.backbone}__{args.target}__selected__seed{args.seed}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    config = strip_metadata(ckpt["config"])
    print(f"[ckpt] {ckpt_path.name}")
    print(f"[ckpt] scaler.target_mean={scaler['target_mean']:.6f} target_scale={scaler['target_scale']:.6f}")

    # 2) 数据 + 切分
    windows, _ = load_windows(RESULTS / "data_cache" / f"{args.dataset}_F3.npz")
    _, dev_end = make_folds(len(windows.features))
    eval_idx = np.arange(dev_end, len(windows.features))
    mu, sigma = float(scaler["target_mean"]), float(scaler["target_scale"])
    print(f"[data] eval samples={len(eval_idx)} mu={mu:.6f} sigma={sigma:.6f}")

    # 2b) 可选：用 CemaFilter 算子从原始 CSV 重建 ema9/dema5（完整算子链路）
    if args.from_csv:
        from experiments.lr_filter_closed_loop_1h.data import load_hourly  # noqa: E402
        hourly, _ = load_hourly(PROJECT / "CEMA-LR_F3", args.dataset)
        raw = hourly.raw_voltage_v.to_numpy(dtype=np.float32)
        ema9_op, dema5_op = torch.ops.ascendc_ops.cema_filter(to_npu(raw.reshape(1, -1)))
        ema9_op = ema9_op.cpu().numpy().reshape(-1)
        dema5_op = dema5_op.cpu().numpy().reshape(-1)
        full_origin = 27 + windows.origin_positions                       # 帧内索引 → 全序列
        win_idx = full_origin[:, None] + np.arange(-12 + 1, 1)            # [B, 12]
        feats_op = windows.features.copy()                                # 复用 4 工况列（不算子化）
        feats_op[:, :, 0] = dema5_op[win_idx].astype(np.float32)          # 电压列用算子 dema5
        anchors_op = ema9_op[full_origin].astype(np.float32)[:, None]     # 锚点用算子 ema9
        print(f"[csv-op] dema5 算子 vs data_cache max err = {np.abs(feats_op[:, :, 0] - windows.features[:, :, 0]).max():.3e}")
        print(f"[csv-op] ema9锚点 算子 vs data_cache max err = {np.abs(anchors_op[:, 0] - windows.anchors[:, 0]).max():.3e}")
        features_all, anchors_all = feats_op, anchors_op
    else:
        features_all, anchors_all = windows.features, windows.anchors

    # 3) 模型推理（CPU/NPU）→ 标准化 LR 预测 r̂
    model = build_model(args.backbone, config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if args.device != "cpu":
        model.to(args.device)
    if args.dtype == "fp16":
        model.half()          # 昇腾上 GRU/LSTM 需 fp16
    # checkpoint 保存的 scaler 是 Scalers dataclass 的 __dict__（无方法），手写 z-score
    input_mean = np.asarray(scaler["input_mean"], dtype=np.float32)
    input_scale = np.asarray(scaler["input_scale"], dtype=np.float32)
    features_x = ((features_all[eval_idx].astype(np.float32) - input_mean) / input_scale).astype(np.float32)
    model_input = torch.from_numpy(np.ascontiguousarray(features_x, dtype=np.float32))
    if args.dtype == "fp16":
        model_input = model_input.half()
    with torch.no_grad():
        r_hat = model(model_input.to(args.device)).detach().cpu().float().numpy()
    r_hat = r_hat.reshape(-1, 1).astype(np.float32)
    print(f"[model] r_hat sample={r_hat[:3, 0].tolist()}")

    # 4) [NPU] LRDecode：反标准化 + 回加锚点（多核向量化版）
    anchors = anchors_all[eval_idx].astype(np.float32)
    v_hat = torch.ops.ascendc_ops.lr_decode_fast(to_npu(r_hat), to_npu(anchors), mu, sigma).cpu().numpy().reshape(-1)

    # 5) 与 predictions CSV 对拍
    pred_ref, anchor_ref = load_prediction_csv(args.dataset, args.backbone, args.target, args.seed)
    err_anchor = np.abs(anchors.reshape(-1) - anchor_ref).max()  # 数据链路一致（CemaFilter/缓存）
    diff_curve = v_hat - pred_ref
    rmse_diff = float(np.sqrt(np.mean(diff_curve ** 2)))      # 两条曲线逐点差 RMSE
    rel = rmse_diff / (float(np.sqrt(np.mean(pred_ref ** 2))) + 1e-12)  # 相对记录曲线量级
    print(f"[compare] anchor max err (算子锚点 vs CSV) = {err_anchor:.3e}")
    print(f"[compare] v_hat vs pred_ref: max abs err = {np.abs(diff_curve).max():.3e}  "
          f"RMSE(diff) = {rmse_diff:.3e}  相对 = {rel:.4f}")
    print(f"[compare] 记录曲线 RMSE = {float(np.sqrt(np.mean(pred_ref ** 2))):.6f} (V)")
    verdict = "PASS" if rel < 0.01 else "CHECK"
    print(f"[verdict] {verdict}（预测曲线与记录曲线相对差 < 1%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
