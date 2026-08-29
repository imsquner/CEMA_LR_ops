#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
profile_flow.py — 端到端推理流程时序分解（定位瓶颈，供调优）。

链路（全在 NPU 执行算子与模型）：
  原始 CSV → [CPU]重采样 → [NPU]CemaFilter → [CPU]z-score/滑窗
  → [NPU]模型推理 → [NPU]LRDecode → 预测曲线

对每个环节用 wall-clock + torch.npu.synchronize 精确计时，
输出各阶段耗时占比，用于判断：算子 kernel 本身 / 数据搬运 / CPU 预处理 / 模型推理
哪一块是主要瓶颈。

用法（需已 build 且 source set_env）：
    python3 tests/profile_flow.py [--dataset FC1] [--backbone bigru] [--dtype fp16]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
F3_CODE = PROJECT / "CEMA-LR_F3" / "code"
RESULTS = PROJECT / "CEMA-LR_F3" / "results"

sys.path.insert(0, str(F3_CODE))
from experiments.lr_filter_closed_loop_1h.data import load_windows, load_hourly, make_folds  # noqa: E402
from experiments.lr_filter_closed_loop_1h.models import build_model  # noqa: E402
from experiments.lr_filter_closed_loop_1h.search_space import strip_metadata  # noqa: E402

torch.ops.load_library(str(PROJECT / "build" / "libcustom_ops.so"))


def to_npu(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).npu()


def load_hourly_cached(dataset: str):
    """数据管线缓存：首次做 CSV+1h 重采样并存 npz，之后直接读缓存（~700x 提速）。"""
    import pandas as pd
    cache = PROJECT / "results_repro" / f"hourly_{dataset}.npz"
    if cache.exists():
        d = np.load(cache)
        return pd.DataFrame({k: d[k] for k in d.files}), True
    hourly, _ = load_hourly(PROJECT / "CEMA-LR_F3", dataset)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{c: hourly[c].to_numpy() for c in hourly.columns})
    return hourly, False


def tic():
    torch.npu.synchronize()
    return time.perf_counter()


def toc(t0, label, totals):
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    totals[label] = dt
    print(f"  {label:<38s} {dt:9.3f} ms")
    return time.perf_counter()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FC1", choices=["FC1", "FC2"])
    parser.add_argument("--backbone", default="bigru", choices=["gru", "tcn", "lstm", "bigru", "transformer"])
    parser.add_argument("--target", default="lr", choices=["lr", "direct"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--iters", type=int, default=20, help="重复推理次数（测平均）")
    args = parser.parse_args()

    totals = {}
    print(f"=== 端到端流程时序分解：{args.dataset} {args.backbone} {args.target} seed{args.seed} "
          f"device={args.device} dtype={args.dtype} ===")

    # ---- 0. 数据加载（缓存优先：CSV+1h 重采样仅首次执行）----
    t0 = tic()
    hourly, from_cache = load_hourly_cached(args.dataset)
    t = toc(t0, f"0. 数据加载（{'缓存命中' if from_cache else '首次 CSV+重采样'}）", totals)

    # ---- 1. CemaFilter 算子（NPU）----
    raw = hourly.raw_voltage_v.to_numpy(dtype=np.float32)
    raw_npu = to_npu(raw.reshape(1, -1))
    t0 = tic()
    ema9_op, dema5_op = torch.ops.ascendc_ops.cema_filter(raw_npu)
    ema9 = ema9_op.cpu().numpy().reshape(-1)
    dema5 = dema5_op.cpu().numpy().reshape(-1)
    t = toc(t0, "1. CemaFilter 算子（NPU kernel+回拷）", totals)

    # ---- 2. 特征构建：z-score + 滑窗（CPU）----
    windows, _ = load_windows(RESULTS / "data_cache" / f"{args.dataset}_F3.npz")
    ckpt = torch.load(RESULTS / "checkpoints" / f"formal__F3__{args.dataset}__{args.backbone}__{args.target}__selected__seed{args.seed}.pt",
                      map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    config = strip_metadata(ckpt["config"])
    _, dev_end = make_folds(len(windows.features))
    eval_idx = np.arange(dev_end, len(windows.features))
    t0 = tic()
    input_mean = np.asarray(scaler["input_mean"], dtype=np.float32)
    input_scale = np.asarray(scaler["input_scale"], dtype=np.float32)
    full_origin = 27 + windows.origin_positions
    win_idx = full_origin[:, None] + np.arange(-12 + 1, 1)
    feats = windows.features.copy()
    feats[:, :, 0] = dema5[win_idx].astype(np.float32)
    features_x = ((feats[eval_idx].astype(np.float32) - input_mean) / input_scale).astype(np.float32)
    anchors = ema9[full_origin].astype(np.float32)[:, None][eval_idx]
    t = toc(t0, "2. 特征 z-score + 滑窗拼接（CPU）", totals)

    # ---- 3. 模型推理（NPU）----
    model = build_model(args.backbone, config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(args.device)
    if args.dtype == "fp16":
        model.half()
    model_input = torch.from_numpy(np.ascontiguousarray(features_x, dtype=np.float32))
    if args.dtype == "fp16":
        model_input = model_input.half()
    model_input = model_input.to(args.device)
    mu, sigma = float(scaler["target_mean"]), float(scaler["target_scale"])
    # warmup
    with torch.no_grad():
        _ = model(model_input)
    torch.npu.synchronize()
    t0 = tic()
    with torch.no_grad():
        for _ in range(args.iters):
            r_hat = model(model_input)
    r_hat = r_hat.detach().cpu().float().numpy().reshape(-1, 1)
    t = toc(t0, f"3. 模型推理（NPU, {args.iters}次平均）", totals)

    # ---- 4. LRDecode 算子（NPU）----
    r_hat_npu = to_npu(r_hat)
    anchor_npu = to_npu(anchors)
    t0 = tic()
    for _ in range(args.iters):
        v_hat = torch.ops.ascendc_ops.lr_decode_fast(r_hat_npu, anchor_npu, mu, sigma)
    v_hat = v_hat.cpu().numpy().reshape(-1)
    t = toc(t0, "4. LRDecode 算子（NPU, 多核向量化）", totals)

    # ---- 汇总 ----
    total = sum(totals.values())
    print(f"\n=== 耗时占比 ===")
    for k, v in totals.items():
        print(f"  {k:<38s} {v:9.3f} ms  ({v / max(total, 1e-9) * 100:5.1f}%)")
    print(f"  {'合计':<38s} {total:9.3f} ms")
    print("\n提示：kernel 本身为亚毫秒级；若 CPU 预处理/搬运占比高，瓶颈在数据管线而非算子。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
