#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
reproduce_cema_lr.py — 第 0 层：在服务器上复现 F3 正式闭环的单任务结果。

目的：验证本服务器能复现原始实验的数字（训练 + 推理 + 指标），为后续
CemaFilter / LREncode / LRDecode 算子对拍建立 golden 基线。

用法：
    python3 reproduce_cema_lr.py                # 默认 FC1 bigru lr seed42
    python3 reproduce_cema_lr.py --dataset FC2 --backbone tcn --target lr --seed 2024

说明：
    - 直接复用 CEMA-LR_F3 的原装代码（data.py / training.py / models.py / metrics.py），
      不重写任何算法，保证"复现"就是跑同一条管线。
    - 与 results/formal_metrics_seedwise.csv 逐 seed 对拍 RMSE。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ---- 让 F3 原装包可导入 -----------------------------------------------------
F3_CODE = Path(__file__).resolve().parent / "CEMA-LR_F3" / "code"
sys.path.insert(0, str(F3_CODE))

from experiments.lr_filter_closed_loop_1h.data import load_windows, make_folds  # noqa: E402
from experiments.lr_filter_closed_loop_1h.metrics import compute_metrics  # noqa: E402
from experiments.lr_filter_closed_loop_1h.models import paired_initial_states  # noqa: E402
from experiments.lr_filter_closed_loop_1h.training import predict_physical, train_model  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "CEMA-LR_F3" / "results"


def load_reference_row(dataset: str, backbone: str, target_mode: str, seed: int) -> dict | None:
    """从 formal_metrics_seedwise.csv 读出该任务的已记录指标（RMSE 等）。"""
    wanted = f"formal__F3__{dataset}__{backbone}__{target_mode}__selected__seed{seed}"
    with (RESULTS / "formal_metrics_seedwise.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["task_id"] == wanted:
                return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="复现 F3 单任务（正式闭环）")
    parser.add_argument("--dataset", default="FC1", choices=["FC1", "FC2"])
    parser.add_argument("--backbone", default="bigru", choices=["gru", "tcn", "lstm", "bigru", "transformer"])
    parser.add_argument("--target", default="lr", choices=["lr", "direct"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu / npu:0 / cuda:0")
    parser.add_argument("--output", default=None, help="保存复现预测曲线的目录（可选）")
    args = parser.parse_args()

    # 1) 取 selected 配置（与原始 formal 任务完全一致）
    with (RESULTS / "configs" / "F3_independent_selected.json").open(encoding="utf-8") as handle:
        selected = json.load(handle)
    branch = selected[args.dataset][args.backbone][args.target]
    config = branch["config"]
    fixed_epochs = int(branch["fixed_epochs"])
    print(f"[spec] {args.dataset} {args.backbone} {args.target} seed={args.seed}")
    print(f"[spec] config={config} fixed_epochs={fixed_epochs}")

    # 2) 加载 data_cache + 切分（formal = 前 80% 训练 / 后 20% 测试）
    windows, audit = load_windows(RESULTS / "data_cache" / f"{args.dataset}_F3.npz")
    folds, development_end = make_folds(len(windows.features))
    train_idx = np.arange(development_end)
    eval_idx = np.arange(development_end, len(windows.features))
    print(f"[data] windows={len(windows.features)} train={len(train_idx)} eval={len(eval_idx)} "
          f"sha256={audit.get('data_sha256', '')[:16]}...")

    # 3) 严格配对初始权重（direct 用 left，lr 用 right）
    left, right = paired_initial_states(args.backbone, config, args.seed, disable_cudnn=True)
    initial = left if args.target == "direct" else right

    # 4) 训练（fixed_epochs 与原始 formal 一致；CPU/NPU 均可）
    result = train_model(
        args.backbone, config, windows, train_idx, eval_idx, args.target, args.seed, args.device,
        max_epochs=120, validate_every=2, patience_checks=15,
        fixed_epochs=fixed_epochs, initial_state=initial, disable_cudnn=True,
    )
    print(f"[train] best_epoch={result.best_epoch} seconds={result.seconds:.1f}")

    # 5) 测试段推理 + 指标
    prediction = predict_physical(result, windows, eval_idx, args.target, config["batch_size"], args.device)
    truth = windows.targets[eval_idx, 0]
    metrics = compute_metrics(truth, prediction)
    rmse = metrics["rmse"]

    # 6) 与已记录指标对拍
    ref = load_reference_row(args.dataset, args.backbone, args.target, args.seed)
    if ref is None:
        print(f"[warn] 未在 formal_metrics_seedwise.csv 中找到 {args.dataset}_{args.backbone}_{args.target}_seed{args.seed}")
    else:
        ref_rmse = float(ref["rmse"])
        ref_mse = float(ref["mse"])
        print(f"[compare] rmse 复现={rmse:.8f} 记录={ref_rmse:.8f} "
              f"相对差={abs(rmse - ref_rmse) / ref_rmse * 100:.3f}%")
        print(f"[compare] mse  复现={metrics['mse']:.8f} 记录={ref_mse:.8f}")
        print(f"[compare] mae  复现={metrics['mae']:.8f} 记录={float(ref['mae']):.8f}")
        verdict = "PASS" if abs(rmse - ref_rmse) / ref_rmse < 0.01 else "CHECK"
        print(f"[verdict] {verdict}（相对差 < 1% 视为通过）")

    # 7) 可选：保存预测曲线（后续算子端到端对拍用）
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / f"repro__F3__{args.dataset}__{args.backbone}__{args.target}__seed{args.seed}.npz",
            timestamp=windows.timestamps[eval_idx], truth=truth, prediction=prediction,
            raw_truth=windows.raw_targets[eval_idx], anchor=windows.anchors[eval_idx, 0],
        )
        print(f"[save] -> {out / ('repro__...npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
