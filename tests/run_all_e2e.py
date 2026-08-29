#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
tests/run_all_e2e.py — 批量端到端验证（文档 §11.6 对拍矩阵扩展）。

遍历 FC1/FC2 × 5 骨干 × LR × seed42，逐个跑 tests/test_e2e.py，
验证：checkpoint 模型 → NPU LRDecode → 预测曲线与记录一致（PASS）。

用法（需已 build 且 source set_env）：
    python3 tests/run_all_e2e.py
    python3 tests/run_all_e2e.py --from-csv     # 额外用 CemaFilter 重建特征
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

DATASETS = ("FC1", "FC2")
BACKBONES = ("bigru", "tcn", "lstm", "gru", "transformer")
TARGET = "lr"
SEED = 42
# GRU/LSTM 类算子在昇腾上仅支持 fp16；TCN/Transformer 可 fp32
RNN_BACKBONES = ("bigru", "lstm", "gru")


def run_one(args, dataset, backbone):
    cmd = [
        sys.executable, "tests/test_e2e.py",
        "--dataset", dataset, "--backbone", backbone, "--target", TARGET, "--seed", str(SEED),
    ]
    if args.device != "cpu":
        cmd += ["--device", args.device]
        if args.dtype == "auto":
            cmd += ["--dtype", "fp16" if backbone in RNN_BACKBONES else "fp32"]
        else:
            cmd += ["--dtype", args.dtype]
    if args.from_csv:
        cmd.append("--from-csv")
    proc = subprocess.run(cmd, cwd=str(PROJECT), capture_output=True, text=True)
    ok = proc.returncode == 0 and "[verdict] PASS" in proc.stdout
    return ok, proc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu / npu:0")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16"])
    args = parser.parse_args()

    results = []
    for ds in DATASETS:
        for bb in BACKBONES:
            ok, proc = run_one(args, ds, bb)
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
            results.append((ds, bb, ok))
            print(f"{'PASS' if ok else 'FAIL'}  {ds} {bb} lr seed{SEED}" + ("" if ok else f"  -> {tail}"))

    passed = sum(1 for _, _, ok in results if ok)
    print(f"\n=== 汇总: {passed}/{len(results)} PASS ===")
    if passed != len(results):
        print("失败项:")
        for ds, bb, ok in results:
            if not ok:
                print(f"  - {ds} {bb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
