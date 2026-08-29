#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
tests/test_ops.py — 在 NPU 上运行三个 Ascend C 算子，并与 CPU 参考实现对拍。

覆盖（文档 §11.6 对拍矩阵）：
  1. LRDecode  随机输入（T=1/7/12/2048/4096）
  2. LREncode  随机输入（[B,T]=[1,12]/[3,1156]/[8,2048]，含多核边界长度）
  3. CemaFilter随机输入（[B,T]=[1,12]/[1,1156]/[4,256]）
  4. 真实 FC1 数据：CemaFilter 整条电压 vs 参考 + data_cache golden；
     LREncode 差分标准化；LRDecode 用真实 scaler/锚点还原 targets

运行（需已 build 且 source set_env）：
    python3 tests/test_ops.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
F3_CODE = PROJECT / "CEMA-LR_F3" / "code"
RESULTS = PROJECT / "CEMA-LR_F3" / "results"
WARMUP = 27

sys.path.insert(0, str(HERE))
from test_reference import cema_filter, lr_encode, lr_decode  # noqa: E402

torch.ops.load_library(str(PROJECT / "build" / "libcustom_ops.so"))


def to_npu(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).npu()


def max_err(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def ref_cema_batch(raw):
    raw = np.asarray(raw, dtype=np.float64)
    e9 = np.empty_like(raw)
    d5 = np.empty_like(raw)
    for b in range(raw.shape[0]):
        e9[b], d5[b] = cema_filter(raw[b])
    return e9, d5


def ref_lr_encode_batch(e, mu, sigma):
    e = np.asarray(e, dtype=np.float64)
    r = np.empty_like(e)
    for b in range(e.shape[0]):
        _, r[b] = lr_encode(e[b], mu, sigma)
    return r


# ---------------- 1. LRDecode 随机（多核向量化版，唯一实现） ----------------
def test_lr_decode():
    rng = np.random.default_rng(3)
    for n in (1, 7, 8, 12, 2048, 2050, 4096, 100000):
        r = rng.standard_normal(n)
        a = rng.standard_normal(n) + 3.2
        mu, sg = 0.5, 0.1
        out = torch.ops.ascendc_ops.lr_decode_fast(to_npu(r), to_npu(a), mu, sg).cpu().numpy()
        ref = r * sg + mu + a
        err = max_err(out, ref)
        assert err < 1e-6, f"lr_decode_fast n={n} err={err}"
    print("[ok] LRDecode 多核向量化（n=1/7/8/12/2048/2050/4096/100000）")


# ---------------- 2. LREncode 随机 ----------------
def test_lr_encode_random():
    rng = np.random.default_rng(1)
    # 随机输入的差分 d~O(1)：若 σ 取真实值(8.7e-4)会被放大到 r~O(1000)，
    # fp32 误差随之放大（病态测试）。随机对拍用 σ=1.0；真实量级在
    # test_real_fc1 中用真实 μ/σ 严格对拍（r~O(0.1)）。
    mu, sg = 0.0, 1.0
    for (b, t) in ((1, 12), (3, 1156), (8, 2048)):
        e = rng.standard_normal((b, t)) + 3.2
        out = torch.ops.ascendc_ops.lr_encode(to_npu(e), mu, sg).cpu().numpy()
        ref = ref_lr_encode_batch(e, mu, sg)
        err = max_err(out, ref)
        assert err < 1e-5, f"lr_encode {b}x{t} err={err}"
    print("[ok] LREncode 随机（[1,12]/[3,1156]/[8,2048], σ=1.0）")


# ---------------- 3. CemaFilter 随机 ----------------
def test_cema_filter_random():
    rng = np.random.default_rng(2)
    for (b, t) in ((1, 12), (1, 1156), (4, 256)):
        raw = rng.standard_normal((b, t)) * 0.1 + 3.2
        e9, d5 = torch.ops.ascendc_ops.cema_filter(to_npu(raw))
        e9, d5 = e9.cpu().numpy(), d5.cpu().numpy()
        r9, r5 = ref_cema_batch(raw)
        assert max_err(e9, r9) < 1e-5, f"cema ema9 {b}x{t} err={max_err(e9, r9)}"
        assert max_err(d5, r5) < 1e-5, f"cema dema5 {b}x{t} err={max_err(d5, r5)}"
    print("[ok] CemaFilter 随机（[1,12]/[1,1156]/[4,256]）")


# ---------------- 4. 真实 FC1 数据 ----------------
def test_real_fc1():
    sys.path.insert(0, str(F3_CODE))
    from experiments.lr_filter_closed_loop_1h.data import (  # noqa: E402
        load_windows, make_folds, load_hourly, prepare_filter_frames)
    from experiments.lr_filter_closed_loop_1h.training import fit_scalers  # noqa: E402

    # 4a. CemaFilter：整条 FC1 电压序列（含 warmup，从原始 CSV 重建）
    hourly, _ = load_hourly(PROJECT / "CEMA-LR_F3", "FC1")
    raw = hourly.raw_voltage_v.to_numpy(dtype=np.float64)
    e9_op, d5_op = torch.ops.ascendc_ops.cema_filter(to_npu(raw.astype(np.float32)))
    e9_op = e9_op.cpu().numpy()
    d5_op = d5_op.cpu().numpy()
    r9, r5 = cema_filter(raw)
    assert max_err(e9_op, r9) < 1e-6, f"cema ema9 整条 err={max_err(e9_op, r9)}"
    assert max_err(d5_op, r5) < 1e-6, f"cema dema5 整条 err={max_err(d5_op, r5)}"

    # 4b. 与 F3 data_cache golden 对拍（丢弃 warmup 27h 后）
    frames = prepare_filter_frames(hourly)
    frame = frames["F3"]
    valid = np.ones(len(raw), dtype=bool)
    valid[:WARMUP] = False
    assert max_err(e9_op[valid], frame.target_voltage.to_numpy(float)) < 1e-6
    assert max_err(d5_op[valid], frame.voltage_input.to_numpy(float)) < 1e-6

    # 4c. LREncode：对整条 ema9 差分标准化（用 F3 训练集 scaler）
    windows, _ = load_windows(RESULTS / "data_cache" / "FC1_F3.npz")
    folds, dev_end = make_folds(len(windows.features))
    scaler = fit_scalers(windows, np.arange(dev_end), "lr")
    mu, sg = scaler.target_mean, scaler.target_scale
    r_op = torch.ops.ascendc_ops.lr_encode(
        to_npu(e9_op.astype(np.float32)), mu, sg).cpu().numpy()
    _, r_ref = lr_encode(e9_op, mu, sg)
    assert max_err(r_op, r_ref) < 1e-6, f"lr_encode 整条 err={max_err(r_op, r_ref)}"

    # 4d. LRDecode：用真实 r̂ + 窗口末锚点还原 targets（完整滑窗数据）
    r_hat = (windows.targets.astype(np.float64) - windows.anchors.astype(np.float64) - mu) / sg
    v_op = torch.ops.ascendc_ops.lr_decode_fast(
        to_npu(r_hat), to_npu(windows.anchors), mu, sg).cpu().numpy()
    assert max_err(v_op, windows.targets) < 1e-5, f"lr_decode 还原 targets err={max_err(v_op, windows.targets)}"
    print("[ok] 真实 FC1: CemaFilter(整条) + LREncode + LRDecode == data_cache golden")


if __name__ == "__main__":
    test_lr_decode()
    test_lr_encode_random()
    test_cema_filter_random()
    test_real_fc1()
    print("\nALL NPU OP TESTS PASSED")
