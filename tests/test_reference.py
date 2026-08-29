#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
tests/test_reference.py — CEMA-LR 三个算子的 CPU 参考实现（golden）与自检。

这是 Ascend C 算子的对拍基准（文档 §10.2 步骤 1–3、§11.6 golden 来源 a）：
  - cema_filter : CemaFilter（EMA9 + DEMA5 两路）
  - lr_encode   : LREncode（差分 + 标准化）
  - lr_decode   : LRDecode（反标准化 + 回加锚点）

自检覆盖：
  1. cema_ema / cema_filter 与 pandas ewm(adjust=False) 逐位一致
  2. LR encode→decode 往返还原 EMA 序列
  3. 真实 FC1/FC2 数据：参考实现 == data_cache golden（features/targets/anchors）

运行：
    python3 tests/test_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
F3_CODE = PROJECT / "CEMA-LR_F3" / "code"
RESULTS = PROJECT / "CEMA-LR_F3" / "results"
WARMUP = 27          # protocol.warmup_hours
LOOKBACK = 12        # protocol.lookback


# ============================================================================
# 参考实现（独立 numpy，不依赖 pandas 也能跑）
# ============================================================================

def cema_ema(values, span):
    """因果指数移动平均：E_t = α·V_t + (1-α)·E_{t-1}，E_0 = V_0。
    等价 pandas ewm(span=span, adjust=False).mean()（min_periods=0）。"""
    alpha = 2.0 / (span + 1.0)
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    if values.size == 0:
        return out
    acc = values[0]
    out[0] = acc
    for t in range(1, values.size):
        acc = alpha * values[t] + (1.0 - alpha) * acc
        out[t] = acc
    return out


def cema_filter(raw_voltage, ema9_span=9, dema5_span=5):
    """CemaFilter：EMA9（目标/锚点）+ DEMA5（输入特征）两路输出。"""
    raw = np.asarray(raw_voltage, dtype=np.float64)
    ema9 = cema_ema(raw, ema9_span)
    ema5 = cema_ema(raw, dema5_span)
    ema5_twice = cema_ema(ema5, dema5_span)
    dema5 = 2.0 * ema5 - ema5_twice
    return ema9, dema5


def lr_encode(ema, mu, sigma):
    """LREncode：d_t = E_t - E_{t-1}；r_t = (d_t - μ)/σ。
    首点差分未定义 → 输出 0（与 Ascend C 算子约定一致，见文档 §8.2）。"""
    ema = np.asarray(ema, dtype=np.float64)
    d = np.zeros_like(ema)
    d[1:] = np.diff(ema)
    r = (d - mu) / sigma
    r[0] = 0.0
    return d, r


def lr_decode(r_hat, anchor, mu, sigma):
    """LRDecode：d̂ = r̂·σ + μ；V̂ = anchor + d̂。"""
    r = np.asarray(r_hat, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    d_hat = r * sigma + mu
    v_hat = a + d_hat
    return d_hat, v_hat


# ============================================================================
# 自检 1：EMA / DEMA 与 pandas 完全一致
# ============================================================================

def test_ema_matches_pandas():
    rng = np.random.default_rng(0)
    for span in (3, 5, 7, 9):
        x = rng.standard_normal(5000) * 0.1 + 3.2
        ref = cema_ema(x, span)
        pds = pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()
        np.testing.assert_allclose(ref, pds, rtol=1e-12, atol=1e-14, err_msg=f"span={span}")
    print("[ok] cema_ema == pandas ewm(adjust=False)")


def test_dema_matches_pandas():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000) * 0.1 + 3.2
    ema9, dema5 = cema_filter(x)
    pd_e9 = pd.Series(x).ewm(span=9, adjust=False).mean().to_numpy()
    pd_e5 = pd.Series(x).ewm(span=5, adjust=False).mean().to_numpy()
    pd_d5 = 2.0 * pd_e5 - pd.Series(pd_e5).ewm(span=5, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(ema9, pd_e9, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dema5, pd_d5, rtol=1e-12, atol=1e-14)
    print("[ok] cema_filter(ema9, dema5) == pandas 双层 EMA")


# ============================================================================
# 自检 2：LR 编解码往返
# ============================================================================

def test_lr_roundtrip():
    rng = np.random.default_rng(2)
    ema = np.cumsum(rng.standard_normal(1000) * 1e-3) + 3.2
    d = np.diff(ema)                                   # d = E_{t+1} - E_t
    r = (d - 0.0) / 1e-3                               # LREncode（无 μ/σ 偏移时）
    _, v_hat = lr_decode(r, anchor=ema[:-1], mu=0.0, sigma=1e-3)  # anchor = E_t
    np.testing.assert_allclose(v_hat, ema[1:], rtol=1e-12, atol=1e-12)
    print("[ok] LR encode->decode 往返还原 EMA 序列")


# ============================================================================
# 自检 3：真实 FC 数据（参考实现 == F3 data_cache golden）
# ============================================================================

def test_real_fc_data(dataset: str):
    sys.path.insert(0, str(F3_CODE))
    from experiments.lr_filter_closed_loop_1h.data import load_windows, load_hourly, prepare_filter_frames

    # 从原始 CSV 重建整条 1h 序列（F3 原装管线）
    hourly, audit = load_hourly(PROJECT / "CEMA-LR_F3", dataset)
    raw_full = hourly.raw_voltage_v.to_numpy(dtype=float)

    # 参考实现从完整序列递推（EMA 递归必须从 t=0 起算）
    ema9_full, dema5_full = cema_filter(raw_full)

    # F3 帧 = 丢弃 warmup 前 27 点
    valid = np.ones(len(raw_full), dtype=bool)
    valid[:WARMUP] = False
    frames = prepare_filter_frames(hourly)
    frame = frames["F3"]
    np.testing.assert_allclose(ema9_full[valid], frame.target_voltage.to_numpy(float), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dema5_full[valid], frame.voltage_input.to_numpy(float), rtol=1e-12, atol=1e-14)

    # 与 data_cache npz 的滑窗数据对拍（windows 索引相对 warmup 后的帧）
    windows, _ = load_windows(RESULTS / "data_cache" / f"{dataset}_F3.npz")
    full_origin = WARMUP + np.asarray(windows.origin_positions)          # 帧内索引 → 全序列索引
    idx = full_origin[:, None] + np.arange(-LOOKBACK + 1, 1)             # [B, 12]
    np.testing.assert_allclose(dema5_full[idx], windows.features[:, :, 0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        ema9_full[WARMUP + np.asarray(windows.target_positions)], windows.targets[:, 0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(ema9_full[full_origin], windows.anchors[:, 0], rtol=1e-6, atol=1e-7)
    print(f"[ok] 真实数据 {dataset}: 参考实现 == data_cache golden "
          f"(sha256={audit.get('sha256', '')[:12]}...)")


if __name__ == "__main__":
    test_ema_matches_pandas()
    test_dema_matches_pandas()
    test_lr_roundtrip()
    test_real_fc_data("FC1")
    test_real_fc_data("FC2")
    print("\nALL REFERENCE TESTS PASSED")
