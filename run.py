#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""CEMA-LR 三个算子（cema_filter / lr_encode / lr_decode）的 NPU 烟雾测试。

运行（需已 build 且 source set_env）：
    python3 run.py
"""
import numpy as np
import torch
import torch_npu

torch.ops.load_library("./build/libcustom_ops.so")


def to_npu(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).npu()


def check(name, out, ref, tol=1e-5):
    err = float(np.abs(out - ref).max())
    status = "OK " if err < tol else "FAIL"
    print(f"[{status}] {name}: max_err={err:.3e}")
    return err < tol


def main():
    rng = np.random.default_rng(0)
    ok = True

    # CemaFilter：因果 EMA 滤波（EMA9 + DEMA5）
    raw = rng.standard_normal(1156) * 0.1 + 3.2
    e9, d5 = torch.ops.ascendc_ops.cema_filter(to_npu(raw.reshape(1, -1)))
    e9, d5 = e9.cpu().numpy().reshape(-1), d5.cpu().numpy().reshape(-1)
    # 参考：E_t = a*V + (1-a)*E_{t-1}；DEMA5 = 2*EMA5 - EMA5(2)
    a9, a5 = 0.2, 1.0 / 3.0
    ref9 = np.empty_like(raw); ref9[0] = raw[0]
    for t in range(1, len(raw)):
        ref9[t] = a9 * raw[t] + (1 - a9) * ref9[t - 1]
    e5 = np.empty_like(raw); e5[0] = raw[0]
    for t in range(1, len(raw)):
        e5[t] = a5 * raw[t] + (1 - a5) * e5[t - 1]
    e5s = np.empty_like(raw); e5s[0] = e5[0]
    for t in range(1, len(raw)):
        e5s[t] = a5 * e5[t] + (1 - a5) * e5s[t - 1]
    ref5 = 2.0 * e5 - e5s
    ok &= check("cema_filter ema9", e9, ref9)
    ok &= check("cema_filter dema5", d5, ref5)

    # LREncode：差分 + 标准化（首点 0）
    e = rng.standard_normal((1, 256)) + 3.2
    mu, sg = 0.0, 1.0
    r = torch.ops.ascendc_ops.lr_encode(to_npu(e), mu, sg).cpu().numpy()
    ref_r = np.zeros_like(e); ref_r[:, 1:] = np.diff(e, axis=1) / sg
    ok &= check("lr_encode", r, ref_r)

    # LRDecode：反标准化 + 回加锚点（多核向量化版）
    r_hat = rng.standard_normal(300).astype(np.float32)
    anchor = (rng.standard_normal(300) + 3.2).astype(np.float32)
    v_fast = torch.ops.ascendc_ops.lr_decode_fast(to_npu(r_hat), to_npu(anchor), 0.5, 0.1).cpu().numpy()
    ref_v = r_hat * 0.1 + 0.5 + anchor
    ok &= check("lr_decode_fast (vec)", v_fast, ref_v)

    print("\n" + ("ALL SMOKE TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
