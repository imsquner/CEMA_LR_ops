# CEMA-LR：PEMFC 电压退化预测的昇腾算子化实现

> **CEMA-LR**（**C**ausal **E**xponential **M**oving **A**verage + **L**evel-**R**esidual）
> 把"先差分、再建模"的思想融入神经网络，在 PEMFC（质子交换膜燃料电池）电压退化预测任务中
> 显著降低 RMSE。本仓库将创新点 **CEMA** 与 **LR** 以 **Ascend C 算子**形式在昇腾 NPU 上落地，
> 并打通"原始数据 → 算子/模型 → 电压预测曲线"的端到端链路。

简体中文

---

## ✨ 效果摘要

| 维度 | 结果 |
|---|---|
| **精度** | 算子 fp32 逐点误差 **< 1e-5**；端到端预测曲线与 F3 记录 **max err ~1e-6**（相对差 < 1%） |
| **算子性能** | `lr_decode_fast` 长序列 **~154x** 加速（n=100000：2.61ms → 0.017ms）；三算子 NPU 耗时占比仅 **~6.3%** |
| **端到端** | 原始 CSV → NPU 算子 + 模型推理 → 预测曲线 **~28.8ms**（含数据缓存） |
| **可复现** | 服务器复现 F3 单任务 **RMSE 相对差 0.001%**（FC1×BiGRU×LR×seed42） |

---

## 💡 想法：为什么是 CEMA + LR

传统统计模型 ARIMA 的核心思想是**先差分、再对差分序列建模**。我们由此提出假设：把"差分 + 重构"
融合进神经网络，可能比直接回归原始电压更利于捕捉退化趋势。

- **CEMA（因果指数移动平均滤波）**：用因果 EMA 对原始电压"去噪成趋势"，作为退化趋势锚点；
  只依赖过去数据（因果性），绝不引入未来信息。
- **LR（Level-Residual）**：神经网络学习的不是原始电压，而是**趋势锚点到下一时刻真实 EMA 之间的
  残差（差分）**。$d_t = E_t - E_{t-1}$，再做标准化 $r_t = (d_t - \mu)/\sigma$ 放大特征；
  推理时反标准化并**回加锚点**还原电压。

```mermaid
flowchart LR
    A[原始电压 V_t] --> B[CemaFilter 因果 EMA]
    B --> C[趋势锚点 E_t]
    C --> D[LR 差分标准化 r_t]
    D --> E[神经骨干<br/>预测 r̂_{t+1}]
    E --> F[LR 反标准化+回加]
    C --> F --> G[电压预测 V̂]
```

---

## 📊 对照实验（F3 正式闭环：`LR` vs `Direct`）

> 口径（2026-08-29 核对代码/数据）：本目录正式闭环 = FC1/FC2 × 5 骨干 × 5 种子（200 checkpoint），
> 统一滤波方案 **F3**（`input=dema5, target=ema9, anchor=ema9`）。`LR` 与 `Direct` 严格配对
> （同一骨干/数据集/超参/初始化权重/种子）。

**5 骨干 RMSE 均值（V），`F3` 滤波**：

| 骨干 | FC1 `LR` | FC1 `Direct` | FC2 `LR` | FC2 `Direct` | 改善率（FC1 / FC2） |
|---|---|---|---|---|---|
| GRU | 0.002187 | 0.004055 | 0.001601 | 0.005944 | 46% / 73% |
| TCN | **0.001011** | 0.002227 | 0.001352 | 0.001876 | 55% / 28% |
| LSTM | 0.002641 | 0.005800 | 0.001571 | 0.008045 | 54% / 80% |
| BiGRU | 0.001210 | 0.005020 | 0.001424 | 0.006499 | 76% / 78% |
| Transformer | 0.001133 | 0.007280 | 0.001284 | 0.009363 | 84% / 86% |

> 数值来源：`CEMA-LR_F3/results/formal_metrics_seedwise.csv`（逐 seed 明细）。`LR` 分支在全部
> 5 骨干 × 2 数据集上均优于 `Direct`，验证了 CEMA-LR 方法的普适性。
> （备注：文档 §4.2 提及的 TCN-Attention-BiGRU / PatchTST 为早期实验，本目录无权重，详见交接文档。）

---

## ⚙️ 算子规格（3 个业务算子）

| 算子 | 创新点 | 功能 | 说明 |
|---|---|---|---|
| `cema_filter` | **CEMA** | 原始电压 → **EMA9**（目标/锚点）+ **DEMA5**（输入特征） | 因果 EMA 递归；推理前端必做 |
| `lr_encode` | **LR** | EMA 序列 → 差分 + 标准化 → LR 标签 | 训练标签/数据管线（可选算子） |
| `lr_decode_fast` | **LR** | 预测 LR → 反标准化 + 回加锚点 → 电压 | 推理末端必做（多核向量化实现） |

- 神经网络骨干（BiGRU/TCN/LSTM/GRU/Transformer）**用 `torch` 实现**，通过 `torch_npu` 在 NPU 上推理，
  **不算子化**——算子化范围只覆盖创新点（CEMA + LR 的数据变换），算子与模型**可分离、可组合**（见交接文档 §7.1）。
- 三个算子均为 **fp32** 计算（精度最优），在 `cema_lr_ops.asc` 中通过 `torch.ops.ascendc_ops.*` 直调。

---

## 🧩 工程结构

```text
cema_lr_ops/
├── CMakeLists.txt            # 编译配置
├── build.sh                  # 构建脚本（--debug/--simulator/--onboard）
├── cema_lr_ops.asc           # 3 个 Ascend C 算子 kernel + torch 注册
├── run.py                    # NPU 烟雾测试（3 算子）
├── reproduce_cema_lr.py      # 服务器复现单任务（RMSE 对拍）
├── requirements.txt          # 依赖
├── README.md                 # 本文件
├── CEMA-LR_F3/               # 实验数据/代码/结果（数据管线、200 checkpoint、预测曲线）
└── tests/
    ├── test_reference.py     # CPU 参考实现（golden）
    ├── test_ops.py           # 3 算子 NPU 随机 + 真实数据对拍
    ├── test_e2e.py           # 端到端（checkpoint 模型 + NPU 算子；--from-csv 完整算子链路）
    ├── run_all_e2e.py        # 批量（FC1/FC2 × 5 骨干 × LR × seed42）
    └── profile_flow.py       # 端到端时序分解（性能分析）
```

---

## 🚀 快速开始

### 环境（实测：CANN 9.0.0 / dav-2201 / 2×Ascend910）

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
export NPU_ARCH=dav-2201
cd /mnt/workspace/gitCode/cann/cann-learning-hub/cema-lr/cema_lr_ops
```

### 编译

```bash
bash build.sh            # 产物：build/libcustom_ops.so
```

### 运行与测试

```bash
python3 run.py                                    # 3 算子 NPU 烟雾测试
python3 tests/test_ops.py                         # 算子随机 + 真实数据对拍
python3 tests/test_e2e.py --dataset FC1 --backbone bigru --target lr --seed 42 --device npu:0 --dtype fp16 --from-csv
python3 tests/run_all_e2e.py --device npu:0       # 批量 10 组合
python3 tests/profile_flow.py --dataset FC1 --backbone bigru --dtype fp16   # 时序分解
python3 reproduce_cema_lr.py                     # 复现 FC1×BiGRU×LR×seed42
```

---

## 📈 性能与调优过程

### 算子级：`LRDecode`（从标量到多核向量化）

| 阶段 | 实现 | n=1156 | n=100000 | 说明 |
|---|---|---|---|---|
| 初版 | GM 标量全量 | 0.035ms | 2.61ms | 正确但线性慢 |
| **优化版** `lr_decode_fast` | 多核 + Level 2 向量化 | **0.018ms** | **0.017ms** | **~154x** |

- **关键设计**：只对 **8 的倍数部分**用 Level 2 向量化，**尾部分给独立标量 kernel**——
  规避 dav-2201 上 `ShiftRight` 不可用、且 Level 2 count 接口对非 8 倍数尾部 mask 不可靠的坑。
  结果逐点 `err = 0`。（故删除初版标量实现，只保留最优版。）

### 流程级：端到端时序分解（`msprof` + `profile_flow.py`）

| 环节 | 优化前 | 优化后 |
|---|---|---|
| 数据加载（CSV+1h 重采样） | 392ms (93.7%) | **2.6ms**（hourly 缓存） |
| 模型推理（BiGRU, fp16） | 24.6ms | 24.6ms |
| 三算子（`cema_filter`/`lr_decode`） | 1.7ms | 1.7ms |
| **合计** | **418ms** | **~28.8ms（~14.5x）** |

`msprof` 算子级占比（端到端全 NPU，FC1×BiGRU×LR×seed42）：

| 算子/操作 | 总耗时 | 占比 |
|---|---|---|
| `DynamicGRUV2`（GRU 模型） | 395us | 41.7% |
| TransData / Transpose / Cast / ReverseV2（模型格式转换） | ~390us | ~40% |
| **`cema_filter`（我们）** | 55.6us | **5.9%** |
| **`lr_decode_vec`（我们）** | 3.7us | **0.4%** |

> **结论**：我们的三算子 NPU 占比仅 ~6.3%，瓶颈在模型推理及其数据格式转换（框架/模型侧），
> 而非算子。若要进一步压缩，可换更轻骨干或模型转 `.om` 用 ACL 推理。

---

## 🎯 精度与量化说明

- 三个算子统一 **fp32**，逐点误差 **< 1e-5**，已与 `data_cache` golden 对拍一致。
- **fp16 量化实测不达标**（电压量级 3.2V 下 fp16 绝对精度约 2e-3V，已超 1e-3 目标；
  EMA 递归累积 + σ 放大使 CemaFilter 达 1.4e-2、LREncode 达 2.5、LRDecode 达 1.8e-3）。
  故算子保持 fp32（与交接文档 §10.1 结论一致）。
- 仅 **GRU/LSTM 类模型**在 NPU 推理时，因昇腾 `DynamicGRUV2` 算子仅支持 fp16，需用 fp16
  （预测曲线仍与记录一致，~1e-6）；非 GRU 骨干（TCN/Transformer）可全程 fp32。

---

## ⚖️ 相关说明

- 完整研究方法、部署手册与验收标准见 **`CEMA-LR_昇腾算子_交接文档.md`**（v0.5）。
- 关键实测坑（`ShiftRight` 不可用、Level 2 尾部 mask 不可靠）已在交接文档 §9.2/§11.8 记录。
- **许可证：GNU AGPL v3.0（`LICENSE`）**。本仓库为**自创算法**，采用最严格的 copyleft 许可
  （AGPL-3.0-only）：任何修改/网络部署（含云服务）须开源完整源码，保护算法不被闭源商用。

## 🤝 致谢

本工程基于此前在 2201/Ascend910B 上的 Ascend C 算子开发经验（GELU/Erf），
并参考昇腾官方 `msopprof` 等仓库的工程组织方式。感谢昇腾社区生态。

