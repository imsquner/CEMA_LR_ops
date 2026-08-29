# cema_lr_ops注册自定义算子直调样例
## 概述
本样例展示了如何使用PyTorch的torch.library机制注册自定义算子，并通过<<<>>>内核调用符调用核函数，以简单的Add算子为例，实现两个向量的逐元素相加。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
- Atlas A5 训练系列产品（Ascend 950）

## A4 vs A5（910B-class vs Ascend 950 / Atlas A5）

从 **Huawei OP DevTools** 侧边栏编译/运行时，扩展会在子进程中自动注入 `SOC_VERSION`、`NPU_ARCH` 等芯片环境变量（`CMakeLists.txt` 通过 `$ENV{NPU_ARCH}` 读取）。

| 环境 | spawn 注入（扩展） | 手动终端 `bash build.sh` |
|------|-------------------|-------------------------|
| A4 / 910B-class | `SOC_VERSION=ascend910b`，`NPU_ARCH=dav-2201` | 须 `export` 上述变量 |
| A5 / 950 | `SOC_VERSION=ascend950`，`NPU_ARCH=dav-3510`；仿真器可用 `Ascend950PR_9589` | 须 `export` 上述变量 |

工作区设置 **`op-devtools.environment.socVersion`** 为 `ascend950` 时扩展会解析 A5 默认值。详见 [`docs/ascend-env-spec.md`](../../../../../docs/ascend-env-spec.md)。

## 基于一站式算子开发平台进行算子开发
基于一站式算子开发平台快速完成算子创建、开发、异常检测、性能调优 \
指导文档：**[https://gitcode.com/opdevtools/plugin_release](https://gitcode.com/opdevtools/plugin_release)**

## 编译运行

如何快速编译算子
| 模式 | 命令 | 用途 |
|------|------|------|
| 普通编译 | `bash build.sh` | 日常开发验证 |
| 异常检测 | `bash build.sh --mssanitizer` | 问题检测 |
| 上板调优 | `bash build.sh --onboard` | 真实环境性能分析 |
| 仿真调优 | `bash build.sh --simulator <SOC版本>`，或先 `export SOC_VERSION=<SOC版本>` 再 `bash build.sh --simulator`（与工具链/IDE 解析的 SOC 一致） | 指令级详细分析 |

## 目录结构介绍
```
├── cema_lr_ops
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── add_custom_test.py      // PyTorch调用自定义算子的测试脚本
│   └── add_custom.asc          // Ascend C算子实现 & 自定义算子注册
```

## 算子描述
- 算子功能：  
  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
  ```
  z = x + y
  ```

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- 算子实现：

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

- 自定义算子注册：

  本样例在add_custom.asc中定义了一个名为ascendc_ops的命名空间，并在其中注册了ascendc_add函数。

  PyTorch提供`CEMA_LR_OPS`宏作为自定义算子注册的核心接口，用于创建并初始化自定义算子库，注册后在Python侧可以通过`torch.ops.namespace.op_name`方式进行调用，例如：
  ```c++
  CEMA_LR_OPS(ascendc_ops, m) {
      m.def(ascendc_add"(Tensor x, Tensor y) -> Tensor");
  }
  ```

  `CEMA_LR_OPS_IMPL`用于将算子逻辑绑定到特定的DispatchKey（PyTorch设备调度标识）。针对NPU设备，需要将算子实现注册到PrivateUse1这一专属的DispatchKey上，例如：
  ```c++
  CEMA_LR_OPS_IMPL(ascendc_ops, PrivateUse1, m)
  {
      m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
  }
  ```
  在ascendc_add函数中通过`c10_npu::getCurrentNPUStream()`函数获取当前NPU上的流，并通过内核调用符<<<>>>调用自定义的Kernel函数add_custom，在NPU上执行算子。

- Python测试脚本

  在add_custom_test.py中，首先通过`torch.ops.load_library`加载生成的自定义算子库，调用注册的ascendc_add函数，并通过对比NPU输出与CPU标准加法结果来验证自定义算子的数值正确性。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
- 请参考与您当前使用的版本配套的[《Ascend Extension for PyTorch
软件安装指南》](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)，获取PyTorch和torch_npu详细的安装步骤。  

- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/ascend-toolkit/set_env.sh
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/ascend-toolkit/set_env.sh
    ```
- 样例执行
  ```bash
  mkdir -p build && cd build;     # 创建并进入build目录
  cmake ..;make -j;               # 编译工程
  python3 ../add_custom_test.py   # 执行测试脚本
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  Ran 1 test in **s.
  OK
  ```
