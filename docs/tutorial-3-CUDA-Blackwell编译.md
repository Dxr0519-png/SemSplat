# 教程 3：CUDA / Blackwell / 为什么 gsplat 必须源码编译

> 对应文件：`scripts/build_env.sh`、`scripts/build_gsplat.sh`、`scripts/install_nerfstudio.sh`。
> 读完你能回答：什么叫 sm_120；为什么装了 PyTorch 还会报 "no kernel image"；预编译 wheel 与源码编译差在哪；CUDA 工具链 nvcc/ptxas 各自干嘛。

## 0. 问题

本项目显卡是 **RTX 5060 Laptop**。它属于 NVIDIA 的 **Blackwell** 架构。很多深度学习库的预编译包**不认识它**，装上后一跑就崩。解决它需要的知识链在本文。

---

## 1. 知识一：GPU "算力代号"（compute capability / arch）

NVIDIA GPU 每代有一个算力代号，PyTorch/CUDA 用它决定生成什么机器码：
```
老架构：sm_50/60/70 …   →   Volta sm_70, Ampere sm_80/86, Ada sm_89
RTX 40 系 Ada         →  sm_89
RTX 50 系 Blackwell   →  **sm_120**   （本项目）
```
"sm_120" 意思是"Streaming Multiprocessor，架构 12.0"。**给别的架构编译的内核，在这个 GPU 上无法运行**。

## 2. 知识二：为什么装了 PyTorch 还崩——"no kernel image"

PyTorch 官方从 PyPI 下载的轮子里，CUDA 内核只针对一部分常见算力编好（一般到 sm_90 为止）。在 RTX 50 上执行会报：
```
RuntimeError: CUDA error: no kernel image is available for execution on the device.
```
= "这个 GPU 需要 sm_120，你给我的库只带 sm_90 及以下的核"。

解决办法：装一个**为 sm_120 编译过的 PyTorch**，即 **CUDA 12.8（cu128）配套的 wheel**——`cu128` 是第一个在编译时原生包含 sm_120/Blackwell 内核的版本线。CUDA 版本本身和"轮子带哪些 arch"是两件事：`cu126` 的轮子同样不含 sm_120。所以规则：
```
sm_120 需要：torch wheel 是 cu128（或更新）的
             ↓
本项目锁定 torch==2.7.0+cu128（首个稳定支持 Blackwell 的版本）
```

## 3. 知识三：gsplat 没有 sm_120 的现成轮子 → 源码编译

gsplat 的官方预编译 wheel 长期只针对 **torch 2.4 + CUDA 12.4、且只编到 sm_90**。于是哪怕 PyTorch 已经是 cu128，`pip install gsplat` 拿到的还是不含 sm_120 内核的旧轮子 → 又要崩。

所以必须**从这个 GPU 上现编** gsplat，编译时告诉编译器"目标架构 sm_120"。源码编译需要三样：

| 需要 | 是什么 | 本项目怎么来 |
|---|---|---|
| **nvcc** | CUDA 编译器前端（C 宿主 + 设备代码编译入口） | CUDA ≥12.8 工具链，`~/cuda-12.8/bin/nvcc` |
| **ptxas** | 把 PTX 汇编成 GPU 机器码（cubin）的汇编器 | 随 nvcc 一起 |
| **CUDA 头文件 + 宿主编译器** | `cuda_runtime.h` 等 + g++ | 工具链自带 include；系统有 g++ |

关键：**nvcc 版本 ≥12.8** 才会认 `compute_120`；12.6 的工具链连这个架构名都不认识（会报错或只能退到更老的 arch 让驱动 JIT，慢）。

## 4. 知识四：编译目标 `12.0+PTX`

PyTorch 通过 `TORCH_CUDA_ARCH_LIST` 告诉扩展要编哪些 arch：
```
TORCH_CUDA_ARCH_LIST="12.0+PTX"
```
含义：编 `compute_120`（PTX 中间码 + 原生 sm_120 cubin）。
- 只有 cubin：固定给 sm_120，最快；
- 加 `+PTX`：留一份 PTX，驱动可在需要时 JIT，更稳（向下兼容到同代别的变体）。

## 5. 知识五：gsplat 是怎么"被 pip 编译"的

gsplat 用 PyTorch 的 **cpp_extension** 机制：`setup.py` 里把 `.cu`/`.cpp` 交给 ninja+nvcc 编成 Python 扩展，import 时加载。因此编译时：
1. **能看到 torch** —— 要用 `--no-build-isolation`（否则 pip 会在一个"干净沙箱"里编，看不到我们 venv 里的 torch，报 `No module named 'torch'`）；
2. **能找到头文件** —— nvcc 的 `-I...torch/include -I$CUDA_HOME/include`；
3. **要拿到 glm** —— gsplat 的 CUDA 代码 `#include <glm/glm.hpp>`（开源 GLM 数学库）。**PyPI 的 sdist 把 glm 子模块漏了**，所以直接 `pip install gsplat==1.4.0` 会报 `glm/glm.hpp not found`。解法：`git clone --recursive v1.4.0`（子模块含 glm）再从本地编。

## 6. 知识六：CUDA 工具链怎么免 sudo 装上（runfile）

装整套 CUDA toolkit 传统要 root。NVIDIA 的 **runfile** 允许只装 toolkit 到用户目录：
```bash
curl -fL -o /tmp/cuda.run <developer.download.nvidia.com/.../cuda_12.8.0_570.86.10_linux.run>
bash /tmp/cuda.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8
```
`--toolkit` = 只装编译工具链（不覆盖驱动）；`--toolkitpath` = 用户目录。之后 `nvcc` 在 `~/cuda-12.8/bin/nvcc`。

> 注意下载 URL 要精确：文件名里带的是配套驱动版本号（`570.86.10`），拼错会 404。别用 `sh` 跑 runfile（会因终端参数解析报 `exec: -t: invalid option`），用 `bash` 跑。

## 7. 本项目脚本（把这些串成可复现步骤）

| 脚本 | 职责 | 关键行 |
|---|---|---|
| `scripts/build_env.sh` | venv + torch cu128 | 用 `uv`（系统 python3.12 缺 ensurepip，`python3 -m venv` 会失败）；`torch==2.7.0 torchvision==0.22.0 --index-url .../whl/cu128`；断言 `capability==(12,0)` |
| `scripts/build_gsplat.sh` | 定位 nvcc → git clone v1.4.0 → 源码编 | 候选 `~/cuda-12.8`；`TORCH_CUDA_ARCH_LIST=12.0+PTX pip install --no-build-isolation <clone>` |
| `scripts/install_nerfstudio.sh` | nerfstudio + 依赖 + 本插件 | `pip install nerfstudio==1.1.5`；`pip install -e ".[dev]"` |
| `scripts/smoke_plugin.sh` | 验收 | 算力 → gsplat 一次真实 raster 前反向 → `ns-train --help` 出现 semsplat → 迷你 60 iter 训练 |

> 另两条环境事实（**本机特有**，别当通用知识）：出网要走本地代理 `http://127.0.0.1:7897`（装包/下权重前 `export https_proxy=...`）；`~/.cache/uv` 会缓存大 wheel，重建环境能复用。

## 8. 常见误区

- **"装最新版 CUDA 就能跑"**：PyTorch wheel 里编的内核与你的驱动版本、工具链版本是两套体系。能否跑取决于**wheel 有没有带你的 arch 的核**，其次才是驱动够不够新。
- **"PyPI 的 gsplat 都编过很多架构，应该够用"**：看它的发布 wheel，上限只到 sm_90。
- **"编译是为了新功能"**：这里主要是为了**适配新硬件架构**（内核不存在 = 跑不了，不是慢一点的问题）。
- **"`sh runfile` 和 `bash runfile` 一样"**：不一样，前者会解析错 `-t` 终端参数。

## 9. 延伸学习 / 自查

- 看自己 GPU 架构：`python -c "import torch;print(torch.cuda.get_device_capability())"`；`nvidia-smi`。
- 看 torch wheel 到底带哪些 arch：搜索 `"no kernel image" PyTorch 50系` 相关 issue。
- CUDA 兼容性表（驱动 ↔ 工具链 ↔ runtime）：https://docs.nvidia.com/deploy/cuda-compatibility/

**动手练习**：删掉 `.venv` 里 gsplat，用 `pip install --no-binary gsplat gsplat==1.4.0 --no-build-isolation`（无 glm）复现报错，再用 `build_gsplat.sh` 的 clone 流程成功——把"报错原因"变成肌肉记忆。
