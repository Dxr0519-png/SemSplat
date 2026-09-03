# 项目模块详解（按创建流程）

> 本文按「这个工程是怎么一步步搭起来」的顺序，逐模块讲清每部分是什么、负责什么、为什么这么设计。
> 对应实现为：`semsplat`（nerfstudio 插件，开放词汇语义 3DGS）。总述与命令见 [usage.md](usage.md)。
>
> **想深入学习某一块背后的知识**：对应阅读 `docs/tutorial-N-*.md` 系列（每块一个教学文档，含知识讲解 + 本项目实现），详见 [README 文档索引](../README.md)。

## 0. 一句话总览

整个系统 = **普通的 Splatfacto 三维重建**（gsplat 光栅化 RGB）+ 在**每个高斯上加一个低维语义向量 `sem_features`**，用一个第二路 ND 光栅化把它渲染成图，再与 **冻结的 OpenAI-CLIP 图像 patch 特征**（教师）做逐 patch 对齐蒸馏；推理时拿 CLIP 文本嵌入去和逐高斯语义向量做余弦相似度，把分数再光栅化成**语义掩码/热图**，实现开放词汇查询。

```
图片+位姿 ──(Splatfacto RGB 损失)──▶ 高斯几何/颜色
                                       └─ + gauss_params["sem_features"] [N,32]
训练损失 = RGB 损失 + γ·dist( feature_head(光栅化(sem_features)), CLIP_patch_teacher(GT图) )
查询    : 文本 ──CLIP text──▶ t ; 逐高斯 f → feature_head(f) ; cos(f, t) → 掩码/热图/PLY
```

| 阶段（创建顺序） | 目录/文件 | 职责 | 对应里程碑 |
| --- | --- | --- | --- |
| ① 打包与注册 | `pyproject.toml`、`Makefile` | 让 nerfstudio 发现方法、提供常用目标 | S0 |
| ② 环境 | `scripts/build_env.sh` `build_gsplat.sh` `install_nerfstudio.sh` `smoke_plugin.sh` | 搭出可在 RTX 50 (sm_120) 上跑的环境 | M0 |
| ③ 数据 | `semsplat/data/replica_to_transforms.py`、`scripts/prepare_replica.sh` `make_depth_init_ply.py` | Replica RGB-D → nerfstudio 数据集 + 初始化点云 | M1 |
| ④ 方法核心 | `semsplat/semsplat_config.py`、`semsplat/semsplat_model.py` | 模型子类：语义特征 + 蒸馏损失 | M2 |
| ⑤ 教师 | `semsplat/teachers/clip_local_teacher.py` | 冻结 CLIP 图像/文本编码器 | M2 |
| ⑥ 推理 | `semsplat/inference/*` | 开放词汇查询 CLI、PLY、mIoU 预留 | M3 |
| ⑦ 辅助脚本 | `scripts/make_mini_scene.py` | 合成冒烟数据 | M0 |
| ⑧ 测试 | `tests/*` | 注册/光栅/致密化/蒸馏冒烟 | M0/M2 |

---

## ① 打包与注册（S0）

### `pyproject.toml`
插件的“身份证 + 接线图”，三件关键事：
- **[project.entry-points."nerfstudio.method_configs"]**：把包里的两个 `MethodSpecification` 对象注册成 `ns-train` 的子命令（`semsplat` / `semsplat-replica`）。这是 nerfstudio 发现第三方方法的官方机制。
- **[project.scripts]**：提供命令行 `semsplat-query`，指向 `semsplat.inference.query_cli:main`。
- **依赖**：固定 `nerfstudio==1.1.5`、`gsplat==1.4.0`、`torch>=2.7`，外加 `transformers`（CLIP 教师）、`plyfile`。

### `Makefile`
把最常用命令串成目标：`env gsplat nerfstudio install smoke mini mini-train test data-demo demo-train query help`。注意：
- `PROXY`（默认 `http://127.0.0.1:7897`）会注入训练/下载进程，因为本机出网需走本地代理。
- 训练统一走 `.venv/bin/ns-train`（而非 `python -m ns.train`）。

### `README.md`
顶层介绍 + 最短使用路径。本 `docs/` 下的 `architecture.md`（本文）与 `usage.md` 是其深化。

---

## ② 环境搭建（M0，风险最高的一步）

GPU 是 **RTX 5060 Laptop（Blackwell，sm_120，8GB）**，而 gsplat 预编译 wheel 最高只编到 sm_90，**没有 sm_120 内核**；PyTorch 也必须要 cu128 版才有 sm_120。所以环境的核心是“**从源码给 sm_120 编译 gsplat**”。

### `scripts/build_env.sh`
1. 用 `uv` 建 `.venv`（系统 python3.12 缺 ensurepip，`python3 -m venv` 会失败；`uv` 免 sudo）。
2. 装 `torch==2.7.0 + torchvision==0.22.0`（`--index-url .../whl/cu128`）→ 原生 sm_120。
3. 断言 `torch.cuda.get_device_capability()==(12, 0)`。

### `scripts/build_gsplat.sh`
1. 找一个能编 `compute_120` 的 nvcc（CUDA ≥12.8）：优先 `$HOME/cuda-12.8`（由 NVIDIA runfile 装到用户目录，免 sudo），否则找系统 nvcc，找不到则打印安装指引并退出。
2. 从 **git clone gsplat v1.4.0（`--recursive`）** 构建 —— 关键坑：PyPI sdist 不含 vendored `glm` 头文件子模块，直接 `pip install gsplat==1.4.0` 会报 `glm/glm.hpp not found`。
3. `TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install --no-build-isolation <clone>`：`--no-build-isolation` 让构建能看到 venv 里的 torch（gsplat 的 setup 要 import torch）。
4. 打印 gsplat 版本与设备算力冒烟。

> 如果换新机器：不再需要 `build_gsplat.sh` 时，可改用 `pip install --no-binary gsplat gsplat==1.4.0 --no-build-isolation`（该机器有完整 CUDA ≥12.8 工具链 + glm 系统包时）。

### `scripts/install_nerfstudio.sh`
`pip install nerfstudio==1.1.5`（其依赖会被逐一装齐，含 `nerfacc==0.5.2`、`viser==0.2.7` 等）→ `transformers`/`plyfile` → `pip install -e ".[dev]"` 装入本插件并校验 import。

### `scripts/smoke_plugin.sh`
四步验收：算力 → gsplat 一次真实 raster 前反向 → `ns-train --help` 里出现 `semsplat` → 在迷你场景上跑 60 iter 训练并成功退出。

---

## ③ 数据（M1）

目标是“免 COLMAP”：用 Replica 预处理序列自带的 GT 位姿直接把 RGB 帧变成 nerfstudio 的 `transforms.json`。

### `scripts/prepare_replica.sh`
下载并解压 NICE-SLAM 预处理 Replica zip（默认 `Demo.zip`，157MB 单场景，数据源 `cvg-data.inf.ethz.ch/nice-slam/data/Demo.zip`），支持 `--scene` 只解压某一场景目录。

该数据集的目录形态（被 `replica_to_transforms.py` 适配）：
```
<scene>/frames/
  color/<id>.jpg        # 1296×968 RGB
  depth/<id>.png        # 640×480 uint16，单位毫米
  pose/<id>.txt         # 16 个 float = 4×4 相机到世界(c2w)
  intrinsic/intrinsic_color.txt / intrinsic_depth.txt   # 3×4 内参 K
```

### `semsplat/data/replica_to_transforms.py`
把上面的原始帧 → nerfstudio 数据集目录 `<out>/`（内含 `transforms.json` + `images/` 符号链接）：
- `discover_color / discover_poses`：**打分式自动探测** color/pose 目录（跳过 `depth`），也支持 `--color-dir/--pose-dir/--intrinsics-file` 显式指定。
- 内参：从 `intrinsic_color.txt`（4×4/3×4/3×3 均可）解析 `fx fy cx cy`，缺省则读图取 `w h`。
- **位姿保真**：`semsplat-replica` 方法用的是 `orientation_method=none, center_method=none, auto_scale_poses=False`，因此这里把 Replica 的 c2w **原样**写进 `transform_matrix`，不做自动朝向/缩放，保证度量坐标一致。
- `--subset/--offset`：抽帧（本项目用 300 帧）；`--probe`：只打印探测结果不改写。

### `scripts/make_depth_init_ply.py`
**为什么要它**：纯随机初始化（`random_init`）在无 SfM 点的大场景上收敛差（demo3 前我们发现 main_loss 只降到 ~0.30）。改为从 depth 反投影出初始化点云喂给 splatfacto，几何快速变好（main_loss ~0.13、alpha 覆盖 ~1.0）。

实现要点：
- 用 `intrinsic_depth.txt`（640×480）+ Replica 位姿，按 OpenGL 约定（相机 −Z 朝前、y 向上）把像素反投影成世界点，做体素降采样（默认 2cm → 本项目用 4cm，约 8.6 万点）。
- 写成 `<ns>/points3d.ply`，并往 `transforms.json` 加 `"ply_file_path": "points3d.ply"`。
- nerfstudio 的 `NerfstudioDataParser(load_3D_points=True)` 会把它读进 dataparser metadata `points3D_xyz/rgb`，pipeline 再作为 `seed_points` 传给 Splatfacto（见 `base_pipeline.py` 的 `seed_points=seed_pts`）。
- 约定假设与我们的训练自洽：跨帧深度重投影误差 ~2%（位姿一致，无镜像问题）。

---

## ④ 方法核心（M2）

### `semsplat/semsplat_model.py`（核心文件）
内含一个配置 dataclass 与一个模型子类。

**`SemsplatSplatfactoModelConfig(SplatfactoModelConfig)`** —— 新增的语义超参：
- 语义场：`semantic_feature_dim`(K=32)、`semantic_head_dim`(M=512)、`semantic_start_iter`(3000)、`semantic_loss_type`(cos/l1/mse)、`semantic_loss_mult`、`semantic_detach_geometry/opacity`。
- 教师：`teacher_model_name`(HF checkpoint)、`teacher_cache_fp16`、`teacher_cache_max_entries`。

**`SemsplatModel(SplatfactoModel)`** —— 相对父类的 4 处增量：

1. `populate_modules()`：`super()` 之后往 `self.gauss_params["sem_features"]` 塞一个 `[N,K]` 的 `Parameter`（小随机初始化），并注册共享解码头 `self.feature_head = nn.Linear(K, M)`。要点：
   - **放进 `gauss_params`** 是因为 gsplat `DefaultStrategy` 对参数字典里每个 key 都做同一套分裂/克隆/剪枝 → `sem_features` 行数与 `means` 自动保持同步（我们的单测锁死这一点）。
   - `feature_head` 是共享的、**不放**进 `gauss_params`（否则会被当成逐高斯参数），它是“Feature-3DGS 的 1×1 卷积上采样器”的线性等价物。
2. `get_gaussian_param_groups() / get_param_groups()`：父类这两个方法返回的是**硬编码** 6 个 key（`means…opacities`），必须重写补上 `sem_features` 与 `semantic_head`，否则没有对应优化器组、且致密化时会因 `optimizers[name]` 缺失而崩。
3. `get_outputs()`：**整段复制 v1.1.5 Splatfacto 的 `get_outputs` 主体**（nerfstudio 已被我们锁在 1.1.5，所以复制无漂移风险；源文件版本写在了模块 docstring），并在 RGB 光栅化之后追加**语义特征第二路光栅化**：
   - `gsplat.rasterization(colors=sem_features, sh_degree=None, render_mode="RGB", backgrounds=None)`：`sh_degree=None` 时 gsplat 会把 `[N,K]` 当作任意 D 维逐顶点属性做 alpha 混合（ND pass）。
   - 返回是 pre-multiplied（已乘 alpha），所以除以 `alpha` 得到**期望特征图** `outputs["sem_feature_map"]`（仿 gsplat 的 “ED/期望深度” 归一）。几何/不透明度默认 `detach()`，让语义损失只优化语义向量，不动几何。
   - 门控：仅在 `training` 且 `step >= semantic_start_iter` 时计算。
4. `get_loss_dict()`：`super()` 得到 RGB/SSIM 损失后追加语义损失——
   - 从 `get_gt_img(batch["image"])`（与 RGB 损失同一分辨率）经 `_get_teacher()` 惰性构建的 CLIP 教师取 patch 网格 `[gH,gW,M]`（缓存按 `self._last_cam_idx` 即 `camera.metadata["cam_idx"]`，避免每步重算）。
   - `pred = feature_head(sem_feature_map)` 后用 `adaptive_avg_pool2d` 对齐到教师网格（**不要求图像尺寸被 16 整除**，规避早期 1/4 分辨率不是 16 倍数的坑）。
   - 双方 L2 归一后按 `semantic_loss_type` 算 cos/l1/mse，在 `alpha>0.5` 的有效 patch 上取均值，×`semantic_loss_mult` 记为 `semantic_loss`。

### `semsplat/semsplat_config.py`
导出两个 `MethodSpecification`（pyproject 入口点指向这里）：
- `semsplat_method`：通用版（沿用 stock 的自动朝向/居中/缩放，适合 COLMAP/一般数据）。
- `semsplat_replica_method`：Replica 专用（dataparser `orientation=none/center=none/auto_scale=False` + 大一点 `random_scale=5.0`）。
两者共用 `_make_trainer()`：把 stock splatfacto 的优化器组原样搬来，**再追加 `sem_features` 与 `semantic_head` 两个 Adam 组**（与模型 `get_param_groups()` 一一对应，缺一不可）。

> 为什么 `semantic_loss` 一开始可能为 0：若几何差、`alpha_pooled>0.5` 的 patch 为空，代码会把损失置零而非 NaN —— 语义质量依赖几何质量（这也是我们做 depth 初始化的原因）。

---

## ⑤ 教师（M2）

### `semsplat/teachers/clip_local_teacher.py`
封装 **HF transformers 的 OpenAI CLIP ViT-B/16** 作稠密教师：
- **为何用 transformers 而非裸 OpenCLIP**：需要 ViT 的逐 patch embedding（OpenCLIP 不给）、且输入非 224 时要位置编码插值；HF `CLIPVisionModel(interpolate_pos_encoding=True)` 两者都直接支持。权重就是 OpenAI 的，即“纯 CLIP 局部特征”。
- `ClipLocalTeacher.dense_features(image)`：把 `[H,W,3]` 图 resize 到 16 的倍数 → `vision_model` → 取 `last_hidden_state` 去掉 `[CLS]` reshape 成 `[gH,gW,768]` → **逐 patch 过 `visual_projection`**（768→512，对齐到文本空间）→ L2 归一 → `[gH,gW,512]`。
- `compute(image, key)`：LRU 缓存（`teacher_cache_fp16`、`teacher_cache_max_entries`，按 `cam_idx` 命中）。
- `encode_text_prompts(...)`：文本侧用 `text_model(...).pooler_output @ text_projection`（**不是** `get_text_features`，那个返回的是 tuple 会踩 `AttributeError`），L2 归一后每个 prompt 得 `[512]`。
- 全模型 `eval()` + 冻结；进程首次使用需联网下权重到 `~/.cache/huggingface`（本机需带代理）。

---

## ⑥ 推理 / 开放词汇查询（M3）

### `semsplat/inference/scoring.py`
把“分数渲染”复用到与训练一致的方式：
- `per_gaussian_semantics(model)`：`feature_head(gauss_params["sem_features"])` 再 L2 归一 → `[N,M]`。
- `rasterize_class_scores(model, camera, [N,C])`：对任意逐高斯分数张量做 ND 光栅化（`sh_degree=None`, pre-multiplied, 除以 alpha）→ `[H,W,C]`。
- `semantic_maps_from_prompts`：`sem @ text_emb.T` 得 `[N,C]` 分数再渲染。

### `semsplat/inference/query_cli.py`（`semsplat-query`）
主流程：
1. monkeypatch `torch.load(..., weights_only=False)`（torch≥2.6 默认 True 读不了 nerfstudio 存档）。
2. `nerfstudio.utils.eval_utils.eval_setup(config)` 装载 pipeline（注意它**返回 `(config, pipeline, ckpt_path, step)`**，第一个是 config）。
3. 文本 → CLIP 嵌入；`per_gaussian_semantics` 对全场算最高类与置信度 →（可选）写 `scene_semantic.ply`。
4. 对选中的相机：渲染逐类分数 → argmax 语义图（未覆盖区域置灰）、逐类 heatmap、scores `.npz`、`query_meta.json`。

### `semsplat/inference/ply_writer.py`
按点写语义 PLY：`x,y,z` + `red/green/blue`（按最优类着色）+ `semantic_id` + `confidence`。

### `semsplat/inference/eval_mioU.py`
**预留接口（未实现）**：把 Replica `mesh_semantic.ply` + `info_semantic.json` 渲染成每视图 GT 类别图，再与 `semsplat-query` 存下的 argmax 比对，算逐类 IoU / mIoU。接口 `compute_miou` 已可用，`evaluate_replica` 抛 `NotImplementedError`，待 M4 补。

---

## ⑦ 辅助脚本

### `scripts/make_mini_scene.py`
合成一个迷你 nerfstudio 数据集（随机轨道相机 + 渐变彩色图，无 COLMAP），用于 **M0 冒烟 / 语义阶段快速验证**：不改超参也能验证“前向-反向-优化器-致密化-语义损失是否跑通”。

### `tests/*`（⑧）
- `test_plugin_registration.py`（CPU）：两个方法名/优化器组/model `_target` 是否配齐、head 维度是否等于 512。
- `test_feature_rasterization.py`（GPU）：常量特征渲染 → `render/alpha≈常量`；无高斯覆盖处 pre-multiplied 应为 0。这两条锁死我们依赖的 ND 光栅语义。
- `test_param_densification.py`（GPU）：直接调 gsplat `ops.split/remove`，断言 `sem_features` 行数始终等于 `means`（即我们无需改 strategy 的依据）。

> 跑 pytest 需 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`（本机系统有 `launch_testing` 之类的 pytest 插件会与 pytest 版本冲突）。

---

## 关键设计取舍小结

| 取舍 | 为什么 |
| --- | --- |
| 特征 K=32 + 线性头到 M=512 | gsplat ND pass D>32 变慢；`pool→head == head→pool` 线性可交换，省算力 |
| 教师 = 纯 CLIP patch（不用 LSeg/SAM） | 用户选择：轻量、无额外权重；代价是稠密文本对齐弱，掩码精细度有限 |
| 语义损失在“教师网格”上算 | 损失降到 ~1/16 分辨率，省显存/时间；教师网格天然就是 patch 级 |
| 语义阶段默认 `detach` 几何/不透明度 | 语义损失只精修语义向量，不干扰 RGB/几何收敛（类似 nd_detach 思路） |
| 从 depth 反投影初始化（而非 random_init） | 无 SfM 点的大场景 random_init 收敛差；初始化点云让几何快速成形 |
| 锁定 nerfstudio 1.1.5 + 复制 `get_outputs` | 插件与宿主强耦合，锁版本后复制最稳、无漂移 |
| Replica 位姿保真（不做 auto 变换） | 需要度量一致的坐标；留 `--invert-pose` 开关供约定不同时使用 |

## 模块依赖关系（自上而下）

```
query_cli ─▶ eval_setup(nerfstudio) · scoring · teachers(text)
   ▲
   └─(config.yml)▶ pipeline ─▶ SemsplatModel
                                  ├─ gauss_params{means…, sem_features}
                                  ├─ feature_head : Linear(32→512)
                                  └─ get_outputs: RGB pass + ND 语义 pass
SemsplatModel ─▶ get_loss_dict ─▶ teachers/clip_local_teacher(视觉 patch) ◀─ GT 图
scoring ─▶ rasterize_class_scores（同一 ND 光栅）
replica_to_transforms / make_depth_init_ply ─▶ transforms.json(+points3d.ply) ─▶ nerfstudio-data
```
