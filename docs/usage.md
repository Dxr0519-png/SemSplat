# 使用手册（命令）

> 面向「在这台已配好的机器 / 新机器上跑起来」的实操命令。模块是什么见 [architecture.md](architecture.md)。

## 0. 常用速查

```bash
# 环境激活
source .venv/bin/activate          # venv 在 repo 根 .venv/

# 冒烟（sm_120 内核 + 注册 + 迷你训练）
bash scripts/smoke_plugin.sh

# 训练（Replica Demo 场景）
.venv/bin/ns-train semsplat-replica \
    --data data/replica_demo_ns \
    --max-num-iterations 10000 \
    --pipeline.model.num-downscales 1 \
    --pipeline.model.resolution-schedule 100000 \
    --vis tensorboard \
    --output-dir outputs/demo

# 查询
.venv/bin/semsplat-query \
    --config outputs/demo/<场景>/semsplat-replica/<时间戳>/config.yml \
    --prompts "wall,floor,ceiling,sofa,chair,table,lamp,window" \
    --camera-ids 0 5 10 --export-ply --out-dir results/demo

# 测试（需禁用本机冲突的 pytest 插件）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q
```

本机出网走代理时，任何**需要联网的 python/curl 命令**前加：
```bash
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
```
（HF 下载 CLIP 权重、下载数据集都需要。）

---

## 1. 目录约定

| 路径 | 内容 |
|---|---|
| `.venv/` | Python 环境（venv，勿提交） |
| `data/replica_demo/` | NICE-SLAM Demo 原始帧（color/depth/pose/intrinsic） |
| `data/replica_demo_ns/` | 转换后的 nerfstudio 数据集（`transforms.json` + `images/` + `points3d.ply`） |
| `outputs/<...>/<方法>/<时间戳>/` | 每次训练的 config.yml、`nerfstudio_models/step-*.ckpt`、tensorboard events |
| `results/<场景>/` | `semsplat-query` 产物 |
| `semsplat/` | 插件源码 | 

---

## 2. 环境搭建（新机器才需要）

要求：Linux + NVIDIA 驱动 ≥570（CUDA 13.2 亦可，向下兼容）+ **GPU 支持 sm_120**（RTX 50 系）。

```bash
# 1) venv + torch 2.7.0 cu128（原生 sm_120）
bash scripts/build_env.sh

# 2) 源码编译 gsplat==1.4.0（sm_120）
#    需要 CUDA ≥12.8 工具链。本机已装于 ~/cuda-12.8（runfile 免 sudo）。
#    换机器装法：
#      curl -fL -o /tmp/cuda.run https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
#      bash /tmp/cuda.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8
bash scripts/build_gsplat.sh

# 3) nerfstudio 1.1.5 + transformers + 本插件
bash scripts/install_nerfstudio.sh

# 4) 冒烟验收
bash scripts/smoke_plugin.sh
```

> 等价 `make install`（依次跑 1→3）。

---

## 3. 数据准备（Replica）

### 3.1 下载原始序列
```bash
# Demo.zip = 单场景 ~157MB（开发最快）
bash scripts/prepare_replica.sh \
    --url https://cvg-data.inf.ethz.ch/nice-slam/data/Demo.zip \
    --out data/replica_demo

# 或全量 8 个房间（~12GB）
# bash scripts/prepare_replica.sh --url https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip --out data/replica --scene office0
```

### 3.2 转成 nerfstudio 数据集
```bash
# 先探测布局（Demo 布局会自动识别；其他布局用 --color-dir/--pose-dir/--intrinsics-file 指定）
python semsplat/data/replica_to_transforms.py --scene data/replica_demo --probe

# 转换（取前 300 帧）
python semsplat/data/replica_to_transforms.py \
    --scene data/replica_demo \
    --out data/replica_demo_ns \
    --intrinsics-file frames/intrinsic/intrinsic_color.txt \
    --subset 300
```
产物：`data/replica_demo_ns/transforms.json`（c2w 原样写入，度量坐标）+ `images/`。

### 3.3 用 depth 生成初始化点云（强烈推荐）
无 SfM 点、纯随机初始化在大场景收敛差。从深度反投影 8~9 万点做 seed：
```bash
python scripts/make_depth_init_ply.py \
    --scene data/replica_demo --ns data/replica_demo_ns --n-frames 300 --voxel 0.04
```
会写 `data/replica_demo_ns/points3d.ply` 并往 `transforms.json` 加 `ply_file_path`。

---

## 4. 训练

### 4.1 一句话
```bash
.venv/bin/ns-train semsplat-replica --data <数据集> --output-dir outputs/<名> [参数]
```

### 4.2 常用参数（cli 覆盖 `--pipeline.model.xxx` / `--pipeline.datamanager.xxx`）
| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-num-iterations` | 30000 | 总迭代；本项目示范 10000 |
| `--pipeline.model.semantic-start-iter` | 3000 | 从第几步开始语义蒸馏（更早=特征更早生效） |
| `--pipeline.model.semantic-feature-dim` | 32 | 逐高斯语义维度 K（≤32 更快） |
| `--pipeline.model.semantic-loss-type` | cos | `cos/l1/mse` |
| `--pipeline.model.semantic-loss-mult` | 1.0 | 语义损失权重 |
| `--pipeline.model.num-downscales` | 2 | 训练起始降采样级数 |
| `--pipeline.model.resolution-schedule` | 3000 | 分辨率切换步长（设很大=全程固定低档，8GB 显存推荐 `1` + `100000`） |
| `--vis` | viewer | `viewer` / `tensorboard` |
| `--output-dir` | outputs | 运行输出根 |
| `--viewer.quit-on-train-completion True` | False | 训练完自动退出（脚本用） |

### 4.3 8GB 显存的推荐档
```bash
.venv/bin/ns-train semsplat-replica \
  --data data/replica_demo_ns \
  --max-num-iterations 10000 \
  --pipeline.model.num-downscales 1 \
  --pipeline.model.resolution-schedule 100000 \
  --vis tensorboard \
  --output-dir outputs/demo
```
训练产物：
- `outputs/demo/<scene>/semsplat-replica/<时间戳>/config.yml` —— **查询时要传给 semsplat-query**
- `.../nerfstudio_models/step-000009999.ckpt`
- tensorboard events（`tensorboard --logdir outputs/demo` 可看 `main_loss / semantic_loss / psnr`）。

### 4.4 可视化 viewer（可选）
```bash
.venv/bin/ns-train semsplat-replica --data data/replica_demo_ns ... --vis viewer
# 浏览器打开 http://localhost:7007
```

---

## 5. 开放词汇查询

```bash
.venv/bin/semsplat-query \
  --config outputs/demo/<scene>/semsplat-replica/<时间戳>/config.yml \
  --prompts "wall,floor,ceiling,sofa,chair,table,lamp,window" \
  --camera-ids 0 5 10 \
  --export-ply \
  --out-dir results/demo
```

| 参数 | 说明 |
|---|---|
| `--config` | 训练输出的 config.yml（必填；自动加载最新 ckpt） |
| `--prompts` | 逗号分隔文本 |
| `--camera-ids` | 用 eval 集里的相机索引（省略=全部） |
| `--export-ply` | 额外导出逐点语义 PLY |
| `--out-dir` | 产物目录 |

产物：
- `cam_XXXX_semantic.png` —— 每像素 argmax 类别着色图
- `cam_XXXX_heat_NN.png` —— 第 NN 个 prompt 的热图
- `cam_XXXX_scores.npz` —— argmax + 全部分数（留作 mIoU）
- `scene_semantic.ply` —— 每个高斯按最优类着色（可在 MeshLab/Blender 打开）
- `query_meta.json` —— prompt 列表与调色板

> 图片与 PLY 请用本机查看器直接打开确认（CLI 不会校验“好不好看”）。

---

## 6. 测试

```bash
# 8 个用例：方法注册(CPU) + 特征光栅(GPU) + 致密化联动(GPU)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q
```
GPU 用例会自动 `skip`（无 cuda 时）。

---

## 7. 常见问题排查

| 现象 | 原因/处理 |
|---|---|
| `no kernel image is available...` | gsplat/torch 不是 sm_120 内核 → 重跑 `build_gsplat.sh`，确认 torch 是 cu128 |
| `glm/glm.hpp not found` | 别用 PyPI sdist 装 gsplat，走 `build_gsplat.sh` 的 git clone |
| `ModuleNotFoundError: No module named 'torch'`（装 gsplat 时） | 需 `--no-build-isolation`（脚本已带） |
| `torch.load ... Weights only load failed` | torch≥2.6 默认 `weights_only=True` → `semsplat-query` 已内置补丁；自己写脚本也照做 |
| `eval_setup` 解包错 | 返回是 `(config, pipeline, ckpt, step)`，不是 `(pipeline, …)` |
| `semantic_loss` 恒为 0 | 几何未成形 / `alpha>0.5` patch 为空：用 depth 初始化点云 + 多训几何（见 3.3） |
| 重建 PSNR 低(~16) | 固定 1/2 分辨率所致；想更好需更充分训练或更高分辨率（注意 8GB 显存） |
| `launch_testing` pytest 插件报错 | 用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` |
| 联网超时/下载失败 | 命令前 `export https_proxy=http://127.0.0.1:7897 http_proxy=...` |
| Replica 位姿看起来镜像/错 | 转换加 `--invert-pose` 重建再验（本项目位姿已验证自洽，误差~2%） |

---

## 8. 常见开发循环（改代码 → 快速验证）

```bash
# 1) 极小场景、极短训练验证“跑不跑得通”（几秒）
python scripts/make_mini_scene.py --out /tmp/mini_scene
.venv/bin/ns-train semsplat --data /tmp/mini_scene \
    --max-num-iterations 120 --vis tensorboard \
    --pipeline.model.semantic-start-iter 30 --pipeline.model.semantic-feature-dim 8 \
    --output-dir /tmp/mini_out

# 2) 看语义损失是否出现并下降
tensorboard --logdir /tmp/mini_out

# 3) 好了再上真实数据（第 4 节命令）
```

## 9. 可选/后续（M4）
- 定量 mIoU：`semsplat/inference/eval_mioU.py` 预留；需把 Replica `mesh_semantic.ply`+`info_semantic.json` 在 eval 位姿渲染成 GT 类别图后接入 `compute_miou`。
- 更强的开放词汇教师（SAM+CLIP masked pooling / LSeg）替换 `teacher_model_name`，相应把 `semantic_head_dim` 改成该空间维度。
