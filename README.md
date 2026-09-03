# semsplat — open-vocabulary semantic 3DGS on nerfstudio / gsplat

A nerfstudio plugin that extends **Splatfacto** (gsplat backend) with a distilled,
per-Gaussian **open-vocabulary semantic field** (Feature-3DGS-style), trained
offline on posed image sequences (Replica processed RGB-D) and queried with
free-form text.

- **Method name:** `semsplat` (general) / `semsplat-replica` (Replica preset)
- **Query CLI:** `semsplat-query`
- **Backend:** nerfstudio 1.1.5, gsplat 1.4.0 (source-built for Blackwell sm_120)

## 文档

**概览 / 手册**
- [docs/architecture.md](docs/architecture.md) —— 模块详解（按工程创建流程逐个文件讲清）
- [docs/usage.md](docs/usage.md) —— 使用手册（环境/数据/训练/查询/测试命令）

**分模块教学（每块独立成文：知识讲解 + 本项目实现）**
| 模块 | 文档 | 讲什么 |
|---|---|---|
| 背景 | [tutorial-1-3dgs与开放词汇语义.md](docs/tutorial-1-3dgs与开放词汇语义.md) | 3DGS 原理、三种语义 3DGS 路线、开放词汇概念 |
| 框架 | [tutorial-2-nerfstudio与gsplat.md](docs/tutorial-2-nerfstudio与gsplat.md) | nerfstudio 插件架构、Splatfacto/gsplat 源码剖析 |
| 框架(深读) | [tutorial-2b-splatfacto源码精读与分辨率阶梯.md](docs/tutorial-2b-splatfacto源码精读与分辨率阶梯.md) | 分辨率精度阶梯(1/4→1/2→全)机制 + 按 7 步带你精读 Splatfacto 源码 |
| 环境 | [tutorial-3-CUDA-Blackwell编译.md](docs/tutorial-3-CUDA-Blackwell编译.md) | sm_120、CUDA 版本、为何源码编译 gsplat |
| 数据 | [tutorial-4-相机模型与数据.md](docs/tutorial-4-相机模型与数据.md) | 针孔相机/坐标系约定/transforms.json/Replica/深度初始化 |
| 语义 | [tutorial-5-CLIP与特征蒸馏.md](docs/tutorial-5-CLIP与特征蒸馏.md) | CLIP patch token、教师对齐与蒸馏损失 |
| 推理 | [tutorial-6-推理查询与评估.md](docs/tutorial-6-推理查询与评估.md) | 余弦打分、分数光栅化、PLY、mIoU |
| 工程 | [tutorial-7-测试与工程化.md](docs/tutorial-7-测试与工程化.md) | 测试守护的约定、扩展与 SLAM 方向 |

## Architecture

Each Gaussian gains a low-dim semantic vector `sem_features` (K=32) stored in
`SplatfactoModel.gauss_params` (so gsplat's densifier keeps it row-aligned).
Training renders it with a second `gsplat.rasterization` pass
(`sh_degree=None` → N-D feature blending, alpha-normalized), decodes K→M=512 via a
shared `feature_head` linear layer, and matches the result (area-pooled to the
patch grid) against a **frozen OpenAI-CLIP ViT-B/16 patch-token teacher** of the
same training image (cosine / L1). Geometry stays untouched by the semantic loss
(`semantic_detach_*`, default True). Open-vocabulary text query compares each
Gaussian's decoded vector with CLIP text embeddings and rasterizes per-class
score maps / masks.

```
input images+poses ──(splatfacto RGB)──> Gaussians
                                        + gauss_params["sem_features"] [N,32]
train loss = L_rgb + γ·dist( feature_head( render(features) ),
                             CLIP_patch_teacher(gt_img) )     (after semantic_start_iter)
query     : text → CLIP text emb → cos( feature_head(f_g) , t ) → argmax mask / heatmaps / PLY
```

## Environment (RTX 50-series / sm_120)

Prebuilt gsplat wheels have no sm_120 kernels, so **gsplat must be source-built**
with CUDA ≥ 12.8. Order is load-bearing:

```bash
scripts/build_env.sh        # venv + torch 2.7.0 cu128 (native sm_120)
scripts/build_gsplat.sh     # pip nvcc>=12.8 + gsplat==1.4.0 --no-binary (sm_120)
scripts/install_nerfstudio.sh  # nerfstudio 1.1.5 + transformers + `pip install -e .`
scripts/smoke_plugin.sh     # capability + raster + registration + 30-iter train
```

Outbound network on this machine goes through a local proxy
(`http://127.0.0.1:7897`); set `PROXY` if yours differs.

## Data

Replica scenes (preprocessed RGB-D, ground-truth poses) are used, converted to a
nerfstudio `transforms.json` dataset:

```bash
scripts/prepare_replica.sh --out data/replica_demo          # NICE-SLAM Demo.zip (~157 MB)
python semsplat/data/replica_to_transforms.py --scene data/replica_demo/<scene> \
       --out data/replica_demo_ns --probe                    # inspect layout first
# then, once the pose/intrinsic flags match the layout:
python semsplat/data/replica_to_transforms.py --scene data/replica_demo/<scene> \
       --out data/replica_demo_ns
```

Full multi-room data: `https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip`
(~12 GB, all rooms).

## Train

```bash
ns-train semsplat-replica --data data/replica_demo_ns \
    --max-num-iterations 20000 --output-dir outputs/demo
```

Key knobs (model config): `semantic_feature_dim`, `semantic_start_iter`,
`semantic_loss_type` (cos/l1/mse), `semantic_loss_mult`, `teacher_model_name`.
Watch `semantic_loss` drop on Tensorboard after `semantic_start_iter`.

## Query

```bash
semsplat-query --config outputs/demo/<run>/config.yml \
    --prompts "sofa,chair,coffee table,rug,tv monitor,door,wall" \
    --camera-ids 0 50 100 --export-ply --out-dir results/demo
```

Produces per-camera argmax semantic maps, per-prompt heatmaps, score `.npz`, and
an optional colored `scene_semantic.ply`. Per-class mIoU vs Replica GT is an
optional later milestone (interface reserved in `semsplat/inference/eval_mioU.py`).

## Tests

```bash
make test    # pytest tests/ (registration on CPU; raster/densification need GPU)
```
