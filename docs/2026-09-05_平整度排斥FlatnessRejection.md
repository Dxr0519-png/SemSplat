# 2026-09-05 实现总结：平整度排斥（Flatness Rejection）——"墙面碎斑"专项

> 项目：`semsplat`（nerfstudio 1.1.5 插件，gsplat 1.4.0；Feature-3DGS 式逐高斯开放词汇语义 3DGS）。
> 背景：`2026-09-05_空间正则化与法向几何先验.md` 把 mIoU 抬到 0.240（+正则）/ 0.255（+查询期法向先验）。剩一类伪影没解决：**墙面上极度自信的 tv/door/lamp/clock 误检**（"碎斑"）。TV/熵平滑拉不回错误标签 → 本轮用**训练期几何**做硬否决。
> 环境：RTX 5060 Laptop 8GB (sm_120)、torch 2.7.0+cu128、Python 3.12 `.venv`。评估口径同前：office0_est320、50 帧 `0,6,…,294`、13 prompts、`eval_miou_room.py`。

**目录**
- 0 结论速览
- 1 机制（按原方案字面实现）
- 2 实证前提与一个崩溃 bug
- 3 office0 20k 验证与量化
- 4 判读：为何没能"彻底消灭"
- 5 产物索引
- 6 关键命令速抄

---

## 0 结论速览

1. **机制按原方案字面实现并验证成立**：几何平整度（per-grid-cell 平面拟合残差 ≤1cm）× 教师文本顶格 ∈ {tv/door/lamp/clock} → 该蒸馏单元 Semantic Loss ×0；门控比例 `sem_flat_gated_frac` 逐帧 0.2%–32%（墙面多的帧 gate 多）。6 个默认 OFF config 字段 + 惰性文本嵌入 + RGB+ED 复用深度，无新增参数。
2. **office0 20k mIoU：0.240（+正则 reg）→ 0.242（再 +平整度排斥）——净 +0.002，约等于没动**。受击物体小幅升（lamp +.039 / tv +.016 / wall +.014 / door +.009），小幅降（chair −.041 / table −.026 / rug −.024）。
3. **担心的最坏情况没发生**：真 flush 物体（door/tv/clock/lamp 与墙共面）**没有塌缩**，door/tv recall ≥0.98 基本保住。
4. **但"彻底消灭墙面碎斑"没达成**：墙面 tv/door 误检像素只 −7~12%（door 1.318M→1.160M，tv 789k→732k）。门控清除的墙面误检 9.2%，同时**新引入 6.8% 反方向误检**，互相抵消。
5. **根因（数据驱动）**：office0 的"碎斑"是**连贯大块**不是盐椒孤岛（前置分析：<20px 孤岛仅占 FP 的 2–4%），且错误标签在 gate 开启前的 iters 200–1000 已烤进特征；×0 只停未来的错监督，TV 只能从大块**边界**向内侵蚀（最大 >50k px 大块被削掉 46% 质量、10→5 块，但碎成中等块）。大块内部无干净墙可扩散 → 真空区原地保留。**结论：×0 门对连贯大块是"削皮"不是"铲除"。**

---

## 1 机制（字面实现）

**平整度定义**（纯 GPU、与教师网格同池化对齐，`semsplat/losses.py`）：
- 每个教师网格单元内对 feature-pass 期望深度做最小二乘平面拟合 `d(u,v)=a·u+b·v+c`（像素坐标，无需 K/位姿）。把 u,v,d 及其乘积的矩量图用 `adaptive_avg_pool2d` 池到 `[gH,gW]` → 闭式解 3×3 线性系统 → 每单元拟合残差 RMS。
- `flat(cell) = 覆盖≥50% ∧ 残差² ≤ sem_flat_plane_resid_m²(0.01m)`。深度 `detach`，纯几何信号。

**文本门**（与查询端同文本塔，懒加载缓存）：教师目标向量 L2 归一后对两段小词表 argmax —— stuff（`plain room wall,floor,flat white ceiling`）∪ reject（`television screen,door,table lamp,wall clock`）。`teacher_reject = 顶格 ∈ reject`。

**Loss 门**：`gate = flat ∧ teacher_reject ∧ 现语义 alpha 掩码`；`sem_loss = loss[mask ∧ ¬gate].mean()`。门控单元 Semantic Loss ×0 → 真空区，由既有 Depth-aware TV（跨深度边不抹）扩散填补。

**config（默认全 OFF，旧 config/ckpt 原样可载）**：`sem_flat_enabled / sem_flat_start_iter=1000 / sem_flat_plane_resid_m=0.01 / sem_flat_min_covered_frac=0.5 / sem_flat_stuff_prompts_csv / sem_flat_reject_prompts_csv / sem_flat_sim_min=0.0`。
reject 清单**限定为"贴附竖直平面小物体"**——绝不扩到 table/rug/desk/chair/sofa（桌面/坐垫也平整，含了会坍缩）。

---

## 2 实证前提与一个崩溃 bug

**崩溃（首轮 20k，step ~1020 即 gate 首开附近）**：
```
torch.linalg.solve: (Batch element 0): The solver failed because the input matrix is singular.
```
根因：求解对**所有单元**跑（含未覆盖空单元）。office0 全帧角落有房间外的空区 → α=0 → u,v 矩量全 0 → 3×3 法方程**精确奇异**，批量 `torch.linalg.solve` 对任一奇异元素直接 raise。12 帧冒烟集恰好每格都有覆盖，故冒烟没炸。
**修复**：用平移不变的 (u,v) 2×2 协方差奇异值判 `spread`（覆盖像素须在 u、v 两向都有延展，~1-D 细线/空区判非 spread），对非 spread 单元加恒等正则后照解、结果按 `flat &= spread` 丢弃。回归测试 `test_flatness_singular_uncovered_cell_does_not_raise`。重启后 0 报错跑满 20k。

**数据验证（决定 reject 清单）**：真 door/clock/lamp/tv 与墙共面（GT 深度边仅 1.3–5%、平整度 ~100%）→ 若 gate 按"真物体必有几何突变"的直觉做会**误伤真物体**；用户已拍板"直跑原方案先量化"，接受真 flush 物体可能被误伤的风险（后验：没误伤，§3 recall 证明）。

---

## 3 office0 20k 验证与量化

**配方** = `+正则 reg` 配方 + 开 gate（run：`outputs/office0_320_sam20k_flat/office0_est320_ns/semsplat-replica/2026-09-05_105902/`，20000 iter / num-downscales 0 / semantic-start 200 / teacher samclip amg 8pp ctx / head_dim 512 / sem_tv_mult=1.0 / opacity_entropy_mult=0.05 / sem_flat_enabled / sem_flat_start_iter=1000）。整跑 1635s，0 报错；semantic_loss settle 0.0247 ≈ reg 0.0245。

**同口径 50 帧 mIoU**：

| 类别 | +正则 reg | +平整度 flat | Δ |
|---|---|---|---|
| sofa with cushions | .422 | .451 | +.029 |
| table lamp | .178 | .217 | +.039 |
| television screen | .179 | .195 | +.016 |
| plain room wall | .435 | .449 | +.014 |
| door | .128 | .137 | +.009 |
| floor | .057 | .069 | +.012 |
| flat white ceiling | .005 | .005 | 0 |
| black office chair | .865 | .824 | −.041 |
| table | .648 | .622 | −.026 |
| rug | .200 | .176 | −.024 |
| potted green plant / wall clock / desk organizer | 0 | 0 | 0 |
| **mIoU (13 类均在 GT)** | **0.240** | **0.242** | **+.002** |

**墙面误检 vs 真物体 recall**（50 帧聚合；reject 类落在 GT stuff 平面上 = 误检）：

| 类 | FP-on-stuff reg | FP-on-stuff flat | Δ | recall reg | recall flat |
|---|---|---|---|---|---|
| door | 1,318,135 | 1,159,894 | **−12.0%** | 0.993 | 0.987 |
| tv | 788,627 | 732,441 | **−7.1%** | 0.986 | 0.979 |
| lamp | 213 | 619 | +190%（绝对极小） | 0.337 | 0.293 |
| clock | 0 | 0 | 0 | 0 | 0 |

**大块结构**（door+tv 误检在 GT stuff 上的连通块，按大小聚合）：>50k px 大块 reg 10 块 / 618k px → flat 5 块 / 335k px（**−46% 质量**，被削/碎成中等块）；10k–50k 块 reg 724k → flat 790k（中等块反增）。净效果：清除 9.2% lit stuff 像素的 tv/door，新引入 6.8%。

---

## 4 判读：为何没能"彻底消灭"

1. **连贯大块（非碎斑）**：office0 墙面 tv/door 误检是**连贯区域**（<20px 孤岛仅 2–4%）。×0 制造真空后靠 TV 从"干净邻居"扩散填补；若整面墙区域都是同一错误标签（大块），块内没有干净墙像素可扩散，只沿**边界**被正确墙侵蚀一圈 → 总量只 −10% 左右，大块削成中块。与前置分析的风险 2 一致。
2. **先烤进去的错误**：gate 1000 轮才开，错误标签已在 iters 200–1000（600 轮）写入特征。×0 是"停新错"，不是"擦旧错"。
3. **反方向新错抵消**：6.8% 的 stuff 像素从墙/地/顶被 flat 判成 tv/door（门控单元 TV 扩散的副作用 / 判别边界移动），把清除的 9.2% 抵消到净 ~2.5%。
4. **mIoU 层面接近洗牌**：wall 增的 .014 ≈ chair/table/rug 掉的总和。正则抛光已把能拿的分数拿得差不多，gate 动的是判别边界而非类支持。
5. **收获**：① 真 flush 物体没塌缩（误伤恐惧解除）——recall door/tv ≥0.98；② 机制全链路（几何平整度 × 文本顶格 × loss×0）在训练中可观测地工作，`sem_flat_gated_frac` 稳定入 TB；③ 对比图可见最大几块 tv/door 墙斑确实被削小。

**方向建议（供后续定夺）**：若继续追"墙斑铲除"，×0 真空 + TV 被动扩散对连贯大块无效；更可能有效的是 **① 门开早**（semantic-start 同刻开，错误不先烤进去）、**② 不是 ×0 而是把 gated 单元的目标直接替换成 stuff 文本嵌入**（主动写墙，不留真空）、或 **③ 复用查询期法向先验**（0.255，已是三者最优）。本轮 plain-×0 单独不构成收益，不建议保留为默认。

---

## 5 产物索引

| 类别 | 路径 |
|---|---|
| 平整度/文本门 | `semsplat/losses.py`（`flatness_cell_mask` / `flat_reject_mask`，含 spread 正则防奇异） |
| 模型接线 | `semsplat/semsplat_model.py`（config 字段默认 OFF / `_flat_text_embs` 惰性嵌入 / `_flat_active` / `need_depth` 扩为 RGB+ED / `get_loss_dict` gate） |
| 单测 | `tests/test_flat_rejection.py`（含奇异空单元回归） |
| 训练 run | ~~`outputs/office0_320_sam20k_flat/office0_est320_ns/semsplat-replica/2026-09-05_105902/`~~ 已随"取消改动"删除（本实现代码已回退，ckpt 不可再用） |
| mIoU 文本 | `results/orb_ate/flat_cmp/flat_miou.txt`(0.242) vs `results/orb_ate/reg_cmp/new_miou.txt`(0.240) |
| 逐帧 npz | `results/orb_ate/flat_cmp/flat_npz/` |
| 对比 HTML | `results/orb_ate/flat_cmp/gallery/index.html`（6 帧 18/24/186/192/234/240 × 5 列：原图/重建/GT/之前+正则/当前+平整度） |

---

## 6 关键命令速抄

```bash
# 训练（reg 配方 + 开 gate）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/ns-train semsplat-replica \
  --data data/office0_est320_ns --max-num-iterations 20000 \
  --output-dir outputs/office0_320_sam20k_flat \
  --pipeline.model.num-downscales 0 --pipeline.model.semantic-start-iter 200 \
  --pipeline.model.semantic-head-dim 512 --pipeline.model.teacher-kind samclip \
  --pipeline.model.teacher-sam-mode amg --pipeline.model.teacher-sam-points-per-side 8 \
  --pipeline.model.teacher-sam-crop-mode ctx \
  --pipeline.model.sem-tv-mult 1.0 --pipeline.model.sem-tv-depth-sigma 0.1 \
  --pipeline.model.opacity-entropy-mult 0.05 --pipeline.model.opacity-entropy-start-iter 1000 \
  --pipeline.model.sem-flat-enabled True --pipeline.model.sem-flat-start-iter 1000 \
  --viewer.quit-on-train-completion True --vis viewer+tensorboard

# 评估（与 reg 同口径）
.venv/bin/python scripts/orb_slam/eval_miou_room.py \
  --config outputs/office0_320_sam20k_flat/<run>/config.yml --ns data/office0_est320_ns \
  --gt-dir data/office0_est320/frames/semantic \
  --prompts "sofa with cushions,black office chair,table,potted green plant,plain room wall,floor,flat white ceiling,table lamp,door,television screen,rug,wall clock,desk organizer" \
  --gt-ids 76 20 80 44 93 40 31 47 37 87 98 22 35 \
  --frames $(seq -s ' ' 0 6 294) \
  --out results/orb_ate/flat_cmp/flat_miou.txt --dump-scores results/orb_ate/flat_cmp/flat_npz

# 图册（列标题用 --*-label/--*-tag 参数化）
.venv/bin/python scripts/orb_slam/gallery_prev_new.py \
  --config outputs/office0_320_sam20k_flat/<run>/config.yml --ns data/office0_est320_ns \
  --gt-dir data/office0_est320/frames/semantic \
  --prev-npz results/orb_ate/reg_cmp/new_npz --new-npz results/orb_ate/flat_cmp/flat_npz \
  --miou-prev results/orb_ate/reg_cmp/new_miou.txt --miou-new results/orb_ate/flat_cmp/flat_miou.txt \
  --prev-label "之前语义预测 · +空间正则 (reg)" --new-label "当前语义预测 · +平整度排斥 (flat)" \
  --frames 18 24 186 192 234 240 --out results/orb_ate/flat_cmp/gallery

# 单测
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_regularizers.py tests/test_flat_rejection.py -q
```
