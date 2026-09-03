# 教程 5：CLIP、patch token 与「特征蒸馏」的实现

> 对应文件：`semsplat/teachers/clip_local_teacher.py`、`semsplat/semsplat_model.py`（`_get_teacher` 与 `get_loss_dict` 的语义部分）。
> 读完你能回答：CLIP 的 512 维向量是从哪来的；怎么拿到"逐位置"的 CLIP 特征；为什么选 HF 实现；蒸馏损失到底在比什么。

## 0. 目标

训练时给每个高斯配了一个 32 维向量 `sem_features`。我们要让"**这块 3D 表面应该是什么语义**"被写进这个向量。办法：拿**冻结的 CLIP** 当老师——它给每张训练图一个"逐 patch 的语义特征图"，我们让"把高斯语义向量渲染出来的特征图"去逼近它。这就是**特征蒸馏（distillation）**：学生（可学习的高斯语义）向老师（冻结的预训练模型）对齐。

---

## 1. 知识一：CLIP 是什么结构

CLIP 用**双塔 + 对比学习**训练：
- 图像塔（ViT 视觉编码器）、文本塔（Transformer 文本编码器）；
- 训练目标是：配对的图文向量靠近、不配对的推远（InfoNCE 式对比损失）。

结果是一个**共享嵌入空间**：图像和文本都能编码成一个向量，向量点积 ≈ 语义相似度。CLIP 权重来自 OpenAI；`ViT-B/16` 图像塔把 224×224 切成 16×16 的 patch = 14×14=196 个 patch，文本塔输出的语义向量维度是 **512**。

## 2. 知识二：CLIP 的"整图向量" vs "patch token"——我们要哪个

- 视觉塔内部：一张图 → patch → Transformer 一堆 block → 一串 token（1 个 `[CLS]` + N 个 patch token）。
- 常规 `encode_image` 只取 **`[CLS]` 过一层 `visual_projection`** → 得到整图向量（对比学习就是用它）。它很准，但**没有空间位置**——你不知道"哪里像椅子"。
- 我们需要的**逐位置语义**在哪？在那些 **patch token**：每个 patch token 携带"这一小块 + 全局上下文"的信息，reshape 回 `[gH, gW]` 就有了空间位置。

> 难点：patch token **没有被显式训练成逐像素分割器**，语义对齐不如整图向量强。所以"纯 CLIP patch"方案的上限一般——这正是教程 1 说的"老师决定天花板"。可换成 LSeg/SAMCLIP 提升。

## 3. 知识三：为什么让 patch token 也"过 visual_projection"

`visual_projection` 是把 768 维 patch 内部表示压到 512 维文本空间的那层线性映射。整图向量能跟文本比，是因为过了它；**patch token 也过它**，才能大致落进同一个可比空间（查询时才敢拿文本向量去点积）。虽然 CLIP 没保证 patch 层语义强对齐，这是工程上最便宜的近似，也是 LERF 等做法的常见招式。

## 4. 知识四：为什么用 HF 的 transformers 实现（而不是裸 OpenCLIP）

要拿 patch token，需要绕开只返回 `[CLS]` 的 `encode_image`。两条路：
- OpenCLIP：得自己 hook 进 ViT 内部、处理位置编码插值，脆弱。
- **HuggingFace `transformers`**：`CLIPVisionModel(pixel_values, interpolate_pos_encoding=True)` 直接返回 `last_hidden_state`（含所有 patch token），对非 224 输入还能自动插值位置编码 → 本项目选用。

（代价：`transformers` 是个较重的依赖 + 首次下权重 ~350MB；都是可接受的。）

## 5. 本项目教师的实现（`clip_local_teacher.py`）

```python
class ClipLocalTeacher:
    def dense_features(self, image):        # image: [H,W,3] float 0..1
        # 1) 尺寸对齐：resize 到 16 的倍数 → grid = (gH,gW) = (H/16, W/16)
        # 2) CLIP 归一化（mean/std）+ → pixel_values [1,3,gH·16,gW·16]
        out = model.vision_model(pixel_values, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state[0]        # [1 + gH*gW, 768]
        patch  = tokens[1:].reshape(gH, gW, -1)  # 去掉 [CLS]，还原成网格
        feats  = model.visual_projection(patch)  # 768→512，逐 patch
        return L2_normalize(feats)               # [gH, gW, 512]   ← 教师特征
```
辅助 `compute(image, key)` 做 **LRU 缓存**：训练每帧只真正编码一次（`key` 用帧的 `cam_idx`），`fp16` 缓存省显存。

### 文本侧
```python
pooled = model.text_model(**tokens).pooler_output   # [B, 768]
emb    = model.text_projection(pooled)              # [B, 512]
emb    = L2_normalize(emb)
```
> 别用 `model.get_text_features(...)`——它在某些 transformers 版本返回的是 `BaseModelOutputWithPooling`，直接 `.float()` 会踩 `AttributeError`（本项目实际踩过，已改用上面的写法）。

## 6. 蒸馏损失长什么样（`semsplat_model.py::get_loss_dict`）

一次训练步（`step ≥ semantic_start_iter` 后）：

```
1. feat_map  = outputs["sem_feature_map"]      # 渲染出的 [H,W,K]（高斯语义，K=32）
2. pred      = feature_head(feat_map)          # K → M=512（共享解码头）
3. grid      = teacher.compute(gt_img, key)    # [gH,gW,M] 冻结 CLIP 特征
4. pred_p    = adaptive_avg_pool2d(pred, (gH,gW))   # 池化到教师网格
   alpha_p   = adaptive_avg_pool2d(alpha,(gH,gW))
   mask      = alpha_p > 0.5                   # 只比"有几何盖住"的 patch
5. loss = dist( L2norm(pred_p), L2norm(grid) )[mask].mean()   # cos/l1/mse
6. loss_dict["semantic_loss"] = loss × semantic_loss_mult
```

设计要点：

| 点 | 为什么 |
|---|---|
| 先 `feature_head` 后池化？ | 线性可交换：池化再升维 = 升维再池化，但**先池化省算力** |
| 损失算在"教师网格"(≈1/16 分辨率) | 省显存/时间；教师本身就只有网格分辨率 |
| `adaptive_avg_pool2d` 而非整除 | 训练早期有 1/4 分辨率（宽/高不是 16 倍数），自适应池化免去"必须被 16 整除"的约束 |
| `L2 归一` 后比 cos/l1 | 尺度无关，专比"方向=语义" |
| `mask = alpha>0.5` | 没几何的空 patch 不参与，避免把"空气"教成某个类别 |
| 几何/不透明度 detach | 语义损失只精修语义向量，不动 RGB/几何（教程 2 详述） |
| `semantic_start_iter` 门控 | 先让几何成型再教语义（本项目 3000 步起步） |

### 为什么把损失做成"往 CLIP 空间对齐"而不是"直接回归类别"

直接回归类别=封闭集（只能预测训练时的 20 类）。对齐到 CLIP 空间后，**查询阶段的任意文本**都能用余弦匹配到这个空间里的 3D 特征 → 开放词汇。

## 7. 本项目里 teacher 的接入（生命周期）

`_get_teacher()` 在第一次要算语义损失时才**惰性创建**，并且**故意不把 CLIP 注册成 model 的 submodule**：否则 CLIP 的几百 MB 权重会被当成模型参数存进 ckpt、也被 `.to()`/DDP 误处理。它只是"训练时借用的冻结工具"。

## 8. 常见误区

- **"CLIP patch token 就是好的逐像素分割器"**：不是，它们在对比损失下主要被训练为"整图判别的聚合输入"，局部语义对齐一般。
- **"特征 loss 会帮我重建几何"**：本项目默认 detach，语义不帮几何；两者各司其职。
- **"semantic_loss 一直 0 是模型没在学"**：先查几何/`alpha`（空 mask 会把损失置 0），教程 4 的 depth 初始化就是解法之一。
- **"维度越大语义越细"**：光栅化高维慢；瓶颈在教师质量，不在 32→64 这种维度差。

## 9. 延伸学习

- CLIP 论文：Radford et al., ICML 2021。
- Feature-3DGS 用 LSeg（CLIP 的逐像素版）做教师：看看"更强的教师"长什么样。
- HuggingFace `transformers` 的 CLIP 文档：`CLIPModel`、`CLIPVisionModel`。

**动手练习**：写个小脚本取一张图，对比 `encode_image`（整图向量）与 `dense_features`（patch 网格）对一个词（如 "chair"）的热度图，直观体会"patch token 空间上不够锐利"这一核心局限。
