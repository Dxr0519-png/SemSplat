# 教程 6：推理 / 开放词汇查询 与 评估

> 对应文件：`semsplat/inference/query_cli.py`、`scoring.py`、`ply_writer.py`、`eval_mioU.py`。
> 读完你能回答：训练完怎么"问"场景；逐高斯相似度怎么变成图像上的掩码；PLY 里存了啥；什么叫 mIoU，为什么它还没有实现。

## 0. 目标

训练产出一个 `.ckpt` + 一个 `config.yml`。查询脚本负责：加载模型 → 读入任意文本 → 对每个高斯算它"像不像这个词" → 把分数光栅化成**图上的热图/类别图**，同时导出一个**逐高斯着色的 3D 点云**。

---

## 1. 知识一：如何把"加载训练好的 nerfstudio 模型"变成几行代码

用 nerfstudio 现成工具 `eval_setup`：
```python
from nerfstudio.utils.eval_utils import eval_setup
config, pipeline, ckpt_path, step = eval_setup(Path(".../config.yml"))
model = pipeline.model.eval()
```
`config.yml` 是训练时自动存的整棵配置树（含 `output_dir`），`eval_setup` 会自动找最新 checkpoint 并加载。

两个易错点（本项目实际踩过）：
1. 返回顺序是 **(config, pipeline, ckpt, step)**，不是 pipeline 开头。
2. torch ≥ 2.6 的 `torch.load` 默认 `weights_only=True`，读不了含 numpy 标量的 nerfstudio 存档 → 查询前 monkeypatch：
```python
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})
```
（`query_cli.py` 已内置，自己写脚本记得复制这 2 行。）

## 2. 知识二：开放词汇打分 = 把 3D 和文本放进同一个空间比较

训练时（教程 5）我们逼"高斯语义向量过 `feature_head` 后 ≈ CLIP patch 特征"。查询就是它的逆用：

```
1. 逐高斯语义：   F = L2norm( feature_head(gauss_params["sem_features"]) )   # [N, 512]
2. 文本向量：     T = CLIP_text(每个 prompt)                                 # [C, 512]（已归一）
3. 分数矩阵：     S = F @ Tᵀ                                                # [N, C]
                 S[i, c] = 余弦(高斯 i 的语义, 第 c 个词)
4. 每个高斯取 argmax → 它最像哪个词；分数最高的值 → 置信度
```

这就是"开放词汇"的含义：prompt 列表只是**查询时**的输入，不是训练时的固定类别；换成没见过的词（"显示器"、"粉色花瓶"）也能比。

## 3. 知识三：把逐高斯分数画到图上 = 再光栅化一次

3D 高斯地图要想"可视化某张视角的语义"，就把 `S` 当作**另一种逐高斯颜色通道**再走一次 ND 光栅化（跟训练渲特征完全同一套）：

```python
maps, alpha = rasterization(..., colors=S, sh_degree=None, render_mode="RGB", backgrounds=None)
maps = maps / alpha.clamp_min(1e-6)        # [H, W, C] 每像素每类相似度
argmax = maps.argmax(axis=-1)              # 每像素最像的词
```
再按 prompt 调色板给 argmax 上色，就得到语义图；把每个类的分数单独画成热图，得到 heatmap。没被高斯盖住的像素（`alpha<=0.5`）标灰，表示"模型不知道这里有什么"。

## 4. 知识四：PLY：把"3D 语义"导成可查看的点云

语义图是**某几个视角**的可视化；3D 里每个高斯本身也有语义（第 2 步已算好每高斯的 argmax/置信度）。把**每个高斯的位置 + 按最优类着色的 RGB + 语义 id + 置信度**写成一个 `.ply`，就得到"整张语义地图"，可在 MeshLab/Blender/`open3d` 里旋转查看：
```
scene_semantic.ply
  vertex: x y z | red green blue(=palette[best]) | semantic_id(u2) | confidence(f4)
```
PLY 是简单文本/二进制点格式：头部声明属性，随后逐点数据（`plyfile` 库帮你写）。

## 5. 常用评估：mIoU（还没实现，讲清概念）

要对结果**定量**，需要"标注真值"。Replica 提供**场景级语义**：`mesh_semantic.ply`（每个面片带语义 id）+ `info_semantic.json`（id→类别名）。

一个常用指标 **mIoU（mean Intersection over Union）**：
```
对每个类 c：
  IoU_c = |预测=c 且 真值=c| / |预测=c 或 真值=c|     # 交集/并集
mIoU = 各类 IoU 的平均
```
做法：把 GT 语义网格按**评估相机位姿**渲染成每视图"真值类别图"，再和 `semsplat-query` 保存的 argmax 比对算 IoU。

**为什么还没实现**：需要写"网格语义 → 每视图 GT 图"的渲染器（用 open3d 离屏渲染，把类色映射回 id），加上 prompt 列表与 Replica 类别表的对齐（要逐词对应、处理背景/杂类）。这是可选的 M4。项目已把接口留好：
- `eval_mioU.compute_miou(pred, gt, n_classes)` 可用；
- `evaluate_replica(...)` 抛 `NotImplementedError` 占位；
- `semsplat-query` 存的 `*_scores.npz`（含 argmax）就是为它备好的输入。

## 6. query 全流程（对照 `query_cli.py`）

```
semsplat-query --config <config.yml> --prompts "wall,floor,chair,table,…"
               --camera-ids 0 5 10 --export-ply --out-dir results/<scene>
  ├─ monkeypatch torch.load(weights_only=False)
  ├─ eval_setup(config) → model.eval()
  ├─ encode_text_prompts(prompts) → T [C,512]
  ├─ 全场景每高斯 argmax/置信度；(--export-ply) 写 scene_semantic.ply
  └─ 对每个 camera id：
        maps, alpha = semantic_maps_from_prompts(...)   # 渲染 C 类分数
        写 cam_XXXX_semantic.png / *_heat_NN.png / *_scores.npz / query_meta.json
```

产物清单与字段：
| 文件 | 内容 |
|---|---|
| `cam_XXXX_semantic.png` | 每像素 argmax 类别着色；alpha≤0.5 标灰 |
| `cam_XXXX_heat_NN.png` | 第 NN 个 prompt 的相似度热图（相对最强非本类的余量） |
| `cam_XXXX_scores.npz` | `argmax` + 全类分数 `scores` + `prompts`（评估用） |
| `scene_semantic.ply` | 逐高斯语义点云（可选） |
| `query_meta.json` | prompts、相机 id、调色板 |

> `--camera-ids` 是 **eval 集**里的索引（`eval_setup` 已把数据按 eval 分好）。

## 7. 常见误区

- **"看数值低就以为代码错了"**：先区分"管线错"（掩码和几何完全不对齐）与"教师弱"（定位大概对但不锐利），再决定调什么。
- **"PLY 是照片"**：它是 3D 点云，要用 3D 查看器看，不是图片。
- **"mIoU 高=开放词汇好"**：它测的是"在这些预设类别上的分割"，只覆盖开放词汇能力的一部分。
- **"图上看不出 = 没救"**：可以先用教程 4 的跨帧重投影确认几何、再用单帧教师自比确认语义，逐层定位。

## 8. 延伸学习 / 练习

- 在 MeshLab 打开 `results/demo3/scene_semantic.ply`，旋转确认"椅子/墙"的空间一致性。
- 换个 prompt（如英文描述性句子 "a white table"）重跑，体会开放性；再换成一个训练里没有的词，看分布会不会崩（教师局限）。
- 实现 M4：写 `mesh_semantic.ply → 视图 GT 类别图`（open3d offscreen 渲染），接入 `eval_mioU`。

**动手练习**：跑 `semsplat-query --camera-ids 0 3 6`（不同视角同一场景），确认同一样东西在不同视角的掩码大致一致——这检验的是"语义真的长在 3D 高斯上，而不只是某张图上的 2D 伪影"。
