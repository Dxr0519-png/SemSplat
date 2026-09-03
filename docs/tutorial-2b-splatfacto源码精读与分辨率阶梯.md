# 教程 2b：Splatfacto 源码精读路线 × 训练"分辨率/精度阶梯"机制

> 配套源码（已装在本地）：`.venv/lib/python3.12/site-packages/nerfstudio/models/splatfacto.py`（v1.1.5，下文所有代码片段均摘自它）。
> 本文两件事合在一起：**(A)** 讲清 nerfstudio 训练时 `1/4 → 1/2 → 全分辨率` 的精度阶梯机制；**(B)** 按推荐顺序，一处一处带你精读对应源码，每处讲"这段要掌握什么"。
> 建议先读 [tutorial-2-nerfstudio与gsplat.md](tutorial-2-nerfstudio与gsplat.md)（架构总览）再看本文。

---

## Part A：先懂机制——为什么训练"从模糊到清晰"

### A.0 直觉

高斯**一开始是错的**（随机点/粗略点云）。如果一上来就在最高分辨率上优化：
- 代价高：像素越多、越慢；
- 没意义：结构还没对，去抠细节只会过拟合噪声。

所以 Splatfacto 像"图像金字塔"一样**从低分辨率爬到高分辨率**：先用低分辨率稳定整体结构，再把分辨率翻上去"加细节"。每个分辨率档停留若干步。

### A.1 两个旋钮决定"在哪一档"

配置里的两个字段（在 `SplatfactoModelConfig` 顶部）：

```python
num_downscales: int = 2          # 最粗从 1/2^2 = 1/4 开始
resolution_schedule: int = 3000  # 每 3000 步，降采样系数减半（即翻一倍分辨率）
```
官方注释原话：
> `resolution_schedule`: "training starts at 1/d resolution, every n steps this is doubled"
> `num_downscales`: "at the beginning, resolution is 1/2^d, where d is this number"

### A.2 精确的档位公式（这是"精度顺序"的答案）

运行时由 `_get_downscale_factor()` 计算当前降采样系数：

```python
def _get_downscale_factor(self):
    if self.training:
        return 2 ** max(
            (self.config.num_downscales - self.step // self.config.resolution_schedule),
            0,
        )
    else:            # 评估/查询永远用全分辨率
        return 1
```

以默认 `num_downscales=2, resolution_schedule=3000` 为例：

| 训练步范围 | `step // 3000` | 系数 | 实际渲染分辨率（对 1296×968） |
|---|---|---|---|
| `0 ≤ step < 3000` | 0 | `2^(2-0)=4` | **1/4 → 324×242**（最模糊档） |
| `3000 ≤ step < 6000` | 1 | `2^(2-1)=2` | **1/2 → 648×484** |
| `step ≥ 6000` | ≥2 | `2^(2-2)=1` | **全分辨率 → 1296×968** |

要点：
- 档位**只在训练时**生效；评估/推理 `_get_downscale_factor()` 返回 1（全分辨率渲染）。
- `max(...,0)` 保证到了全分辨率后不会"超采样"。

> **本项目为什么另设档位**：显卡只有 8GB，全分辨率会 OOM，所以我们训练时用
> `--pipeline.model.num-downscales 1 --pipeline.model.resolution-schedule 100000`
> → 系数恒为 `2^(1-0)=2`，**全程 1/2 分辨率**（代价是细节差，教程 usage 里有解释）。

### A.3 分辨率在"渲染 + 真值"两端同时降，必须保持一致

不是只把渲染降采样——**真值图也要降同一系数**再算损失，否则尺寸对不上、损失没法算。看代码里三处怎么配合：

**(1) 降采样用什么插值**——面积平均（抗锯齿、保平均亮度）：

```python
def resize_image(image: torch.Tensor, d: int):
    """Downscale images using the same 'area' method in opencv"""
    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return F.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d)...
```
（对 d×d 邻域做均值卷积 → 等价于 area/平均降采样。）

**(2) 真值图按当前档降**：

```python
def get_gt_img(self, image):
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    gt_img = self._downscale_if_required(image)   # 内部用 _get_downscale_factor()
    return gt_img.to(self.device)
```

**(3) 渲染端按当前档改相机内参后再光栅化**（在 `get_outputs` 里）：

```python
camera_scale_fac = self._get_downscale_factor()
camera.rescale_output_resolution(1 / camera_scale_fac)   # K、宽高同步缩小
viewmat = get_viewmat(optimized_camera_to_world)
K = camera.get_intrinsics_matrices().cuda()
W, H = int(camera.width.item()), int(camera.height.item())
self.last_size = (H, W)
camera.rescale_output_resolution(camera_scale_fac)       # 用完还原，别污染后续
```
> 学到一招：**降分辨率不是"改图片尺寸"这么简单，是要连内参 K 一起改**（像素少一半，fx/fy/cx/cy 也少一半）——这就是 `camera.rescale_output_resolution` 干的事。

### A.4 并行爬坡的另一个量：球谐颜色阶数（颜色"精度"也在爬）

颜色的"表达能力"也在随着步数逐步放开（高阶梯形求和的系数，越高阶能表示越精细的视角相关高光）：

```python
if self.config.sh_degree > 0:
    sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
else:
    ... sh_degree_to_use = None
```
`sh_degree_interval=1000` → 每 1000 步允许的球谐阶数 +1，最多到配置的 `sh_degree=3`。**分辨率阶梯管空间细节，球谐阶梯管颜色细节**，两者是配套的"由粗到细"。

---

## Part B：源码精读路线（跟着走一遍）

> 目的不是背代码，而是建立"**从配置到一次前向到损失到增删高斯**"的完整心智图。每步都到真实文件里亲眼看。

### 精读顺序总表（建议按序）

| 步 | 去 `splatfacto.py` 读什么 | 这段解决什么问题 | 要掌握的考点 |
|---|---|---|---|
| 1 | `SplatfactoModelConfig`（类最上面） | 有哪些"旋钮" | 知道每个超参是干什么的（特别是分辨率/球谐/致密化那组） |
| 2 | `__init__` + `populate_modules()` | 高斯参数从哪来 | `gauss_params` 这个 `ParameterDict`；尺度存 log、不透明度存 logit 的原因 |
| 3 | `_get_downscale_factor` + `resize_image` + `get_gt_img` | 分辨率阶梯在哪实现 | Part A 全部 |
| 4 | `get_outputs()` | 一次渲染全流程 | 坐标变换→光栅化→补背景；降采样连 K 一起改 |
| 5 | `get_loss_dict()` | 训练学的是什么 | L1+SSIM；mask 的处理 |
| 6 | `get_gaussian_param_groups()` / `load_state_dict()` | 高斯怎么进优化器/怎么随增删变化 | 为什么新参数要"进字典+同名优化器组" |
| 7 | 对照读我们的 `semsplat/semsplat_model.py` | 我们打了哪几个补丁 | `sem_features` 四联动 + 第二路 ND 光栅化 |

### 第 1 步：先读配置（5 分钟）

把 `SplatfactoModelConfig` 从头读一遍，**不背参数值，背"哪些组"**：
- **分辨率组**：`num_downscales`、`resolution_schedule`（Part A）。
- **球谐颜色组**：`sh_degree`、`sh_degree_interval`（A.4）。
- **致密化组**：`warmup_length`、`refine_every`、`cull_alpha_thresh`、`densify_grad_thresh`、`densify_size_thresh`、`stop_split_at`……（控制"何时分裂/剪枝"）。
- **初始化组**：`random_init`、`num_random`、`random_scale`。
- 其余（`background_color`、`rasterize_mode`、`camera_optimizer`…）先认识名字即可。

> 读到 `_target: Type = SplatfactoModel` 时记住：**配置是可实例化的，`_target` 指向真正要被 `new` 出来的类**——这是 nerfstudio 一切"配置驱动"的地基。

### 第 2 步：`populate_modules()`——高斯怎么"出生"（15 分钟）

读这段时抓住主线：创建 N 个高斯的初始参数 → **全塞进一个 `torch.nn.ParameterDict`**：

```python
self.gauss_params = torch.nn.ParameterDict({
    "means": means, "scales": scales, "quats": quats,
    "features_dc": features_dc, "features_rest": features_rest, "opacities": opacities,
})
```

考点（对着代码问自己）：
1. 尺度为什么 `scales = log(平均最近邻距离)`？→ 因为后面渲染要 `torch.exp(scales)`，**log 域优化**让尺度恒为正且数值稳。
2. 不透明度为什么 `logit(0.1)`？→ 后面渲染要 `sigmoid`，logit 域 = 无约束优化。
3. `features_dc/features_rest` 是什么？→ 球谐颜色系数（DC 一阶 + 其余高阶）。
4. 这一坨东西**全体是 `nn.Module` 的参数**，所以 `model.to(device)` 会一起搬 GPU。

> 对照点：我们加的 `sem_features` 也住在这里（多一行 key），这也是它能被 gsplat 自动"分裂时带上"的前提（见第 6 步）。

### 第 3 步：`_get_downscale_factor` / `resize_image` / `get_gt_img`（15 分钟）

把 Part A 的片段在文件里重新定位、连起来读一遍，直到你能画出一句话：
> 每次训练步：`get_gt_img(真值)` 和 `get_outputs(渲染)` 用的是**同一个** `_get_downscale_factor()`，所以"渲染在哪档、真值也在哪档，损失才算得起来"。

### 第 4 步：`get_outputs()`——一次渲染全流程（30 分钟，最重要）

按下面 5 段顺序精读：
1. **取优化后相机**：`optimized_camera_to_world = camera_optimizer.apply_to_camera(camera)`（默认 camera optimizer 关 = 原样）。
2. **坐标系**：`get_viewmat()` 把 c2w 转 gsplat 要的 w2c 视图矩阵（注意翻转 z/y 的手性处理）。
3. **降采样 + 内参**：`camera.rescale_output_resolution(1/d)` → K → 记住 `W,H` → 再还原（A.3）。
4. **光栅化**：`rasterization(means, quats, exp(scales), sigmoid(opacities), colors, viewmats, Ks, width, height, render_mode=..., sh_degree=...)`。
5. **补背景**：`rgb = render[..., :3] + (1-alpha)*background` —— 因为 gsplat 返回的 render 是 **pre-multiplied（已乘 alpha）**，空白处要拿背景色补。

> 读完这段你就能看懂我们 `semsplat/semsplat_model.py` 里为什么**逐字拷贝**它再加第二路特征光栅化：我们要用和 RGB **完全相同**的 `viewmat/K/W/H` 再渲一次语义向量，`super()` 拿不到这些局部变量，复制最稳。

### 第 5 步：`get_loss_dict()`（10 分钟）

考点：
- `Ll1 = |真值−渲染|.mean()`，`simloss = 1 − SSIM(...)`，`main_loss = (1−λ)Ll1 + λ·simloss`（λ=`ssim_lambda`）。
- 若有 `mask`，先对真值和渲染**同时**乘 mask 再算损失（把被掩掉区域抹黑，避免污染）。
- **没有任何"逐像素回归类别"**——几何就是这么纯粹靠 RGB 学的，语义是另一条损失（我们加的），见第 7 步。

### 第 6 步：`get_gaussian_param_groups()` / `load_state_dict()`（15 分钟）

**`get_gaussian_param_groups`** 是**写死的 6 个 key 列表**（不是遍历字典）——记住这个，就明白为什么新加参数必须重写它：

```python
return {name: [self.gauss_params[name]]
        for name in ["means","scales","quats","features_dc","features_rest","opacities"]}
```

**`load_state_dict`**：读 checkpoint 时先拿到新高斯数 `newp`，再把 `gauss_params` **里每个参数**都 resize 成 `[newp, …]` 再加载——所以任何塞进字典的参数（含我们加的）都自动享受"断点续训时行数对齐"。

> 去 `gsplat/strategy/ops.py` 看 `split/remove`：它们对**字典里每个 key** 施加同一个"按掩码 repeat/删行"函数。把这里和上面两段连起来，你就彻底懂了"为什么 `sem_features` 不用我们管也会跟着分裂"。

### 第 7 步：对照我们的补丁（20 分钟）

打开 `semsplat/semsplat_model.py`，逐个找：
1. `populate_modules()` 里新增的两行（`sem_features` + `feature_head`）；
2. 重写的 `get_gaussian_param_groups()` / `get_param_groups()`（把 6 个 key 变 8 个）；
3. `get_outputs()` 里 `# SEMSPLAT` 标记的那段：`rasterization(colors=sem_features, sh_degree=None, …)` 的 **`sh_degree=None`** 是"把 colors 当任意维特征而不是球谐"的开关；返回除 alpha 得 `sem_feature_map`；几何 `.detach()`；
4. `get_loss_dict()` 里的 `semantic_loss`（对照教程 5）。

每找到一处，就在官方版上打个标签问自己：**官方没有这段会发生什么？**——答案基本都写在教程 2/5 的"常见坑"里。

---

## 精度阶梯 × 本项目语义阶段的相互作用（读完后要想通的一点）

本项目语义损失默认 `semantic_start_iter=3000`。把 3000 和分辨率阶梯放一起看：

| 步数 | 分辨率档 | 语义损失 |
|---|---|---|
| 0–3000 | 1/4（`2^(2)`） | 关（几何先成型） |
| 3000–6000 | 1/2 | **开**；教师在"1/2 的真值图"上取 patch |
| ≥6000 | 全分辨率 | 开；教师在"全分辨率"上取 patch |

两个因此成立的工程细节：
1. **1/4 档宽/高常不是 16 的倍数**（如 1296/4=324，324/16=20.25）——所以我们用 `adaptive_avg_pool2d` 把学生特征对齐到教师 patch 网格，而不是要求整除（教程 5）。
2. 教师在"真值图"上编码，而真值图本身按当前档降采样 → 每个分辨率档网格粒度不同，缓存要按帧 key（`cam_idx`）而非"一次全图"长期混用。

---

## 精读自查清单（读完能答 = 达标）

- [ ] 能写出 `num_downscales=2, resolution_schedule=3000` 在 step=0/1500/3000/4500/9000 时的渲染分辨率系数。
- [ ] 能解释"降分辨率时为什么内参 K 也要改"。
- [ ] 能在文件里指认：真值降采样、渲染降采样、球谐阶数爬坡，各自在哪几个函数。
- [ ] 能说清 `gauss_params`、`get_gaussian_param_groups`、`ops.split/remove` 三者如何让"新参数自动被分裂"。
- [ ] 能解释 gsplat `render` 为什么是 pre-multiplied，以及 Splatfacto 怎么补背景。
- [ ] 能在 `semsplat_model.py` 指出 4 个补丁分别对应官方版的哪一段。
- [ ] 知道本项目为什么用 `num-downscales 1 + resolution-schedule 100000`（8GB 显存 → 恒 1/2 档）。

## 延伸

- gsplat `rendering.py` 里 `rasterization` 的完整签名（含 `sh_degree=None` 的 ND 通道分支）值得单独精读一遍。
- 想更彻底，读 `nerfstudio/engine/trainer.py` 的 `train_iteration`，看"分辨率调度/致密化回调"在训练循环里何时被触发。
