# 教程 2：nerfstudio 插件架构 与 Splatfacto / gsplat 源码剖析

> 对应文件：`semsplat/semsplat_config.py`、`semsplat/semsplat_model.py`，依赖 nerfstudio 1.1.5 的 `nerfstudio/models/splatfacto.py` 与 gsplat 1.4.0 的 `gsplat/rendering.py`。
> 读完你能回答：nerfstudio 怎么发现一个"新方法"；Splatfacto 的高斯参数存在哪、谁负责分裂；gsplat 一次光栅化到底传了什么；我们如何在**不改 CUDA** 的情况下渲染 32 维语义图。

## 0. 本模块在做什么

`SemsplatModel` = 给 nerfstudio 的 Splatfacto 模型"打了三个补丁"：
1. 高斯字典里多一个语义参数；
2. 渲染时除了 RGB 再走一路 ND 特征光栅化；
3. 损失里加一条"和 CLIP 教师对齐"的项。

补丁能打成功，前提是彻底看懂宿主 nerfstudio/gsplat 的扩展点。

---

## 1. nerfstudio 怎么认识你的方法

### 1.1 全流程是"配置驱动"

nerfstudio 里"方法 = 一棵配置树"。核心对象：
- **`TrainerConfig`**：顶层，含 `method_name`、训练多少步、优化器字典 `optimizers`、pipeline、viewer 等。
- **Pipeline（VanillaPipeline）**：串起 datamanager + model，负责 `get_train_loss_dict(step)`。
- **Datamanager（FullImageDatamanager）**：取一帧（全图 + 相机）→ 返回 `(camera, batch)`。对高斯模型一次就是**一整张图**（不是逐光线采样）。
- **Model**：`get_outputs(camera)` 渲染 → `get_loss_dict(outputs, batch)` 算损失。**这是我们要改的地方。**

### 1.2 第三方方法怎么被 `ns-train` 发现

nerfstudio 用 Python 的**入口点（entry points）** 做插件发现：

```
pyproject.toml
  [project.entry-points."nerfstudio.method_configs"]
  semsplat = "semsplat.semsplat_config:semsplat_method"
```

`ns-train` 启动时读这个入口点组，把每个对象当成一个 `MethodSpecification`（包着一个 `TrainerConfig`），再按 `config.method_name` 注册成子命令。所以只要 `pip install -e .` 装过，`ns-train semsplat` 就出现。

> 模型配置本身也是"可实例化配置"：`SplatfactoModelConfig` 是 dataclass，带 `_target: Type` 指向真正的类。nerfstudio 用 `_target` 把配置"实例化"成对象，这也让我们可以写 `class SemsplatSplatfactoModelConfig(SplatfactoModelConfig)` 直接继承整棵配置、只追加新字段。

### 1.3 优化器字典 = 与 `get_param_groups()` 一一对应

`TrainerConfig.optimizers` 是 `{名字: {optimizer, scheduler}}`。名字必须**精确匹配**模型 `get_param_groups()` 返回的 key，否则训练器建优化器时会找不到/多出参数。这是新手最容易踩的坑（我们为此加了单测）。

### 1.4 一次训练步的时序（重要）

```
trainer 每步：
  step_cb(step)          # 把 step 写进 model.step（Splatfacto 用 BEFORE_TRAIN_ITERATION 回调）
  pipeline.get_train_loss_dict(step)
     → datamanager.next_train() → (camera, batch)   # batch["image"]=整张GT图
     → model.get_outputs(camera)                    # 渲染，产出 outputs
     → model.get_loss_dict(outputs, batch)          # 算损失
  反向传播
  model.step_post_backward()   # Splatfacto 里调用 strategy 做致密化（分裂/剪枝）
```

**这解释了本项目两个设计**：
- 我们的语义损失在 `get_loss_dict` 里需要知道"当前是哪一帧"来命中教师缓存，但 `batch` 里**没有** `image_idx` → 我们在 `get_outputs`（它收得到 `camera`）里把 `camera.metadata["cam_idx"]` 存成 `self._last_cam_idx` 供损失用。
- 语义阶段门控 `self.step >= semantic_start_iter` 依赖 `self.step` 已由 `step_cb` 写好。

---

## 2. Splatfacto 源码解剖（v1.1.5）

### 2.1 高斯参数住在哪：一个 `ParameterDict`

所有可优化高斯参数打包在一个 **`torch.nn.ParameterDict`**：
```python
self.gauss_params = torch.nn.ParameterDict({
  "means": means, "scales": scales, "quats": quats,
  "features_dc": features_dc, "features_rest": features_rest, "opacities": opacities,
})
```
`features_dc`/`features_rest` 是球谐颜色系数（`sh_degree=3` → 1+15=16 组 3 维）。颜色属性/`num_points` 等都是 `@property` 转发到这个字典。

### 2.2 谁负责"分裂/复制/删除"：gsplat DefaultStrategy

Splatfacto 创建 `DefaultStrategy(...)`（把 `cull_alpha_thresh`、`densify_grad_thresh` 等配置传进去），然后：
- 渲染后 `strategy.step_pre_backward(...)`（更新屏幕空间梯度统计等）
- 反向传播后 `strategy.step_post_backward(params=self.gauss_params, optimizers=..., ...)`

策略内部调 gsplat 的 `ops.split / duplicate / remove`。**关键机制**：这些 ops 对传入的 params **字典里每一个 key 套同一个 `param_fn`**：
```
对 "means"：分裂出的新位置 = 旧位置 + 扰动
对其它 key（含我们加的 "sem_features"）：直接按掩码 repeat 旧值
```
且每次都要 `optimizers[name]` 存在，否则 KeyError。

> **推论（本项目能不改 strategy 的依据）**：只要把 `sem_features` 放进 `gauss_params` **并且**在 `TrainerConfig.optimizers` 里补同名组，gsplat 会自动让语义向量的行数与高斯数保持同步。我们单独写了个 GPU 单测（`tests/test_param_densification.py`）直接调 `ops.split/remove` 锁死这一点。

### 2.3 一次光栅化调用（RGB）

`get_outputs` 里的核心调用（省略部分参数）：
```python
render, alpha, info = rasterization(
    means=..., quats=..., scales=torch.exp(scales),     # 尺度要取 exp（存的是 log）
    opacities=torch.sigmoid(opacities).squeeze(-1),     # 不透明度要过 sigmoid
    colors=colors_crop,                                  # [N, 1+15, 3] 球谐系数
    viewmats=viewmat, Ks=K, width=W, height=H,
    render_mode=..., sh_degree=min(step//1000, 3), ...)
```
几个要点：
- 高斯存 `scales=log(尺度)`、`opacities=logit(α)`，是为了优化时数值稳定（无约束），**用前再反变换**。
- `render_mode="RGB"`（训练时）/ `"RGB+ED"`（评估时，多一条期望深度通道）。
- gsplat 返回的 `render` 是 **pre-multiplied（已乘 alpha）**，Splatfacto 自己补背景：`rgb = render + (1-alpha)*background`。
- `sh_degree=None` 是开关之一——**这正是我们渲染"普通张量特征"的入口**（见 §3）。

### 2.4 它把"渲染+致密化"的元信息存在 `self.info`

`rasterization` 返回第三个值 `info`（means2d、radii、conics、索引等），`strategy` 靠它判断哪些高斯要分裂/剪枝。所以**我们追加的第二路光栅化绝不能覆盖 `self.info`**（本项目中我们只把结果放局部变量，不写回）。

### 2.5 `get_gaussian_param_groups` 是硬编码列表（不是遍历字典）

```python
def get_gaussian_param_groups(self):
    return {name: [self.gauss_params[name]]
            for name in ["means","scales","quats","features_dc","features_rest","opacities"]}
```
→ 父类**不会**自动带上我们加的 key。**必须重写**，把 `sem_features` 加进去（教程里最容易漏的一步）。

### 2.6 `load_state_dict` 会自动把每个参数 resize 到新的高斯数

父类 `load_state_dict` 会先读 `gauss_params.means` 的行数 `newp`，然后把**字典里每个参数**都 resize 到 `[newp, ...]` 再加载 → 我们加的 `sem_features` 也免费享受这个行为（前提：它已在 `gauss_params` 里）。

---

## 3. 知识：gsplat 的 ND（高维属性）光栅化

gsplat 的 `rasterization` 对 `colors` 有两种解释：
- `sh_degree` 给数 → `colors` 被当作球谐系数 `[N,K,3]`，内部展开成 RGB；
- **`sh_degree=None` → `colors` 被当作"逐高斯任意维特征" `[N,D]` 或 `[C,N,D]`**，直接对 D 个通道做同一套 α 混合（ND pass）。文档提示 **D>32 会变慢**。

返回：`render=[C,H,W,D]`（pre-multiplied）。于是：
```
期望特征 = render / alpha        （仿 "ED / 期望深度" 的归一）
```
不覆盖的像素 alpha≈0，应被 mask 掉（本项目用 `alpha>0.5` 判定有效 patch）。

**为什么我们不用改 CUDA 就能渲语义**：把 32 维语义向量当作"另一种颜色通道"光栅化即可，同一套可微 kernel 顺带把梯度传回每个高斯的语义向量。这就是 Feature-3DGS 那套"并行 ND 光栅化"在 gsplat 里的免费等价物。

---

## 4. 本项目三个补丁（对应实现）

| 补丁 | 位置 | 做了什么 |
|---|---|---|
| 语义参数 + 解码头 | `populate_modules()` | `gauss_params["sem_features"]`（`[N,K]` 小随机）与 `self.feature_head = nn.Linear(K,M)`；`feature_head` 不放进 `gauss_params`（是共享头，不该被当逐高斯参数） |
| 参数组重写 | `get_gaussian_param_groups()` / `get_param_groups()` | 把 `sem_features`、`semantic_head` 两个 key 补进优化器分组 |
| 第二路光栅化 | `get_outputs()` | 复制 v1.1.5 `get_outputs` 主体（锁版本后无漂移），在 RGB pass 后加：`rasterization(colors=sem_features, sh_degree=None, render_mode="RGB", backgrounds=None)` → `sem_feature_map = render/alpha`，几何/不透明度默认 `.detach()` |
| 语义损失 | `get_loss_dict()` | 见教程 5（教师对齐） |

`get_outputs` 用"复制整个函数体"而不是"调 `super()` 再补一段"，原因：需要在**同一个** `viewmat/K/分辨率/降采样系数` 下渲染，而父类把这些都算在函数局部；复制最稳。nerfstudio 已锁死 1.1.5，所以复制不随版本漂移（源文件版本写在模块 docstring）。

### 4.1 配置字段速查（`SemsplatSplatfactoModelConfig`）

```python
semantic_feature_dim=32        # K：逐高斯语义向量维度
semantic_head_dim=512          # M：教师/解码空间维度
semantic_start_iter=3000       # 从第几步开语义损失（几何先成型）
semantic_loss_type="cos"       # cos | l1 | mse（都先在 L2 归一上算）
semantic_loss_mult=1.0
semantic_detach_geometry=True  # 语义损失不回传几何梯度
semantic_detach_opacity=True
teacher_model_name="openai/clip-vit-base-patch16"
teacher_cache_fp16=True / teacher_cache_max_entries=2000
```

---

## 5. 常见坑（都是我们实际踩过的）

| 坑 | 为什么 | 解法 |
|---|---|---|
| `sem_features` 不在优化器里/KeyError | 优化器字典必须与 `get_param_groups()` 同名对齐 | `_make_trainer()` 里补 `sem_features`、`semantic_head` 两组；测试锁死 |
| 训练中途高斯数变了但语义向量行数没变 | gsplat ops 只处理 `gauss_params` 字典里的 key | 放字典里 + 见 `test_param_densification` |
| `semantic_loss` 恒 0 | `alpha>0.5` 的有效 patch 为空（几何没成型） | 用 depth 初始化点云 + 多训几何（见教程 4） |
| 特征 pass 覆盖了致密化统计 | 第二路光栅化若写回 `self.info` 会破坏 strategy | 用局部变量接收 meta |
| 大图显存爆 | ND pass + RGB pass + 教师 | 锁 1/2 分辨率训练；损失在低分辨率教师网格算 |

## 6. 延伸学习

- nerfstudio 官方扩展文档：https://docs.nerf.studio/developer_guides/new_methods.html
- 本地可读的权威源码：`.venv/lib/python3.12/site-packages/nerfstudio/models/splatfacto.py`、`.../gsplat/rendering.py`、`.../gsplat/strategy/ops.py`
- gsplat 论文/文档：https://docs.gsplat.studio/

**动手练习**：在 `.venv` 的 `splatfacto.py` 里搜 `get_gaussian_param_groups` 与 `rasterization(`，对照本文逐行读一遍；再把 `semantic_feature_dim` 改成 64 跑冒烟，体会 D>32 的性能变化。
