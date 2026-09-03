# 教程 7：测试与工程化——怎么让"跑得通"可持续

> 对应文件：`tests/*`、`Makefile`、`pyproject.toml`、`scripts/*`。
> 读完你能回答：这 8 个测试各自在守护什么；为什么"改一行就要想测试"；怎么把项目继续往 SLAM 方向扩展。

## 0. 为什么要有这些测试

本项目最贵的东西是**"环境能跑"**（sm_120 编译、若干 API 细节）。一旦环境好了，真正容易再次打破的是几个**隐含约定**。测试就是在显式地记录这些约定，让人/模型改动时能立刻发现。

---

## 1. 知识：测试金字塔与本项目的位置

工程测试分三层：
```
端到端（慢、贵）    —— 真的 `ns-train` 训一小段 → 看 ckpt
集成（中）          —— 真的 import + 组真部件（但不上全量训练）
单元（快、便宜）    —— 单函数/单机制
```
我们对「环境重」的模块用**单元/机制测试**（不依赖 nerfstudio 全家桶，只 import gsplat/torch），对注册这类"接缝"用集成测；端到端由 `scripts/smoke_plugin.sh` 承担（人工触发，不进 pytest）。

## 2. 逐个测试：它在守护什么约定

### `test_plugin_registration.py`（CPU）
守护"配置树配齐"：
- 两个方法（`semsplat` / `semsplat-replica`）名字对、`_target` 指向 `SemsplatModel`；
- 优化器字典至少含 **8 组**：`means…opacities` + `sem_features` + `semantic_head`（缺了训练器会找不到参数/KeyError，见教程 2）；
- `semantic_head_dim==512` 且教师名是 `openai/clip-vit-base-patch16`（两者必须匹配，否则损失维度错）。

### `test_feature_rasterization.py`（GPU）
守护"gsplat ND 光栅语义没变"——整个方法依赖它：
- **常量特征测试**：所有高斯同一个特征 c，渲染出 `render/alpha ≈ c`（覆盖处）。验证"pre-multiplied 除以 alpha=期望特征"这条假设。
- **零背景测试**：没盖住的像素 render≈0（不会把特征泄漏到空背景）。
这两个测试是我们放心在 `get_outputs` 里写 `sem_feature_map = render/alpha` 的依据。

### `test_param_densification.py`（GPU）
守护"致密化行数联动"：
直接调 gsplat `ops.split/remove`，断言分裂/删除后 `sem_features` 行数**始终等于** `means`。
这验证教程 2 的结论——"放进 `gauss_params` + 同名优化器组，gsplat 自动带它分裂"。若哪天 gsplat 升级改变了 param_fn 遍历策略，这里第一个红。

### 如何跑（环境提醒）
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q
```
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`：本机有其他 pytest 插件（如 ROS 的 `launch_testing`）自动加载会与当前 pytest 冲突，禁用自动加载即可（不装不删它们）。GPU 用例在无 CUDA 机器上会自动 skip。

---

## 3. 工程化：`uv`、`Makefile`、`.venv`

### 为什么用 `uv`
系统 python3.12 **缺 ensurepip**，`python3 -m venv` 直接失败（要装 `python3.12-venv`，需要 sudo）。`uv`（在 `~/.local/bin/uv`）能免 sudo 建 venv + 装包，还带全局 wheel 缓存（重建环境快）。
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python <包>
```

### `Makefile` 的作用
把易忘的、顺序敏感的步骤固化成目标：
```
make env          # build_env.sh
make gsplat       # build_gsplat.sh
make install      # 上面两个 + install_nerfstudio.sh（顺序有讲究）
make smoke        # smoke_plugin.sh
make mini-train / demo-train / data-demo / test / query / clean
```
`PROXY` 变量（默认 `http://127.0.0.1:7897`）会在目标里 export，因为本机出网要代理。

### 版本/环境心智模型
- **宿主锁定**：`nerfstudio==1.1.5`、`gsplat==1.4.0` 精确固定——因为我们是"复制 `get_outputs`" + "依赖 strategy 遍历行为"，升级宿主会破坏这些约定（升级=走一遍冒烟+测试）。
- **环境不是一次性的**：`~/cuda-12.8`（编译 gsplat 用）、`~/.cache/uv`、`~/.cache/huggingface`（CLIP 权重）都要保留；重装环境要复用它们。

---

## 4. 如何扩展这个项目（接口在哪）

| 想改什么 | 改哪里 | 需要注意 |
|---|---|---|
| 换更强的教师 | `teacher_model_name`；新增类替换 `semsplat/teachers/clip_local_teacher.py` | 教师输出维度 M 必须等于 `semantic_head_dim`；仍要能在同一个文本/图像共享空间里查询（或同步换文本编码器） |
| 换损失类型 | `semantic_loss_type=cos/l1/mse`；或自定义在 `get_loss_dict` 里加一项 | 若想同时修几何，把 `semantic_detach_geometry=False` |
| 想让语义损失更早生效 | `semantic-start-iter` | 太早=几何未成型时教语义，效果通常更差 |
| 加新高斯属性（比如法向、更多特征） | 仿照 `sem_features`：塞进 `gauss_params` + `get_gaussian_param_groups` + 优化器组 + `get_outputs` 新 pass | 别忘 4 处联动 + 对应单测 |
| 跑更大场景/更多数据 | `scripts/prepare_replica.sh --url Replica.zip --scene xxx`；`replica_to_transforms --subset` 调帧数 | 8GB 显存：锁 1/2 分辨率、限制点数 |
| 定量评估（M4） | 实现 `eval_mioU.evaluate_replica`：GT 语义网格 → 视图类别图 → 比对 `*_scores.npz` | 先做 prompt↔类别表对齐；见教程 6 |
| 往 GS-SLAM 方向集成 | 离线版已产：语义高斯参数（可直接当语义地图）+ 查询 | 在线版要处理：逐帧增量 densify 时语义向量随之更新（本项目已证明 gsplat 会自动带 `sem_features` 分裂——这正是"在线语义建图"最关键的一块已被验证的部分） |

### 快速开发循环
```bash
# 最小验证（十几秒）：迷你场景 + 早期语义门控
python scripts/make_mini_scene.py --out /tmp/mini_scene
.venv/bin/ns-train semsplat --data /tmp/mini_scene --max-num-iterations 120 \
    --vis tensorboard --pipeline.model.semantic-start-iter 30 \
    --pipeline.model.semantic-feature-dim 8 --output-dir /tmp/mini_out
tensorboard --logdir /tmp/mini_out     # 看 semantic_loss 是否出现并下降
# 再上真实数据
```

---

## 5. 一份"改动后该跑什么"清单

| 改动类型 | 必跑 |
|---|---|
| 只改配置/数据 | 训练一小段看 `semantic_loss` + 查询一次 |
| 改 `semsplat_model.py` | `pytest tests/`（尤其 raster + densification）→ smoke |
| 改教师/损失/维度 | 注册测试 + 迷你端到端（确认 `semantic_loss` 非零且下降） |
| 升级 nerfstudio/gsplat | 全部单测 + `smoke_plugin.sh`（可能推翻"复制 get_outputs"的假设） |
| 换机器/重装环境 | `make install` + `smoke_plugin.sh` |

## 6. 常见误区

- **"单测过了=能出好结果"**：单测只保证"约定没破、管线能跑"；结果质量（语义准不准）要靠真实数据训练 + 查看器/评估。
- **"把全量训练放进 pytest"**：太慢太贵，属于 `scripts/smoke_plugin.sh` 这种手动/CI 定时任务。
- **"测试只是为了验收"**：更多是**文档 + 回归护栏**——尤其把"gsplat 会自动带新参数分裂"这种非显式机制固化成断言。

## 7. 延伸学习

- pytest 快速上手：`pytest --fixtures`；`skipif` 用于"无 GPU 环境跳过"。
- 预提交 / CI 建议：`.github/workflows` 里放 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_plugin_registration.py`（CPU 可跑）+ 一台带 sm_120 GPU 的 runner 跑 GPU 用例与 smoke。

**动手练习**：故意把 `semsplat_config.py` 里 `semantic_head` 这组优化器删掉，跑 `test_plugin_registration` ——看到它红，就理解这个测试在护什么。
