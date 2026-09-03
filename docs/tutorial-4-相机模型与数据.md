# 教程 4：相机模型、坐标系约定与 Replica 数据

> 对应文件：`semsplat/data/replica_to_transforms.py`、`scripts/make_depth_init_ply.py`、`scripts/prepare_replica.sh`。
> 读完你能回答：`fx` 是什么；c2w 和 w2c 差在哪；为什么"看起来一样的位姿"会导致重建失败；深度图存的到底是什么；为什么我们要用深度造初始点云。

## 0. 数据从哪来

训练 3DGS 需要的不是裸图，而是"**图 + 每张图相机的位置朝向 + 内参**"。最省事的数据 = **Replica**：一个合成室内数据集，自带**真值位姿**（不用跑 COLMAP 恢复位姿），还带深度。本项目用 NICE-SLAM 预处理的 Demo（单场景，`cvg-data.inf.ethz.ch/nice-slam/data/Demo.zip`）。

```
Demo/frames/
  color/<id>.jpg      # 1296×968 RGB
  depth/<id>.png      # 640×480, uint16, 单位=毫米
  pose/<id>.txt       # 16 个 float = 4×4 相机到世界矩阵
  intrinsic/intrinsic_color.txt | intrinsic_depth.txt
```

---

## 1. 知识一：针孔相机模型（内参）

针孔模型把 3D 点投到图像坐标：
```
u = fx·X/Z + cx        （fx 单位是"像素"：焦距占多少个像素宽）
v = fy·Y/Z + cy
```
内参矩阵 K：
```
     [fx  0  cx]
K =  [ 0  fy  cy]
     [ 0   0   1 ]
```
- `fx≈fy` 通常成立；`cx,cy` 大致在图像中心。
- **降采样时 K 必须跟着缩放**：图缩 1/2 → `fx,fy,cx,cy` 全乘 1/2（代码里 `transforms.json` 的这些字段若写错，相机就"错位"，几何就糊）。

## 2. 知识二：外参 / 世界↔相机

4×4 位姿矩阵有两种，互为逆：
```
c2w（camera-to-world，也叫 T_wc）：把"相机坐标系下的点"搬到"世界坐标"
w2c（world-to-camera，view matrix）：反过来，渲染器内部常用
```
Replica 给的是 **c2w**。它长这样：
```
c2w = [ R  t ; 0 1 ]，R 3×3 是相机朝向，t 3×1 是相机在世界里的位置
```
**同一张图，若把 c2w 当 w2c 用，或反了坐标轴方向，相机就会乱飞/镜像 → 几何永远学不拢。**（这是数据预处理第一坑。）

## 3. 知识三：相机坐标系约定（OpenCV vs OpenGL）——本项目 90% 的"姿势不对"都在这

同一个"相机朝向"，可以有两种坐标系习惯：
| | 右手系？ | y 轴 | 向前方向 | 谁在用 |
|---|---|---|---|---|
| OpenCV/COLMAP | 右手系 | y 朝**下** | +z 朝前 | 图像处理 |
| **OpenGL/nerfstudio** | 右手系 | y 朝**上** | −z 朝前 | 图形学、**3DGS 惯例** |

nerfstudio 生成 `transforms.json` 时用的是 **OpenGL 约定（相机看向 −z，y 朝上）**。所以：
- 一个像素 (u,v) 的**射线方向（在相机系）** 按 OpenGL 是：
```
dir_cam = normalize( (u−cx)/fx ,  −(v−cy)/fy ,  −1 )
```
  注意那个**负号**：y 上、z 前在 −z。
- 世界点 = `c2w · [dir_cam·d; 1]`（d 是深度距离）。

**反投影深度造点云时，用的必须是和渲染同一套约定**，否则点和相机对不上。

## 4. 知识四：nerfstudio 的 `transforms.json`（我们要生成的目标）

nerfstudio 的通用数据格式，一页 JSON：
```jsonc
{
  "camera_model": "OPENCV",     // 或 PINHOLE…；Replica 无畸变用 OPENCV 空畸变即可
  "fl_x": 1170, "fl_y": 1167, "cx": 646, "cy": 490, "w": 1296, "h": 968,
  "ply_file_path": "points3d.ply",          // 可选：给 splatfacto 的初始点云
  "frames": [
    { "file_path": "images/000000.jpg",
      "transform_matrix": [ /* 4×4 c2w, OpenGL 约定 */ ] }
  ]
}
```
nerfstudio 的 `NerfstudioDataParser(load_3D_points=True)` 读它；若给了 `ply_file_path`，会把点云解析成 metadata 里的 `points3D_xyz/rgb`，再被 pipeline 作为 `seed_points` 传给 Splatfacto（**不是随机初始化**，见 §7）。

### 4.1 要不要让 nerfstudio 自动"朝向/居中/缩放"？

`NerfstudioDataParserConfig` 有三个开关：
- `orientation_method`（默认 up：把世界扳成 y-up）、`center_method`（居中）、`auto_scale_poses`（缩到单位尺度）。
对 COLMAP 相机这些很有用。但 Replica 位姿**已是度量的、y-up 的**，自动变换反而破坏坐标 → 我们用了：
```python
NerfstudioDataParserConfig(orientation_method="none", center_method="none", auto_scale_poses=False)
```
这对应注册好的方法 `semsplat-replica`（通用版 `semsplat` 则保留默认，兼容任意数据）。

> 这也解释了为什么初始化随机尺度要大一点（`random_scale=5`）：场景保持了几米尺度，不是被归一化后的单位立方体。

## 5. 知识五：Replica 的深度图到底存的是什么

`depth/<id>.png`：
- dtype = **uint16**，单位**毫米** → 转米要 ×0.001（脚本里 `MM=0.001`）。
- 分辨率 640×480，**内参与 color 不同**（`intrinsic_depth.txt`），所以反投影用 depth 自己的 K，不要用 color 的 K。
- 值是"到相机沿射线的距离"（欧氏距离类），`0` 表示无数据（远处/空洞），要滤掉（`>0.05 且 <8m`）。

> 拿到一份 RGB-D 数据先做三件事：**看 dtype/单位、看分辨率/内参是否和 RGB 一致、看 0 值语义**。本项目踩过"颜色 K 和深度 K 混用"这类坑。

## 6. 知识六：如何数值验证"位姿自洽"（不用看图）

图像没法快速肉眼看时，可以用**跨帧深度重投影**验证约定是否自洽：
1. 帧 A：用 A 的内参 + A 的 c2w，把 A 的深度反投影成世界点；
2. 用 B 的 c2w（逆）把世界点投到 B 的相机、投到 B 的像素；
3. 比较"算出的深度"和 B 深度图上该像素的真实深度。

若位姿与约定自洽，误差应很小（本项目 ~2% 中位误差）。误差巨大 → 位姿被当 w2c、坐标轴方向错、或 K 不匹配。这比盲调网络参数高效得多，是排查"重建很糊"的第一步。

## 7. 知识七：为什么"从深度造初始点云"能救命

Splatfacto 两种初始化：
- **random_init**：在立方体里撒 `num_random` 个随机高斯。它从"不知道场景"开始长，密度/收敛都慢，**在无 SfM 点的大场景尤其差**（本项目体验：同样步数下 RGB 损失只到 ~0.30）。
- **给点云（seed points）**：先用现成三维信息把高斯"摆在场景该在的地方"，训练只负责精修 → 收敛又快又好（本项目：损失到 ~0.13、覆盖 α≈1）。

怎么造点云：**深度反投影 + 体素降采样**。
1. 每 N 帧抽一帧（本项目每 6 帧、300 帧里用 ~50 帧）；
2. 深度像素按 §3 的反投影公式 → 世界点；
3. 帧之间会重复 → **体素网格降采样**（把空间切成 2~4cm 小格，每格只留一个点），从可能上千万点降到 8~9 万个；
4. 写成 PLY 并把 `ply_file_path` 写进 `transforms.json`。

```python
# make_depth_init_ply.py 里的核心反投影（OpenGL 约定、深度视为沿射线距离）
ray = normalize( [(u−cx)/fx, −(v−cy)/fy, −1] )
cam = ray * depth_meters
world = R @ cam + t            # 用该帧 c2w
```
> 为什么点云只需"大致对"：Splatfacto 会继续 densify 精修。体素化丢细节、少量错点是可容忍的噪声。

## 8. 本项目脚本串起来

```
scripts/prepare_replica.sh          # 下载 + 解压原始 Demo 帧
semsplat/data/replica_to_transforms.py
   ├─ discover_color/poses：自动探测 color/pose 目录（跳过 depth）
   ├─ 解析 intrinsic_color.txt(K) → fx fy cx cy；读图得 w h
   └─ 位姿原样写入 transform_matrix（OpenGL 约定），可选 --subset/--invert-pose
scripts/make_depth_init_ply.py      # depth 反投影 → points3d.ply + 改 transforms.json
→ data/replica_demo_ns/  (transforms.json + images/ 符号链接 + points3d.ply)
```
`--probe` 模式只探测不改写，用于"换了个数据集先看布局再配参数"。

## 9. 常见误区

- **"内参不会变"**：图像缩放、裁剪都会改 K；数据从 1296×968 到 648×484 时 K 必须 ×1/2。
- **"把 w2c 当 c2w 也能练出来"**：全局一致的翻转或许能练出"镜像重建"，但混用/逐帧不一致必然崩。先做跨帧重投影验证。
- **"深度直接就是 Z 坐标"**：很多深度图是"沿射线距离"，与"相机平面距离"不同；反投影前先确认，别直接当 Z 用。
- **"random init 训练久点就行"**：对小物体可以，大室内场景最好给 seed points（SfM 或深度）。

## 10. 延伸学习

- nerfstudio 数据格式文档：`docs.nerf.studio/reference/data_formats`（transforms.json 字段）
- 相机模型细节：Hartley & Zisserman《Multiple View Geometry》第 6 章。
- Replica 数据集：Facebookresearch / Replica（`github.com/facebookresearch/Replica-Dataset`）；本机 `data/replica_demo` 可自行翻看真实文件。

**动手练习**：改 `make_depth_init_ply.py` 的 `_matrix_c2w_gl` 返回 `np.linalg.inv(p)`（当 w2c 用），重跑训练看损失是否变差——用实验把"约定错误"的后果记住。
