# scripts/orb_slam — ORB-SLAM3 无真值建图桥接（Replica 验证用）

把 "RGB+D、无真值位姿" 的序列变成 semsplat 可训的 `transforms.json`(+points3d.ply)，
用 ORB-SLAM3 估计每帧位姿（metric、RGB-D）。Replica Demo 的 GT 只用于**事后**量化（ATE）。

## 流程

```bash
# 1) 喂料：Replica 布局 → ORB 序列（注册 depth 到 color 分辨率）
.venv/bin/python scripts/orb_slam/replica_prep_orb.py --scene data/replica_demo --out data/orb_replica

# 2) 跑 ORB-SLAM3（需已编译 ~/github_project/ORB_SLAM3；DISPLAY 出 GUI）
scripts/orb_slam/run_orb.sh            # 轨迹拷回 data/orb_replica/CameraTrajectory.txt

# 3) TUM 轨迹 → “估计位姿 NICE-SLAM scene”（相机系 diag(1,-1,-1) 换算）
.venv/bin/python scripts/orb_slam/traj_to_scene.py --traj data/orb_replica/CameraTrajectory.txt \
    --scene data/replica_demo --out data/replica_orb_est

# 4) ATE（est vs GT，Umeyama，纯 numpy）
.venv/bin/python scripts/orb_slam/eval_ate.py --traj data/orb_replica/CameraTrajectory.txt \
    --gt-scene data/replica_demo --out-dir results/orb_ate

# 5) 估计位姿 → ns 数据集（复用现成转换器，300 帧）
.venv/bin/python semsplat/data/replica_to_transforms.py --scene data/replica_orb_est \
    --out data/replica_orb_est_ns --intrinsics-file frames/intrinsic/intrinsic_color.txt --subset 300
.venv/bin/python scripts/make_depth_init_ply.py --scene data/replica_orb_est --ns data/replica_orb_est_ns --n-frames 300

# 6) A/B 短训（GT 与 EST 各一次）→ ns-eval（需 wrapper 补 torch weights_only）→ 汇总
scripts/orb_slam/ab_train.sh
.venv/bin/python scripts/orb_slam/ns_eval_wrapper.py --load-config <cfg.yml> --output-path <m.json>
.venv/bin/python scripts/orb_slam/ab_summary.py --gt m_gt.json --est m_est.json --out ab_summary.txt
```

## 参考数字（Replica Demo office0, 500 帧, 2026-09）

- ORB-SLAM3 全 500/500 帧跟踪；轨迹长 8.04 m；**ATE RMSE 4.1 cm（0.51%）**。
- 短训 6000 iter 半分辨率 ns-eval（同 30 帧 eval 集）：est-pose **PSNR 20.14 / SSIM 0.852**，
  与"正确朝向约定的 GT 基线"(GT pose 套 `R·diag(1,-1,-1)` 后重训) **19.98 / SSIM 0.838** 基本持平
  （+0.16 dB / +0.014 SSIM）。即无真值建图 ≈ 有真值（同约定）。
- ⚠️ 发现：**NICE-SLAM Demo 原始 GT pose 的朝向约定与本仓/nerfstudio 期望相反**——直接喂原始 GT
  只到 ~14.7 dB、初始化点数 419k，套 `R·diag(1,-1,-1)` 后 250k 点、PSNR→19.98。
  历史 GT 基线(demo3 ~16)因此被低估；做 GT vs est 对比时必须用同一约定（flip 后）的 GT。

## 坑速查
见 memory；重点：ns-train `--vis viewer` 训完卡住需手动 kill；ns-eval 用 wrapper；
CLIP 教师需 HF_HUB_OFFLINE=1；`Camera.fps` 整数。
