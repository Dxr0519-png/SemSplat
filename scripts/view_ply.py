#!/usr/bin/env python
"""极简 3D 查看器：用 open3d 打开一个 PLY（高斯/点云，带 RGB 更好）。

用法：
    python scripts/view_ply.py results/demo3/scene_semantic.ply
    python scripts/view_ply.py data/replica_demo_ns/points3d.ply --point-size 2
无显示环境时加 --headless 可离屏截图。
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", help=".ply 路径")
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--headless", action="store_true", help="离屏：只截图不弹窗")
    ap.add_argument("--shot", default="/tmp/ply_view.png")
    args = ap.parse_args()

    import open3d as o3d

    pcd = o3d.io.read_point_cloud(args.ply)
    print(f"points: {len(pcd.points)}  has color: {pcd.has_colors()}")

    # 若没颜色就按高度染一个伪彩色，方便看结构
    if not pcd.has_colors():
        pts = np.asarray(pcd.points)
        z = pts[:, 2]
        z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        col = np.stack([z, 0.3 + 0.4 * (1 - z), 1 - z], axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(col)

    if args.headless:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.capture_screen_image(args.shot, do_render=True)
        vis.destroy_window()
        print("saved", args.shot)
        return

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="semsplat ply", width=1280, height=800)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([0.1, 0.1, 0.12])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
