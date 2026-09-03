"""Shared helpers for the ORB-SLAM3 -> semsplat pose bridge.

Convention summary
------------------
- ORB-SLAM3 camera frame (OpenCV convention):  +x right, +y down, +z forward.
- This repo's depth unprojection / NICE-SLAM Replica poses use the
  OpenGL/COLMAP convention: +x right, +y up, camera looks down -z.
- Converting an ORB-SLAM world-from-camera rotation R_wc to the repo's
  convention is a pure right-multiplication by R = diag(1,-1,-1) (det = +1, so
  no handedness flip of the world):
      R_gl = R_wc @ diag(1, -1, -1);   t unchanged  ->  c2w_gl = [R_gl | t]
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: rotation that maps ORB-SLAM(OpenCV) camera coordinates onto the repo's
#: OpenGL convention coordinates (a proper rotation, det = +1)
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def natural_key(name: str | Path):
    p = Path(name).name
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]


def load_matrix16(path: Path) -> np.ndarray:
    vals = np.fromstring(Path(path).read_text(), sep=" ")
    if vals.size != 16:
        raise ValueError(f"expected 16 floats in {path}, got {vals.size}")
    return vals.reshape(4, 4)


def load_k(path: Path) -> tuple:
    """Return (fx, fy, cx, cy) from a 3x3 / 3x4 / 4x4 OpenCV-style K file."""
    rows = [[float(x) for x in ln.split()] for ln in Path(path).read_text().strip().splitlines() if ln.strip()]
    K = np.asarray(rows)
    if K.shape == (4, 4):
        K = K[:3, :4]
    if K.shape in ((3, 3), (3, 4)):
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    raise ValueError(f"unexpected K shape {K.shape} in {path}")


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton convention quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    n = w * w + x * x + y * y + z * z
    if n == 0:
        raise ValueError("zero-norm quaternion")
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ]
    )


def read_tum(path: Path):
    """Parse a TUM-format trajectory file.

    Each line:  t tx ty tz qx qy qz qw   (translation = camera centre in world,
    quaternion = R_wc, i.e. the pose is world-from-camera).

    Returns a list of (t, T_wc 4x4) sorted by timestamp.
    """
    out = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        if len(p) != 8:
            raise ValueError(f"malformed TUM line: {ln!r}")
        t = float(p[0])
        tx, ty, tz = (float(v) for v in p[1:4])
        qx, qy, qz, qw = (float(v) for v in p[4:8])
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
        T[:3, 3] = (tx, ty, tz)
        out.append((t, T))
    out.sort(key=lambda e: e[0])
    return out


def orb_to_gl_c2w(T_wc: np.ndarray) -> np.ndarray:
    """Convert an ORB-SLAM world-from-camera pose (OpenCV frame convention)
    into a camera-to-world pose expressed in this repo's OpenGL convention.

    c2w_gl = [ R_wc @ diag(1,-1,-1) | t_wc ; 0 0 0 1 ]
    """
    out = np.eye(4)
    out[:3, :3] = T_wc[:3, :3] @ CV_TO_GL
    out[:3, 3] = T_wc[:3, 3]
    return out


def rigid_align(src: np.ndarray, dst: np.ndarray):
    """Least-squares rigid (scale=1) alignment mapping src onto dst.

    Returns (R 3x3, t 3,) with dst ~= src @ R.T + t  (i.e. R x_s + t ~ x_d).
    Enforces det(R) = +1 (no reflection).
    """
    s = np.asarray(src, float).reshape(-1, 3)
    d = np.asarray(dst, float).reshape(-1, 3)
    mu_s, mu_d = s.mean(0), d.mean(0)
    H = (s - mu_s).T @ (d - mu_d)
    U, _, Vt = np.linalg.svd(H)
    # With H = U S Vt, the optimal rotation is R = V U^T = Vt.T @ U.T
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t
