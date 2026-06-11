# -*- coding: utf-8 -*-
"""导出 A1 相机相对机身位置预览图。

不依赖 IsaacSim，只用 matplotlib 画出原版 WMP camera_box、当前配置相机和候选安装高度。
坐标系沿用 A1/WMP 机身坐标:
    +X: 前方, +Y: 左侧, +Z: 上方
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


TRUNK_CENTER = (0.0, 0.0, 0.0)
TRUNK_SIZE = (0.50, 0.22, 0.12)
CAMERA_X = 0.27
CAMERA_Y = 0.0
CAMERA_BOX_SIZE = (0.025, 0.09, 0.025)
ORIGINAL_CAMERA_Z = 0.03


def _box_vertices(center: tuple[float, float, float], size: tuple[float, float, float]):
    cx, cy, cz = center
    sx, sy, sz = (value / 2.0 for value in size)
    return [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]


def _box_faces(vertices):
    return [
        [vertices[i] for i in (0, 1, 2, 3)],
        [vertices[i] for i in (4, 5, 6, 7)],
        [vertices[i] for i in (0, 1, 5, 4)],
        [vertices[i] for i in (2, 3, 7, 6)],
        [vertices[i] for i in (1, 2, 6, 5)],
        [vertices[i] for i in (0, 3, 7, 4)],
    ]


def _draw_box(ax, center, size, color, alpha, label):
    vertices = _box_vertices(center, size)
    collection = Poly3DCollection(_box_faces(vertices), facecolors=color, edgecolors="black", linewidths=0.7, alpha=alpha)
    ax.add_collection3d(collection)
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=35, label=label)


def _candidate_style(idx: int, current: bool = False):
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#17becf"]
    return colors[idx % len(colors)], 0.72 if current else 0.45


def _draw_side_view(ax, candidates: list[float], current_z: float):
    trunk_x, _, trunk_z = TRUNK_CENTER
    trunk_sx, _, trunk_sz = TRUNK_SIZE
    trunk_left = trunk_x - trunk_sx / 2.0
    trunk_bottom = trunk_z - trunk_sz / 2.0
    trunk_front = trunk_x + trunk_sx / 2.0

    ax.add_patch(
        Rectangle(
            (trunk_left, trunk_bottom),
            trunk_sx,
            trunk_sz,
            facecolor="#9aa0a6",
            edgecolor="black",
            alpha=0.28,
            label="trunk approx",
        )
    )

    # 简化视线模型：侧视图中用 dz/dx 表示光线俯仰斜率，z_ray = z0 + slope * (x - x0)。
    # 这不是 Isaac 相机完整投影，只用于快速判断安装高度是否容易被机身前缘遮挡。
    ray_slopes = (-0.15, 0.0, 0.15)
    ray_x_end = 0.45
    for idx, z in enumerate(candidates):
        color, alpha = _candidate_style(idx, abs(float(z) - float(current_z)) < 1.0e-8)
        box_sx, _, box_sz = CAMERA_BOX_SIZE
        ax.add_patch(
            Rectangle(
                (CAMERA_X - box_sx / 2.0, float(z) - box_sz / 2.0),
                box_sx,
                box_sz,
                facecolor=color,
                edgecolor="black",
                alpha=alpha,
            )
        )
        ax.text(CAMERA_X + 0.015, float(z), f"z={float(z):.2f}", color=color, va="center", fontsize=8)
        for slope in ray_slopes:
            ray_z_end = float(z) + slope * (ray_x_end - CAMERA_X)
            ax.plot([CAMERA_X, ray_x_end], [float(z), ray_z_end], color=color, alpha=0.22, linewidth=1.1)

        blocked_by_front_edge = float(z) - CAMERA_BOX_SIZE[2] / 2.0 <= trunk_bottom + trunk_sz
        if blocked_by_front_edge:
            ax.scatter([CAMERA_X], [float(z)], marker="x", color=color, s=45)

    ax.axvline(trunk_front, color="#555555", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.text(trunk_front + 0.006, trunk_bottom + trunk_sz + 0.006, "body front/top edge", fontsize=8, color="#555555")
    ax.set_title("Side view: +X forward / +Z up")
    ax.set_xlabel("+X forward (m)")
    ax.set_ylabel("+Z up (m)")
    ax.set_xlim(-0.15, 0.47)
    ax.set_ylim(-0.10, 0.20)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)


def main():
    parser = argparse.ArgumentParser(description="Export A1 camera mount preview image.")
    parser.add_argument("--output", type=str, default="logs/a1_camera_mount_preview.png", help="Output PNG path.")
    parser.add_argument("--current-z", type=float, default=0.10, help="Current camera z offset to compare.")
    parser.add_argument(
        "--candidates",
        type=float,
        nargs="*",
        default=(0.03, 0.06, 0.08, 0.10, 0.12),
        help="Candidate camera z offsets to draw.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates = sorted({float(value) for value in args.candidates} | {float(args.current_z), ORIGINAL_CAMERA_Z})

    fig = plt.figure(figsize=(12, 6), dpi=180)
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title("A1 WMP Camera Mount Preview")

    # A1 trunk 近似包围盒，仅用于观察相机相对机身位置。
    _draw_box(ax, center=TRUNK_CENTER, size=TRUNK_SIZE, color="#9aa0a6", alpha=0.35, label="trunk approx")

    # 原版 WMP URDF 中 camera_box:
    # <origin xyz="0.27 0 0.03"/>，<box size="0.025 0.09 0.025"/>
    _draw_box(
        ax,
        center=(CAMERA_X, CAMERA_Y, ORIGINAL_CAMERA_Z),
        size=CAMERA_BOX_SIZE,
        color="#d62728",
        alpha=0.9,
        label="WMP original camera_box",
    )

    for idx, z in enumerate(candidates):
        if abs(float(z) - ORIGINAL_CAMERA_Z) < 1.0e-8:
            continue
        color, alpha = _candidate_style(idx, abs(float(z) - float(args.current_z)) < 1.0e-8)
        _draw_box(
            ax,
            center=(CAMERA_X, CAMERA_Y, float(z)),
            size=CAMERA_BOX_SIZE,
            color=color,
            alpha=alpha,
            label=f"candidate z={float(z):.2f}",
        )

    # 光轴示意: WMP/Isaac camera 默认沿局部 -Z 成像，但这里画安装点向前方向，辅助看相对位置。
    for idx, z in enumerate(candidates):
        color, _ = _candidate_style(idx)
        ax.quiver(CAMERA_X, CAMERA_Y, float(z), 0.22, 0.0, 0.0, color=color, arrow_length_ratio=0.15, linewidth=1.5)

    ax.set_xlabel("+X forward (m)")
    ax.set_ylabel("+Y left (m)")
    ax.set_zlabel("+Z up (m)")
    ax.set_xlim(-0.15, 0.45)
    ax.set_ylim(-0.22, 0.22)
    ax.set_zlim(-0.10, 0.20)
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left")
    ax.set_box_aspect((0.60, 0.44, 0.30))

    side_ax = fig.add_subplot(122)
    _draw_side_view(side_ax, candidates, args.current_z)

    fig.tight_layout()
    fig.savefig(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
