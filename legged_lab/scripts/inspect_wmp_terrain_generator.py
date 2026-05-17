# -*- coding: utf-8 -*-
"""快速检查 WMP terrain generator，不启动 IsaacSim。"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect WMP terrain generator output.")
parser.add_argument("--terrain", choices=("slope", "stair", "gap", "climb", "tilt", "crawl"), default="gap")
parser.add_argument("--border_width", type=float, default=None, help="Temporarily override border width for fast checks.")
parser.add_argument("--report_path", type=str, default=None, help="Optional file path for the report.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.terrains import WMP_TERRAIN_CFGS


def main():
    cfg = WMP_TERRAIN_CFGS[args_cli.terrain].copy()
    if args_cli.border_width is not None:
        cfg.border_width = args_cli.border_width
    generator = cfg.class_type(cfg=cfg, device="cpu")
    lines = [
        f"[INFO] terrain={args_cli.terrain}",
        f"[INFO] height_field_raw shape={generator.height_field_raw.shape}",
        f"[INFO] x_edge_mask shape={tuple(generator.x_edge_mask.shape)} true_count={int(generator.x_edge_mask.sum())}",
        f"[INFO] terrain_origins={generator.terrain_origins.tolist()}",
        f"[INFO] mesh vertices={len(generator.terrain_mesh.vertices)} faces={len(generator.terrain_mesh.faces)}",
    ]
    print("\n".join(lines), flush=True)
    if args_cli.report_path is not None:
        path = Path(args_cli.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    simulation_app.close()


if __name__ == "__main__":
    main()
