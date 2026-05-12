# -*- coding: utf-8 -*-
"""批量预览 WMP 风格单场景地形。

该脚本是一个轻量 wrapper，会逐个调用 inspect_rgbd_terrain.py 启动真实
IsaacSim/IsaacLab 场景并保存 RGB/depth 图像。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


WMP_TERRAINS = ("slope", "stair", "gap", "climb", "tilt", "crawl")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_isaaclab_sh() -> Path:
    return _repo_root().parents[0] / "IsaacLab" / "isaaclab.sh"


parser = argparse.ArgumentParser(description="Preview WMP-style terrain presets with IsaacSim.")
parser.add_argument("--task", type=str, default="b2_rgbd_rough", help="Task name used for preview.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=30, help="Number of zero-action steps.")
parser.add_argument("--save_dir", type=str, default="logs/wmp_terrain_preview", help="Directory for saved previews.")
parser.add_argument(
    "--terrain",
    choices=("all", *WMP_TERRAINS),
    default="all",
    help="WMP terrain preset to preview.",
)
parser.add_argument(
    "--isaaclab_sh",
    type=str,
    default=str(_default_isaaclab_sh()),
    help="Path to IsaacLab isaaclab.sh launcher.",
)
parser.add_argument(
    "--conda_prefix",
    type=str,
    default=os.environ.get("CONDA_PREFIX", ""),
    help="Conda prefix to pass to isaaclab.sh. Use /home/tower/miniconda/envs/isaaclab if your shell is in base.",
)
parser.add_argument("--device", type=str, default=None, help="Device argument forwarded to inspect script.")
parser.add_argument("--gui", action="store_true", help="Open IsaacSim GUI instead of headless rendering.")
parser.add_argument("--no_save_images", action="store_true", help="Do not save RGB/depth images.")
parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait for each headless preview.")
args_cli = parser.parse_args()


def _preview_one(name: str):
    save_dir = Path(args_cli.save_dir) / name
    inspect_script = _repo_root() / "legged_lab" / "scripts" / "inspect_rgbd_terrain.py"
    cmd = [
        args_cli.isaaclab_sh,
        "-p",
        str(inspect_script),
        f"--task={args_cli.task}",
        f"--num_envs={args_cli.num_envs}",
        f"--steps={args_cli.steps}",
        "--enable_cameras",
        f"--wmp_terrain={name}",
        f"--save_dir={save_dir}",
    ]
    if not args_cli.gui:
        cmd.append("--headless")
    if not args_cli.no_save_images:
        cmd.append("--save_images")
    if args_cli.device is not None:
        cmd.append(f"--device={args_cli.device}")

    env = os.environ.copy()
    if args_cli.conda_prefix:
        env["CONDA_PREFIX"] = args_cli.conda_prefix

    print(f"[INFO] Previewing WMP terrain '{name}'")
    print("[INFO] Command: " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=_repo_root(), env=env, check=True, timeout=None if args_cli.gui else args_cli.timeout)
    except subprocess.TimeoutExpired:
        rgb_path = save_dir / "rgb.png"
        depth_path = save_dir / "depth.png"
        if not args_cli.no_save_images and rgb_path.exists() and depth_path.exists():
            print(f"[WARN] Preview '{name}' timed out after saving images; continuing.")
            return
        raise


def main():
    names = WMP_TERRAINS if args_cli.terrain == "all" else (args_cli.terrain,)
    for name in names:
        _preview_one(name)
    print(f"[INFO] WMP terrain previews saved under: {Path(args_cli.save_dir).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Preview command failed with exit code {exc.returncode}.", file=sys.stderr)
        raise
