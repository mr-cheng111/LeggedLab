# -*- coding: utf-8 -*-
"""WMP AMP motion loader。

Derived from ByteDance WMP rsl_rl/datasets/motion_loader.py and Google
Research AMP utilities. 第一版只保留 JSON motion transitions 采样路径。
"""

import glob
import json
from pathlib import Path

import numpy as np
import torch


class AMPMotionDataset:
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE
    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE
    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE
    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE
    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    def __init__(
        self,
        device: str,
        time_between_frames: float,
        motion_files: list[str] | None = None,
        retarget_adapter=None,
        preload_transitions: bool = True,
        num_preload_transitions: int = 100000,
    ):
        self.device = device
        self.time_between_frames = time_between_frames
        motion_files = motion_files or sorted(glob.glob("datasets/wmp_mocap_motions/*.txt"))
        if not motion_files:
            raise FileNotFoundError("No AMP motion files found. Expected datasets/wmp_mocap_motions/*.txt or cfg amp.motion_files.")

        self.trajectories_full = []
        self.trajectory_idxs = []
        self.trajectory_weights = []
        self.trajectory_lens = []
        self.trajectory_frame_durations = []
        for idx, motion_file in enumerate(motion_files):
            path = Path(motion_file)
            with path.open("r", encoding="utf-8") as f:
                motion_json = json.load(f)
            motion_data = np.asarray(motion_json["Frames"], dtype=np.float32)
            frame_duration = float(motion_json["FrameDuration"])
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectories_full.append(
                torch.tensor(motion_data[:, : self.JOINT_VEL_END_IDX], dtype=torch.float32, device=device)
            )
            self.trajectory_idxs.append(idx)
            self.trajectory_weights.append(float(motion_json["MotionWeight"]))
            self.trajectory_frame_durations.append(frame_duration)
            self.trajectory_lens.append(traj_len)
            print(f"[INFO] Loaded AMP motion {path} ({traj_len:.2f}s, {motion_data.shape[0]} frames).")

        self.trajectory_weights = np.asarray(self.trajectory_weights, dtype=np.float64)
        self.trajectory_weights /= np.sum(self.trajectory_weights)
        self.trajectory_lens = np.asarray(self.trajectory_lens, dtype=np.float32)
        self.trajectory_frame_durations = np.asarray(self.trajectory_frame_durations, dtype=np.float32)
        # WMP 原版 discriminator 使用去掉 root_pos/root_quat/toe_pos 后的 30 维 AMP obs:
        # joint_pos(12) + base_lin_vel(3) + base_ang_vel(3) + joint_vel(12)。
        self.retarget_adapter = retarget_adapter
        self.observation_dim = (
            self.JOINT_POSE_END_IDX
            - self.JOINT_POSE_START_IDX
            + self.LINEAR_VEL_END_IDX
            - self.LINEAR_VEL_START_IDX
            + self.ANGULAR_VEL_END_IDX
            - self.ANGULAR_VEL_START_IDX
            + self.JOINT_VEL_END_IDX
            - self.JOINT_VEL_START_IDX
        )

        self.preload_transitions = preload_transitions
        if preload_transitions:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self._full_to_amp_obs(self.get_frame_at_time_batch(traj_idxs, times))
            self.preloaded_s_next = self._full_to_amp_obs(self.get_frame_at_time_batch(traj_idxs, times + time_between_frames))

    def get_preloaded_transitions(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.preload_transitions:
            raise RuntimeError("AMP motion transitions were not preloaded.")
        return self.preloaded_s, self.preloaded_s_next

    def weighted_traj_idx_sample_batch(self, size: int):
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample_batch(self, traj_idxs):
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        return np.maximum(self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst, 0.0)

    def get_frame_at_time_batch(self, traj_idxs, times):
        frames = []
        for traj_idx, time in zip(traj_idxs, times):
            traj = self.trajectories_full[int(traj_idx)]
            traj_len = self.trajectory_lens[int(traj_idx)]
            phase = 0.0 if traj_len <= 0 else float(np.clip(time / traj_len, 0.0, 1.0))
            idx = min(int(phase * (traj.shape[0] - 1)), traj.shape[0] - 1)
            frames.append(traj[idx])
        return torch.stack(frames, dim=0)

    def _full_to_amp_obs(self, frames: torch.Tensor) -> torch.Tensor:
        amp_obs = torch.cat(
            [
                frames[:, self.JOINT_POSE_START_IDX : self.JOINT_POSE_END_IDX],
                frames[:, self.LINEAR_VEL_START_IDX : self.LINEAR_VEL_END_IDX],
                frames[:, self.ANGULAR_VEL_START_IDX : self.ANGULAR_VEL_END_IDX],
                frames[:, self.JOINT_VEL_START_IDX : self.JOINT_VEL_END_IDX],
            ],
            dim=-1,
        )
        if self.retarget_adapter is not None:
            amp_obs = self.retarget_adapter(amp_obs)
        return amp_obs

    def feed_forward_generator(self, num_batches: int, batch_size: int):
        for _ in range(num_batches):
            if self.preload_transitions:
                ids = torch.randint(0, self.preloaded_s.shape[0], (batch_size,), device=self.device)
                yield self.preloaded_s[ids], self.preloaded_s_next[ids]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                state = self._full_to_amp_obs(self.get_frame_at_time_batch(traj_idxs, times))
                next_state = self._full_to_amp_obs(self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames))
                yield state, next_state


AMPLoader = AMPMotionDataset
