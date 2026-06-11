# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import math
import numpy as np
import torch
from torch.nn import functional as F
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.sensors import WMPPartialTiledCamera, WMPPartialTiledCameraCfg, select_wmp_depth_indices
from legged_lab.utils.env_utils.scene import SceneCfg


class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.depth_index_without_crawl_tilt = torch.empty(0, device=self.device, dtype=torch.long)
        self.depth_index = torch.empty(0, device=self.device, dtype=torch.long)
        self.depth_index_inverse = torch.empty(0, device=self.device, dtype=torch.long)
        self.wmp_depth_buffer = None
        self.wmp_vel_violate_buf = None
        self.wmp_fall_buf = None
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        render_interval = self.cfg.sim.render_interval
        if render_interval is None:
            render_interval = self.cfg.sim.decimation
        self.render_interval = max(1, int(render_interval))
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=self.render_interval,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        print(
            "[INFO] BaseEnv timing: "
            f"physics_dt={self.physics_dt:.4f}s, step_dt={self.step_dt:.4f}s, "
            f"render_interval={self.render_interval}, render_dt={self.physics_dt * self.render_interval:.4f}s"
        )

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self._attach_rgbd_cameras_after_scene_creation()
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        self.x_edge_mask = getattr(self.scene.terrain, "x_edge_mask", None)
        self.wmp_edge_query_offset = getattr(self.scene.terrain, "wmp_edge_query_offset", (0.0, 0.0))
        self.wmp_terrain_horizontal_scale = getattr(self.scene.terrain, "wmp_horizontal_scale", 1.0)
        self.wmp_terrain_col_kinds = getattr(self.scene.terrain, "wmp_terrain_col_kinds", ())
        self.wmp_terrain_cols_by_kind = getattr(self.scene.terrain, "wmp_terrain_cols_by_kind", {})
        self.terrain_levels = getattr(self.scene.terrain, "terrain_levels", None)
        self.terrain_types = getattr(self.scene.terrain, "terrain_types", None)
        self.gap_start_col = getattr(self.scene.terrain, "gap_start_col", 0)
        self.climb_end_col = getattr(self.scene.terrain, "climb_end_col", self.cfg.scene.terrain_generator.num_cols if self.cfg.scene.terrain_generator else 0)
        if self.x_edge_mask is not None:
            print(
                "[INFO] WMP x_edge_mask enabled: "
                f"shape={tuple(self.x_edge_mask.shape)}, true_count={int(self.x_edge_mask.sum().item())}, "
                f"offset={self.wmp_edge_query_offset}, horizontal_scale={self.wmp_terrain_horizontal_scale}, "
                f"gap/climb cols=[{self.gap_start_col}, {self.climb_end_col}), "
                f"col_kinds={self.wmp_terrain_col_kinds}"
            )
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]
            self.forward_height_scanner: RayCaster = self.scene.sensors["forward_height_scanner"]
        self.rgbd_camera = self.scene.sensors.get("rgbd_camera")
        if self.rgbd_camera is not None and hasattr(self.rgbd_camera, "camera_env_ids"):
            camera_ids = self.rgbd_camera.camera_env_ids.to(self.device)
            self.wmp_camera_env_ids = camera_ids
            self.wmp_camera_env_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.wmp_camera_env_mask[camera_ids] = True
            self.depth_index = camera_ids
            if hasattr(self.rgbd_camera, "camera_env_id_inverse"):
                self.depth_index_inverse = self.rgbd_camera.camera_env_id_inverse.to(self.device)
            else:
                self.depth_index_inverse = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
                self.depth_index_inverse[self.depth_index] = torch.arange(self.depth_index.numel(), device=self.device)
            if self.depth_index_without_crawl_tilt.numel() > 0:
                valid = self.depth_index_inverse[self.depth_index_without_crawl_tilt] >= 0
                self.depth_index_without_crawl_tilt = self.depth_index_without_crawl_tilt[valid]
            self._init_wmp_depth_buffer()
            print(
                "[INFO] WMP partial RGBD camera enabled: "
                f"model={getattr(self.rgbd_camera.cfg, 'camera_model_name', 'unknown')}, "
                f"camera_envs={int(camera_ids.numel())}/{self.num_envs}, "
                f"depth_index_count={int(self.depth_index.numel())}, "
                f"depth_index_without_crawl_tilt_count={int(self.depth_index_without_crawl_tilt.numel())}, "
                f"tilt_crawl_depth_count={int(self.depth_index.numel() - self.depth_index_without_crawl_tilt.numel())}"
            )

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.reward_curriculum_coef = self._init_reward_curriculum_coef()
        self.terrain_curriculum_max_level = self._terrain_curriculum_allowed_level(0)

        self.init_buffers()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
            self._refresh_wmp_privileged_buffers_from_sim()
        self.reset(env_ids)
        self._is_closed = False

    def _attach_rgbd_cameras_after_scene_creation(self):
        camera = self.cfg.scene.rgbd_camera
        if not camera.enable:
            return

        camera_env_ids = ()
        camera_cfg_cls = TiledCameraCfg
        if camera.partial_camera:
            terrain = getattr(self.scene, "terrain", None)
            depth_index_without_crawl_tilt, depth_index, depth_index_inverse = select_wmp_depth_indices(
                num_envs=self.cfg.scene.num_envs,
                camera_num_envs=camera.partial_camera_num_envs,
                seed=camera.partial_camera_seed,
                terrain_generator=self.cfg.scene.terrain_generator,
                terrain_types=getattr(terrain, "terrain_types", None),
                terrain_cols_by_kind=getattr(terrain, "wmp_terrain_cols_by_kind", None),
                force_tilt_crawl=camera.partial_camera_force_tilt_crawl,
                device="cpu",
            )
            self.depth_index_without_crawl_tilt = depth_index_without_crawl_tilt.to(self.device)
            self.depth_index = depth_index.to(self.device)
            self.depth_index_inverse = depth_index_inverse.to(self.device)
            camera_env_ids = tuple(int(env_id) for env_id in depth_index.tolist())
            camera_cfg_cls = WMPPartialTiledCameraCfg

        prim_path = "{ENV_REGEX_NS}/Robot/" + camera.spawn_prim_path.strip("/")
        prim_path = prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
        spawn_cfg = self._make_camera_spawn_cfg()
        camera_rot = self._make_camera_offset_rotations(camera_env_ids)
        offset_cfg = TiledCameraCfg.OffsetCfg(
            pos=camera.spawn_offset_pos,
            rot=camera.spawn_offset_rot if camera_rot is None else tuple(camera_rot[0].tolist()),
            convention=camera.spawn_offset_convention,
        )

        data_types = []
        if camera.enable_rgb:
            data_types.append("rgb")
        if camera.enable_depth:
            data_types.append("distance_to_image_plane")
        if len(data_types) == 0:
            return

        rgbd_camera = self._make_camera_cfg(
            camera_cfg_cls, prim_path, offset_cfg, spawn_cfg, data_types, camera_env_ids, camera_rot
        )
        if camera.show_visual_model:
            self._spawn_camera_visual_model(prim_path, camera_env_ids)
        self.scene._sensors["rgbd_camera"] = rgbd_camera

    def _spawn_camera_visual_model(self, prim_path: str, camera_env_ids):
        """在真实 USD Camera 下挂一个纯视觉外壳，便于播放时确认相机安装位置。"""
        camera = self.cfg.scene.rgbd_camera
        stage = sim_utils.get_current_stage()
        if camera_env_ids:
            camera_paths = [prim_path.replace("env_.*", f"env_{int(env_id)}", 1) for env_id in camera_env_ids]
        else:
            camera_paths = [prim_path.replace("env_.*", f"env_{env_id}", 1) for env_id in range(self.num_envs)]

        scale = float(camera.visual_model_scale)
        body_cfg = sim_utils.CuboidCfg(
            size=(0.055 * scale, 0.040 * scale, 0.032 * scale),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.025), roughness=0.65),
        )
        lens_cfg = sim_utils.CylinderCfg(
            radius=0.013 * scale,
            height=0.018 * scale,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.12, 0.45), roughness=0.25),
        )
        # USD Camera 局部坐标使用 OpenGL 约定: -Z 为朝向，+Y 向上。
        # 这些可视几何放在相机后方(+Z)，避免挡住真实深度相机的视锥。
        for path in camera_paths:
            if not stage.GetPrimAtPath(path).IsValid():
                continue
            body_path = f"{path}/visual_body"
            lens_path = f"{path}/visual_lens"
            if not stage.GetPrimAtPath(body_path).IsValid():
                body_cfg.func(body_path, body_cfg, translation=(0.0, 0.0, 0.035 * scale))
            if not stage.GetPrimAtPath(lens_path).IsValid():
                lens_cfg.func(lens_path, lens_cfg, translation=(0.0, 0.0, 0.005 * scale))

    def _make_camera_spawn_cfg(self):
        camera = self.cfg.scene.rgbd_camera
        focal_length = camera.focal_length
        if camera.horizontal_fov_deg is not None:
            # USD pinhole 相机水平视场:
            #   f = aperture / (2 * tan(FOV / 2))
            # 这里用原版 WMP 的 horizontal_fov=58 deg 反推 focal_length。
            focal_length = camera.horizontal_aperture / (2.0 * math.tan(math.radians(camera.horizontal_fov_deg) / 2.0))
        kwargs = dict(
            focal_length=focal_length,
            focus_distance=camera.focus_distance,
            horizontal_aperture=camera.horizontal_aperture,
            vertical_aperture=camera.vertical_aperture,
            clipping_range=(camera.render_depth_near, camera.render_depth_far or camera.depth_far),
        )
        if camera.camera_model == "pinhole":
            return sim_utils.PinholeCameraCfg(**kwargs)
        if camera.camera_model == "fisheye":
            return sim_utils.FisheyeCameraCfg(**kwargs)
        raise ValueError(f"Unsupported camera_model: {camera.camera_model}")

    def _make_camera_offset_rotations(self, camera_env_ids):
        camera = self.cfg.scene.rgbd_camera
        if not camera.randomize_rotation:
            return None
        count = len(camera_env_ids) if camera_env_ids else self.num_envs
        seed = self.cfg.scene.seed if camera.randomize_rotation_seed is None else camera.randomize_rotation_seed
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        def sample_degrees(bounds):
            low, high = float(bounds[0]), float(bounds[1])
            if abs(high - low) < 1.0e-8:
                return torch.full((count,), math.radians(low), dtype=torch.float32)
            values = torch.rand(count, generator=generator, dtype=torch.float32)
            return torch.deg2rad(values * (high - low) + low)

        base = torch.tensor(camera.spawn_offset_rot, dtype=torch.float32).repeat(count, 1)
        roll = sample_degrees(camera.random_roll_deg)
        pitch = sample_degrees(camera.random_pitch_deg)
        yaw = sample_degrees(camera.random_yaw_deg)
        local_axes = torch.eye(3, dtype=torch.float32)
        q_roll = quat_from_angle_axis(roll, local_axes[0].repeat(count, 1))
        q_pitch = quat_from_angle_axis(pitch, local_axes[1].repeat(count, 1))
        q_yaw = quat_from_angle_axis(yaw, local_axes[2].repeat(count, 1))
        # world 相机约定: forward=+X, up=+Z。四元数作用为 v_w = q * v_cam * q^{-1}。
        # 因此 pitch 应绕相机局部 +Y 轴旋转；正角会把 +X 转向 -Z，即让相机向下俯视。
        return quat_mul(quat_mul(quat_mul(base, q_roll), q_pitch), q_yaw)

    def _make_camera_cfg(self, camera_cfg_cls, prim_path, offset_cfg, spawn_cfg, data_types, camera_env_ids, camera_rot):
        camera = self.cfg.scene.rgbd_camera
        kwargs = dict(
            prim_path=prim_path,
            offset=offset_cfg,
            update_period=camera.update_period,
            width=camera.width,
            height=camera.height,
            data_types=data_types,
            depth_clipping_behavior=camera.depth_clipping_behavior,
            spawn=spawn_cfg,
        )
        if camera_cfg_cls is WMPPartialTiledCameraCfg:
            kwargs.update(
                class_type=WMPPartialTiledCamera,
                full_num_envs=self.cfg.scene.num_envs,
                camera_env_ids=camera_env_ids,
                camera_model_name=camera.model_name,
            )
        camera_cfg = camera_cfg_cls(**kwargs)
        if camera_rot is not None:
            camera_cfg.wmp_camera_offset_rots = camera_rot
        return camera_cfg.class_type(camera_cfg)

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        self.action_delay_mode = self.cfg.domain_rand.action_delay.params.get("mode", "policy_step")
        if self.cfg.domain_rand.action_delay.enable and self.action_delay_mode == "policy_step":
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
        self.last_processed_actions = self.robot.data.default_joint_pos.clone()
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, device=self.device)

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        self.wmp_privileged_contact_cfg = None
        if self._use_original_wmp_critic_obs():
            self.wmp_privileged_contact_cfg = SceneEntityCfg(
                name="contact_sensor", body_names=self.cfg.robot.wmp_privileged_contact_body_names
            )
            self.wmp_privileged_contact_cfg.resolve(self.scene)
        self._amp_order_logged = False

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._init_wmp_privileged_buffers()
        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        if self._use_original_wmp_critic_obs():
            current_critic_obs = self._compute_original_wmp_critic_obs(
                current_actor_obs=current_actor_obs,
                root_lin_vel=root_lin_vel,
                net_contact_forces=net_contact_forces,
            )
        else:
            feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
            current_critic_obs = torch.cat(
                [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
            )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)
        if self._use_original_wmp_critic_obs() and critic_obs.shape[-1] != 285:
            raise RuntimeError(f"Original WMP critic obs must be 285 dim, got {critic_obs.shape[-1]}.")

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        if hasattr(self, "last_push_step_buf"):
            self.last_push_step_buf[env_ids] = -1
        if hasattr(self, "push_recovered_buf"):
            self.push_recovered_buf[env_ids] = True
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos
        latency_steps = 0
        if self.cfg.domain_rand.action_delay.enable and self.action_delay_mode == "sim_step":
            latency_steps = int(
                torch.randint(
                    low=int(self.cfg.domain_rand.action_delay.params["min_delay"]),
                    high=int(self.cfg.domain_rand.action_delay.params["max_delay"]) + 1,
                    size=(),
                    device=self.device,
                ).item()
            )

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for substep in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            action_target = self.last_processed_actions if substep < latency_steps else processed_actions
            self.robot.set_joint_position_target(action_target)
            self.scene.write_data_to_sim()
            if self.cfg.domain_rand.motor_strength.enable:
                self._apply_motor_strength_to_torque()
            self.sim.step(render=False)
            if self.sim_step_counter % self.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        self.last_processed_actions = processed_actions.detach()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self._update_wmp_depth_buffer()

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        reward_buf = self._post_process_rewards(reward_buf)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset(env_ids)
        if self.cfg.robot.terminate_on_wmp_velocity_violation and self.wmp_vel_violate_buf is not None:
            self.extras.setdefault("log", {})["wmp_vel_violate"] = self.wmp_vel_violate_buf.float().mean()
        if self.cfg.robot.terminate_on_wmp_fall and self.wmp_fall_buf is not None:
            self.extras.setdefault("log", {})["wmp_fall"] = self.wmp_fall_buf.float().mean()

        actor_obs, critic_obs = self.compute_observations()
        obs = TensorDict({"policy": actor_obs, "critic": critic_obs}, batch_size=[self.num_envs])
        self.extras["observations"] = {"critic": critic_obs}
        self.extras["reset_env_ids"] = env_ids
        self.extras["terminal_amp_states"] = terminal_amp_states
        self.extras["time_outs"] = self.time_out_buf

        return obs, reward_buf, self.reset_buf, self.extras

    def get_depth_observations(self):
        if self.rgbd_camera is None:
            if self.cfg.scene.rgbd_camera.allow_missing_depth_fallback:
                return torch.full(
                    (self.num_envs, self.cfg.scene.rgbd_camera.height, self.cfg.scene.rgbd_camera.width, 1),
                    self.cfg.scene.rgbd_camera.depth_far,
                    device=self.device,
                )
            raise RuntimeError("RGBD camera is not enabled for this task.")
        return self.rgbd_camera.data.output["distance_to_image_plane"]

    def _init_wmp_depth_buffer(self):
        camera = self.cfg.scene.rgbd_camera
        if self.depth_index.numel() == 0 or not camera.enable_depth:
            self.wmp_depth_buffer = None
            return
        self.wmp_depth_buffer = torch.zeros(
            (self.depth_index.numel(), 2, camera.height, camera.width, 1),
            device=self.device,
            dtype=torch.float32,
        )

    def _update_wmp_depth_buffer(self):
        if self.wmp_depth_buffer is None or self.rgbd_camera is None:
            return
        if self.sim_step_counter % self.render_interval != 0:
            return
        depth = self.rgbd_camera.data.output.get("distance_to_image_plane")
        if depth is None:
            return
        image = self._depth_to_wmp_image(depth.to(self.device))
        if image.shape[0] != self.depth_index.numel():
            raise RuntimeError(
                "WMP depth buffer expects partial camera batch to match depth_index: "
                f"depth_batch={image.shape[0]}, depth_index={self.depth_index.numel()}"
            )
        reset_like = self.episode_length_buf[self.depth_index] <= 1
        self.wmp_depth_buffer[:, 0].copy_(self.wmp_depth_buffer[:, 1])
        self.wmp_depth_buffer[:, 1].copy_(image)
        if torch.any(reset_like):
            ids = reset_like.nonzero(as_tuple=False).flatten()
            self.wmp_depth_buffer[ids, 0].copy_(image[ids])
            self.wmp_depth_buffer[ids, 1].copy_(image[ids])

    def _depth_to_wmp_image(self, depth: torch.Tensor) -> torch.Tensor:
        camera = self.cfg.scene.rgbd_camera
        near = float(camera.depth_near)
        far = float(camera.depth_far)
        depth = torch.nan_to_num(depth.float(), nan=far, posinf=far, neginf=-far)
        if depth.ndim != 4:
            raise ValueError(f"depth must have shape B,H,W,1 or B,1,H,W, got {tuple(depth.shape)}")
        if depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        elif depth.shape[1] != 1:
            raise ValueError(f"depth channel dimension must be 1, got {tuple(depth.shape)}")
        # 原版 IsaacGym 为负深度: image = ((-depth) - near) / (far - near) - 0.5。
        # IsaacLab distance_to_image_plane 为正深度；若后端给负数，则先取反统一成正距离。
        depth = torch.where(depth < 0.0, -depth, depth)
        depth = torch.clamp(depth, near, far)
        depth = (depth - near) / max(far - near, 1.0e-6) - 0.5
        image = F.interpolate(
            depth,
            size=(camera.height, camera.width),
            mode="bilinear",
            align_corners=False,
        )
        return image.permute(0, 2, 3, 1).contiguous()

    def get_wmp_depth_observations(self):
        if self.wmp_depth_buffer is None:
            raise RuntimeError("WMP depth buffer is not initialized.")
        # 原版返回 depth_buffer[:, -2]；最后一帧已经写入 buffer，但 WMP 消费倒数第二帧。
        return self.wmp_depth_buffer[:, 0]

    def get_depth_camera_env_ids(self):
        if self.depth_index.numel() > 0:
            return self.depth_index.to(self.device)
        if self.rgbd_camera is not None and hasattr(self.rgbd_camera, "camera_env_ids"):
            return self.rgbd_camera.camera_env_ids.to(self.device)
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    def get_depth_index(self):
        if self.depth_index.numel() > 0:
            return self.depth_index.to(self.device)
        return self.get_depth_camera_env_ids()

    def get_depth_index_without_crawl_tilt(self):
        return self.depth_index_without_crawl_tilt.to(self.device)

    def get_depth_index_inverse(self):
        if self.depth_index_inverse.numel() == self.num_envs:
            return self.depth_index_inverse.to(self.device)
        inverse = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        depth_index = self.get_depth_index()
        inverse[depth_index] = torch.arange(depth_index.numel(), device=self.device, dtype=torch.long)
        return inverse

    def get_wmp_proprioception(self):
        robot = self.robot
        command = self.command_generator.command[:, :3]
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        prop = torch.cat(
            [
                robot.data.root_ang_vel_b * self.obs_scales.ang_vel,
                robot.data.projected_gravity_b * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
            ],
            dim=-1,
        )
        if prop.shape[-1] != 33:
            raise RuntimeError(f"WMP proprioception must be 33 dim for quadrupeds, got {prop.shape[-1]}.")
        return prop

    def get_wmp_forward_height_map(self):
        if not self.cfg.scene.height_scanner.enable_height_scan:
            return torch.zeros(self.num_envs, 525, device=self.device)
        height_scan = (
            self.forward_height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self.forward_height_scanner.data.ray_hits_w[..., 2]
            - self.cfg.normalization.height_scan_offset
        ) * self.obs_scales.height_scan
        height_scan = torch.clip(height_scan, -1.0, 1.0)
        if height_scan.shape[-1] != 525:
            raise RuntimeError(f"WMP forward height map must be 525 dim, got {height_scan.shape[-1]}.")
        return height_scan

    def get_amp_observations(self):
        robot = self.robot
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        root_lin_vel = robot.data.root_lin_vel_b
        root_ang_vel = robot.data.root_ang_vel_b
        amp_obs = torch.cat([joint_pos, root_lin_vel, root_ang_vel, joint_vel], dim=-1)
        if not self._amp_order_logged:
            print(f"[INFO] AMP joint order: {self.robot.joint_names}")
            print("[INFO] AMP obs layout: joint_pos(12) + base_lin_vel_b(3) + base_ang_vel_b(3) + joint_vel(12)")
            print(f"[INFO] AMP obs shape: {tuple(amp_obs.shape)}, dim={amp_obs.shape[-1]} (WMP original expects 30)")
            self._amp_order_logged = True
        return amp_obs

    def get_terminal_amp_states(self):
        return self.get_amp_observations()

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        if self.cfg.robot.terminate_on_flight:
            feet_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0]
                > self.cfg.robot.terminate_on_flight_threshold
            )
            reset_buf |= torch.sum(feet_contact, dim=-1) < 0.5
        vel_violate = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.robot.terminate_on_wmp_velocity_violation:
            vel_error = self.robot.data.root_lin_vel_b[:, 0] - self.command_generator.command[:, 0]
            threshold = float(self.cfg.robot.wmp_velocity_violation_threshold)
            vel_violate = ((vel_error > threshold) & (self.command_generator.command[:, 0] < 0.0)) | (
                (vel_error < -threshold) & (self.command_generator.command[:, 0] > 0.0)
            )
            terrain_levels = getattr(self, "terrain_levels", None)
            if terrain_levels is not None:
                vel_violate &= terrain_levels > int(self.cfg.robot.wmp_velocity_violation_min_terrain_level) - 1
            else:
                vel_violate &= False
            reset_buf |= vel_violate
        fall = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.robot.terminate_on_wmp_fall:
            # 原版 WMP: fall = (root_states[:, 9] < -3) | (projected_gravity[:, 2] > 0)。
            fall = (self.robot.data.root_lin_vel_w[:, 2] < float(self.cfg.robot.wmp_fall_z_velocity_threshold)) | (
                self.robot.data.projected_gravity_b[:, 2] > float(self.cfg.robot.wmp_fall_projected_gravity_z_threshold)
            )
            reset_buf |= fall
        self.wmp_vel_violate_buf = vel_violate
        self.wmp_fall_buf = fall
        if self.cfg.robot.wmp_time_out_strictly_greater:
            time_out_buf = self.episode_length_buf > self.max_episode_length
        else:
            time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def _init_wmp_privileged_buffers(self):
        """缓存原版 WMP critic 需要的 privileged 域随机化量。

        原版 critic 布局:
            contact_flag(8) + contact_force(12) + d_gain_scale(12) + p_gain_scale(12)
            + com_pos(3) + added_mass(1) + restitution(1) + friction(1)
            + base_lin_vel(3) + actor_obs(45) = 98

        再在 `compute_observations()` 末尾拼接 height_scan(187)，得到原版的 285 维 critic obs。
        A1 WMP 使用记录型 startup EventTerm，同一次 sample 会同时写入物理仿真和这些 buffer；
        这里的初始化值只作为 EventTerm 缺失时的保底。
        """
        if not self._use_original_wmp_critic_obs():
            return
        self.wmp_priv_friction = torch.zeros(self.num_envs, 1, device=self.device)
        self.wmp_priv_restitution = torch.zeros(self.num_envs, 1, device=self.device)
        self.wmp_priv_added_mass = torch.zeros(self.num_envs, 1, device=self.device)
        self.wmp_priv_com_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.wmp_priv_p_gain_scale = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.wmp_priv_d_gain_scale = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.wmp_priv_default_coms = self.robot.root_physx_view.get_coms().to(self.device).clone()
        self.wmp_critic_obs_slices = {
            "contact_flag": slice(0, 8),
            "contact_force": slice(8, 20),
            "d_gain_scale": slice(20, 32),
            "p_gain_scale": slice(32, 44),
            "com_pos": slice(44, 47),
            "added_mass": slice(47, 48),
            "restitution": slice(48, 49),
            "friction": slice(49, 50),
            "base_lin_vel": slice(50, 53),
            "actor_obs": slice(53, 98),
            "height_scan": slice(98, 285),
        }
        self._sample_wmp_privileged_randomization(torch.arange(self.num_envs, device=self.device))

    def _use_original_wmp_critic_obs(self) -> bool:
        return len(self.cfg.robot.wmp_privileged_contact_body_names) > 0

    def _sample_wmp_privileged_randomization(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        def uniform(bounds, shape):
            low, high = float(bounds[0]), float(bounds[1])
            if abs(high - low) < 1.0e-8:
                return torch.full(shape, low, device=self.device)
            return torch.empty(shape, device=self.device).uniform_(low, high)

        events = self.cfg.domain_rand.events
        physics_material = getattr(events, "physics_material", None)
        if physics_material is not None:
            params = physics_material.params
            self.wmp_priv_friction[env_ids] = uniform(params.get("static_friction_range", (1.0, 1.0)), (len(env_ids), 1))
            self.wmp_priv_restitution[env_ids] = uniform(params.get("restitution_range", (0.0, 0.0)), (len(env_ids), 1))

        add_base_mass = getattr(events, "add_base_mass", None)
        if add_base_mass is not None:
            params = add_base_mass.params
            if params.get("operation", "add") == "add":
                self.wmp_priv_added_mass[env_ids] = uniform(
                    params.get("mass_distribution_params", (0.0, 0.0)), (len(env_ids), 1)
                )

        randomize_base_com = getattr(events, "randomize_base_com", None)
        if randomize_base_com is not None:
            com_range = randomize_base_com.params.get("com_range", {})
            self.wmp_priv_com_pos[env_ids, 0] = uniform(com_range.get("x", (0.0, 0.0)), (len(env_ids),))
            self.wmp_priv_com_pos[env_ids, 1] = uniform(com_range.get("y", (0.0, 0.0)), (len(env_ids),))
            self.wmp_priv_com_pos[env_ids, 2] = uniform(com_range.get("z", (0.0, 0.0)), (len(env_ids),))

        randomize_actuator_gains = getattr(events, "randomize_actuator_gains", None)
        if randomize_actuator_gains is not None:
            params = randomize_actuator_gains.params
            if params.get("operation", "scale") == "scale":
                self.wmp_priv_p_gain_scale[env_ids] = uniform(
                    params.get("stiffness_distribution_params", (1.0, 1.0)), (len(env_ids), self.num_actions)
                ) - 1.0
                self.wmp_priv_d_gain_scale[env_ids] = uniform(
                    params.get("damping_distribution_params", (1.0, 1.0)), (len(env_ids), self.num_actions)
                ) - 1.0

    def _refresh_wmp_privileged_buffers_from_sim(self):
        if not self._use_original_wmp_critic_obs():
            return
        env_ids = torch.arange(self.num_envs, device=self.device)

        try:
            materials = self.robot.root_physx_view.get_material_properties().to(self.device)
            if materials.ndim == 3 and materials.shape[-1] >= 3:
                self.wmp_priv_friction[env_ids] = materials[..., 0].mean(dim=1, keepdim=True)
                self.wmp_priv_restitution[env_ids] = materials[..., 2].mean(dim=1, keepdim=True)
        except Exception as exc:
            print(f"[WARN] Failed to sync WMP material privileged obs from sim: {exc}")

        try:
            body_id = self._first_event_body_id("add_base_mass")
            masses = self.robot.root_physx_view.get_masses().to(self.device)
            default_masses = self.robot.data.default_mass.to(self.device)
            self.wmp_priv_added_mass[env_ids] = masses[:, body_id : body_id + 1] - default_masses[:, body_id : body_id + 1]
        except Exception as exc:
            print(f"[WARN] Failed to sync WMP mass privileged obs from sim: {exc}")

        try:
            body_id = self._first_event_body_id("randomize_base_com")
            coms = self.robot.root_physx_view.get_coms().to(self.device)
            self.wmp_priv_com_pos[env_ids] = coms[:, body_id, :3] - self.wmp_priv_default_coms[:, body_id, :3]
        except Exception as exc:
            print(f"[WARN] Failed to sync WMP COM privileged obs from sim: {exc}")

        try:
            default_stiffness = self.robot.data.default_joint_stiffness.to(self.device)
            default_damping = self.robot.data.default_joint_damping.to(self.device)
            stiffness = self.robot.data.joint_stiffness.to(self.device)
            damping = self.robot.data.joint_damping.to(self.device)
            self.wmp_priv_p_gain_scale[env_ids] = stiffness / torch.clamp(default_stiffness, min=1.0e-8) - 1.0
            self.wmp_priv_d_gain_scale[env_ids] = damping / torch.clamp(default_damping, min=1.0e-8) - 1.0
        except Exception as exc:
            print(f"[WARN] Failed to sync WMP gain privileged obs from sim: {exc}")

    def _first_event_body_id(self, event_name: str) -> int:
        event = getattr(self.cfg.domain_rand.events, event_name, None)
        asset_cfg = None if event is None else event.params.get("asset_cfg")
        body_ids = getattr(asset_cfg, "body_ids", None)
        if isinstance(body_ids, slice) or body_ids is None:
            return 0
        if isinstance(body_ids, torch.Tensor):
            return int(body_ids.flatten()[0].item()) if body_ids.numel() > 0 else 0
        return int(body_ids[0]) if len(body_ids) > 0 else 0

    def _fit_last_dim(self, value: torch.Tensor, target_dim: int) -> torch.Tensor:
        dim = value.shape[-1]
        if dim == target_dim:
            return value
        if dim > target_dim:
            return value[..., :target_dim]
        pad_shape = (*value.shape[:-1], target_dim - dim)
        return torch.cat([value, torch.zeros(pad_shape, device=value.device, dtype=value.dtype)], dim=-1)

    def _compute_original_wmp_critic_obs(
        self,
        current_actor_obs: torch.Tensor,
        root_lin_vel: torch.Tensor,
        net_contact_forces: torch.Tensor,
    ) -> torch.Tensor:
        latest_forces = net_contact_forces[:, -1]
        contact_force = latest_forces[:, self.feet_cfg.body_ids].reshape(self.num_envs, -1)
        contact_force = self._fit_last_dim(contact_force, 12) * self.obs_scales.contact_force

        if self.wmp_privileged_contact_cfg is not None:
            contact_flag = torch.norm(latest_forces[:, self.wmp_privileged_contact_cfg.body_ids], dim=-1) > 0.1
        else:
            contact_flag = torch.zeros(self.num_envs, 0, device=self.device, dtype=torch.bool)
        contact_flag = self._fit_last_dim(contact_flag.float(), 8)

        current_critic_obs = torch.cat(
            [
                contact_flag,
                contact_force,
                self.wmp_priv_d_gain_scale * self.obs_scales.pd_gains,
                self.wmp_priv_p_gain_scale * self.obs_scales.pd_gains,
                self.wmp_priv_com_pos * self.obs_scales.com_pos,
                self.wmp_priv_added_mass,
                self.wmp_priv_restitution,
                self.wmp_priv_friction,
                root_lin_vel * self.obs_scales.lin_vel,
                current_actor_obs,
            ],
            dim=-1,
        )
        if current_critic_obs.shape[-1] != 98:
            raise RuntimeError(f"Original WMP critic current obs must be 98 dim, got {current_critic_obs.shape[-1]}.")
        return current_critic_obs

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        self._clamp_terrain_levels(env_ids)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        if self.terrain_curriculum_max_level is not None:
            extras["Curriculum/terrain_max_allowed_level"] = float(self.terrain_curriculum_max_level)
        return extras

    def _terrain_curriculum_allowed_level(self, current_iter: int) -> int | None:
        settings = getattr(self.cfg.scene, "terrain_curriculum", None)
        if settings is None or not settings.enabled:
            return None
        terrain = getattr(self.scene, "terrain", None) if hasattr(self, "scene") else None
        max_level = int(getattr(terrain, "max_terrain_level", 1)) - 1 if terrain is not None else 0
        start_iter, end_iter, start_level, end_level = [float(x) for x in settings.schedule]
        # 线性课程公式：
        # alpha = clamp((iter - start_iter) / (end_iter - start_iter), 0, 1)
        # level = round((1 - alpha) * start_level + alpha * end_level)
        alpha = 1.0 if abs(end_iter - start_iter) < 1.0e-8 else (float(current_iter) - start_iter) / (end_iter - start_iter)
        alpha = max(0.0, min(1.0, alpha))
        level = round((1.0 - alpha) * start_level + alpha * end_level)
        return max(0, min(int(level), max_level))

    def _clamp_terrain_levels(self, env_ids: torch.Tensor):
        max_level = self.terrain_curriculum_max_level
        terrain = getattr(self.scene, "terrain", None)
        if max_level is None or terrain is None or getattr(terrain, "terrain_origins", None) is None:
            return
        terrain.terrain_levels[env_ids] = torch.clamp(terrain.terrain_levels[env_ids], max=int(max_level))
        terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]

    def update_training_curriculum(self, current_iter: int) -> dict[str, float]:
        logs = self.update_reward_curriculum(current_iter)
        allowed_level = self._terrain_curriculum_allowed_level(current_iter)
        if allowed_level is None:
            return logs
        self.terrain_curriculum_max_level = allowed_level
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._clamp_terrain_levels(env_ids)
        logs["Curriculum/terrain_max_allowed_level"] = float(allowed_level)
        return logs

    def _init_reward_curriculum_coef(self) -> dict[str, float]:
        settings = self.cfg.reward_settings
        terms = tuple(settings.reward_curriculum_term)
        schedules = tuple(settings.reward_curriculum_schedule)
        if not settings.reward_curriculum or len(terms) == 0:
            return {}
        if len(terms) != len(schedules):
            raise ValueError(
                "reward_curriculum_term and reward_curriculum_schedule must have the same length, "
                f"got {len(terms)} and {len(schedules)}."
            )
        return {term: float(schedule[2]) for term, schedule in zip(terms, schedules)}

    def update_reward_curriculum(self, current_iter: int) -> dict[str, float]:
        """按 WMP 原式线性更新奖励课程系数。

        对每个 schedule = (iter_start, iter_end, coef_start, coef_end):
            alpha = clamp((iter - iter_start) / (iter_end - iter_start), 0, 1)
            coef = (1 - alpha) * coef_start + alpha * coef_end
        """
        settings = self.cfg.reward_settings
        if not settings.reward_curriculum:
            return {}
        logs = {}
        for term, schedule in zip(settings.reward_curriculum_term, settings.reward_curriculum_schedule):
            start, end, begin_coef, end_coef = [float(x) for x in schedule]
            alpha = 1.0 if abs(end - start) < 1.0e-8 else (float(current_iter) - start) / (end - start)
            alpha = max(0.0, min(1.0, alpha))
            coef = (1.0 - alpha) * begin_coef + alpha * end_coef
            self.reward_curriculum_coef[term] = coef
            logs[f"Curriculum/{term}_reward_coef"] = coef
            logs[f"{term}_curriculum_coef"] = coef
        return logs

    def _post_process_rewards(self, reward_buf: torch.Tensor) -> torch.Tensor:
        settings = self.cfg.reward_settings
        if settings.reward_curriculum and self.reward_curriculum_coef:
            reward_buf = reward_buf.clone()
            for term, coef in self.reward_curriculum_coef.items():
                if term not in self.reward_manager.active_terms:
                    continue
                index = self.reward_manager.active_terms.index(term)
                term_step_reward = self.reward_manager._step_reward[:, index] * self.step_dt
                reward_buf += (float(coef) - 1.0) * term_step_reward
        if settings.only_positive_rewards:
            reward_buf = torch.clip(reward_buf, min=0.0)
        return reward_buf

    def _apply_motor_strength_to_torque(self):
        low, high = self.cfg.domain_rand.motor_strength.range
        effort_target = getattr(self.robot, "_joint_effort_target_sim", None)
        all_indices = getattr(self.robot, "_ALL_INDICES", None)
        if effort_target is None or all_indices is None:
            raise RuntimeError("WMP motor_strength requires IsaacLab Articulation torque buffers.")
        if self.motor_strength.shape != effort_target.shape:
            self.motor_strength = torch.ones_like(effort_target)

        if abs(float(high) - float(low)) < 1.0e-8:
            self.motor_strength.fill_(float(low))
        else:
            self.motor_strength.uniform_(float(low), float(high))

        # 原版 WMP 在 _compute_torques() 完成 torque clip 后执行 tau_final = s * tau。
        # 这里等待 IsaacLab actuator 生成 _joint_effort_target_sim 后再乘 s，并在 sim.step 前覆盖 PhysX force。
        effort_target.mul_(self.motor_strength)
        self.robot.data.applied_torque[:] = effort_target
        self.robot.root_physx_view.set_dof_actuation_forces(effort_target, all_indices)

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        # 兼容新版 rsl_rl：观测需以 TensorDict 的分组形式返回
        obs = TensorDict({"policy": actor_obs, "critic": critic_obs}, batch_size=[self.num_envs])
        self.extras["observations"] = {"critic": critic_obs}
        return obs

    def close(self):
        """显式释放 IsaacLab 场景、传感器回调和 SimulationContext。

        训练脚本在关闭 Omniverse App 前调用该方法。这里先检查对象是否存在，
        再按 sensor/scene -> sim callbacks -> sim instance 的顺序清理，避免
        camera/syntheticdata 回调在 App 关闭阶段继续持有 USD stage。
        """
        if getattr(self, "_is_closed", False):
            return
        print("[INFO] Closing BaseEnv resources.")
        if hasattr(self, "scene"):
            sensors = getattr(self.scene, "sensors", {})
            for name in ("rgbd_camera",):
                if name in sensors:
                    print(f"[INFO] Releasing sensor: {name}")
            del self.scene
        if hasattr(self, "sim"):
            try:
                self.sim.stop()
            except Exception as exc:
                print(f"[WARN] Failed to stop simulation: {exc}")
            for method_name in ("clear_all_callbacks", "clear_instance"):
                method = getattr(self.sim, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:
                        print(f"[WARN] Failed to call sim.{method_name}(): {exc}")
        self._is_closed = True

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
