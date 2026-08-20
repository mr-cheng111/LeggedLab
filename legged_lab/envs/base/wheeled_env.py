"""Base environment for robots with position-controlled legs and velocity-controlled wheels."""

import torch
from tensordict import TensorDict

from legged_lab.envs.base.base_env import BaseEnv


class WheeledEnv(BaseEnv):
    """Apply position targets to leg joints and velocity targets to wheel joints."""

    def init_buffers(self):
        if self.cfg.robot.policy_joint_names:
            self.policy_joint_ids, self.policy_joint_names = self.robot.find_joints(
                self.cfg.robot.policy_joint_names, preserve_order=True
            )
        else:
            self.policy_joint_ids = list(range(self.robot.num_joints))
            self.policy_joint_names = list(self.robot.joint_names)
        policy_covers_all_joints = (
            len(self.policy_joint_ids) == self.robot.num_joints
            and len(set(self.policy_joint_ids)) == self.robot.num_joints
        )
        if not policy_covers_all_joints:
            raise RuntimeError(
                "cfg.robot.policy_joint_names must match every articulation joint exactly once; "
                f"matched {len(self.policy_joint_ids)}/{self.robot.num_joints}: {self.policy_joint_names}"
            )
        self.sim_joint_to_action_index = {
            joint_id: action_idx for action_idx, joint_id in enumerate(self.policy_joint_ids)
        }

        wheel_expr = self.cfg.robot.wheel_joint_names_expr
        self.wheel_joint_ids, self.wheel_joint_names = self.robot.find_joints(wheel_expr)
        if not self.wheel_joint_ids:
            raise RuntimeError(f"No wheel joints matched cfg.robot.wheel_joint_names_expr={wheel_expr!r}.")

        # BaseEnv sizes its observation buffers through compute_current_observations(),
        # so the wheeled joint layout must exist before the base buffers are built.
        super().init_buffers()

        wheel_ids = set(self.wheel_joint_ids)
        self.leg_joint_ids = [idx for idx in range(self.robot.num_joints) if idx not in wheel_ids]
        self.leg_joint_names = [self.robot.joint_names[idx] for idx in self.leg_joint_ids]
        self.amp_joint_ids = [idx for idx in self.policy_joint_ids if idx not in wheel_ids]
        self.amp_joint_names = [self.robot.joint_names[idx] for idx in self.amp_joint_ids]
        self.leg_action_ids = [self.sim_joint_to_action_index[idx] for idx in self.leg_joint_ids]
        self.wheel_action_ids = [self.sim_joint_to_action_index[idx] for idx in self.wheel_joint_ids]
        self.wheel_velocity_scale = self.cfg.robot.wheel_velocity_scale

        self.leg_position_scale = torch.full(
            (len(self.leg_joint_ids),), float(self.action_scale), device=self.device
        )
        leg_local_indices = {joint_id: idx for idx, joint_id in enumerate(self.leg_joint_ids)}
        for joint_expr, scale in self.cfg.robot.leg_position_scale.items():
            joint_ids, _ = self.robot.find_joints(joint_expr)
            for joint_id in joint_ids:
                if joint_id in leg_local_indices:
                    self.leg_position_scale[leg_local_indices[joint_id]] = float(scale)
        self._wheel_control_logged = False

    def action_indices_for_joint_ids(self, joint_ids: list[int] | slice) -> list[int] | slice:
        """Map articulation joint indices to policy action indices."""
        if isinstance(joint_ids, slice):
            joint_ids = list(range(self.robot.num_joints))[joint_ids]
        return [self.sim_joint_to_action_index[int(idx)] for idx in joint_ids]

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_pos = joint_pos.clone()
        joint_pos[:, self.wheel_joint_ids] = 0.0
        joint_pos = joint_pos[:, self.policy_joint_ids]
        joint_vel = (robot.data.joint_vel - robot.data.default_joint_vel)[:, self.policy_joint_ids]
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
        feet_contact = torch.max(
            torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1
        )[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )
        return current_actor_obs, current_critic_obs

    def get_amp_observations(self):
        """Return canonical AMP state for the actuated legs, excluding wheel joints."""
        robot = self.robot
        joint_pos = robot.data.joint_pos[:, self.amp_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.amp_joint_ids]
        amp_obs = torch.cat([joint_pos, robot.data.root_lin_vel_b, robot.data.root_ang_vel_b, joint_vel], dim=-1)
        if not self._amp_order_logged:
            print(f"[INFO] AMP leg joint order: {self.amp_joint_names}")
            print(
                f"[INFO] AMP obs layout: joint_pos({len(self.amp_joint_ids)}) + base_lin_vel_b(3) + "
                f"base_ang_vel_b(3) + joint_vel({len(self.amp_joint_ids)})"
            )
            print(f"[INFO] AMP obs shape: {tuple(amp_obs.shape)}, dim={amp_obs.shape[-1]}")
            self._amp_order_logged = True
        return amp_obs

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        leg_position_targets = (
            clipped_actions[:, self.leg_action_ids] * self.leg_position_scale
            + self.robot.data.default_joint_pos[:, self.leg_joint_ids]
        )
        wheel_velocity_targets = clipped_actions[:, self.wheel_action_ids] * self.wheel_velocity_scale

        if not self._wheel_control_logged:
            print(
                "[INFO] Mixed wheeled control: "
                f"legs(position)={self.leg_joint_names}, "
                f"wheels(velocity, scale={self.wheel_velocity_scale})={self.wheel_joint_names}, "
                f"policy_order={self.policy_joint_names}"
            )
            self._wheel_control_logged = True

        has_gui_attr = getattr(self.sim, "has_gui", False)
        has_gui = has_gui_attr() if callable(has_gui_attr) else has_gui_attr
        has_rtx_attr = getattr(self.sim, "has_rtx_sensors", self.rgbd_camera is not None)
        has_rtx_sensors = has_rtx_attr() if callable(has_rtx_attr) else has_rtx_attr
        is_rendering = has_gui or has_rtx_sensors
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(leg_position_targets, joint_ids=self.leg_joint_ids)
            self.robot.set_joint_velocity_target(wheel_velocity_targets, joint_ids=self.wheel_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self.sim_step_counter % self.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

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

        actor_obs, critic_obs = self.compute_observations()
        obs = TensorDict({"policy": actor_obs, "critic": critic_obs}, batch_size=[self.num_envs])
        self.extras["observations"] = {"critic": critic_obs}
        self.extras["reset_env_ids"] = env_ids
        self.extras["terminal_amp_states"] = terminal_amp_states
        self.extras["time_outs"] = self.time_out_buf
        return obs, reward_buf, self.reset_buf, self.extras
