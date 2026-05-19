# -*- coding: utf-8 -*-
"""RB160W environment with split leg-position and wheel-velocity actions."""

import torch
from tensordict import TensorDict

from legged_lab.envs.base.base_env import BaseEnv


class RB160WEnv(BaseEnv):
    """Use position targets for legs and velocity targets for wheel joints.

    The action vector keeps the same order and size as the robot joint list:
    leg joint actions are position offsets, while wheel joint actions are
    velocity commands in rad/s after scaling.
    """

    def init_buffers(self):
        super().init_buffers()
        self.wheel_joint_ids, self.wheel_joint_names = self.robot.find_joints(".*_WHEEL_joint")
        self.leg_joint_ids = [idx for idx in range(self.robot.num_joints) if idx not in self.wheel_joint_ids]
        self.leg_joint_names = [self.robot.joint_names[idx] for idx in self.leg_joint_ids]
        self.wheel_velocity_scale = getattr(self.cfg.robot, "wheel_velocity_scale", 8.0)
        self._wheel_control_logged = False

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        if hasattr(self, "wheel_joint_ids"):
            joint_pos = joint_pos.clone()
            joint_pos[:, self.wheel_joint_ids] = 0.0
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
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)

        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        leg_position_targets = (
            clipped_actions[:, self.leg_joint_ids] * self.action_scale
            + self.robot.data.default_joint_pos[:, self.leg_joint_ids]
        )
        wheel_velocity_targets = clipped_actions[:, self.wheel_joint_ids] * self.wheel_velocity_scale

        if not self._wheel_control_logged:
            print(
                "[INFO] RB160W mixed action control: "
                f"legs(position)={self.leg_joint_names}, "
                f"wheels(velocity, scale={self.wheel_velocity_scale})={self.wheel_joint_names}"
            )
            self._wheel_control_logged = True

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(leg_position_targets, joint_ids=self.leg_joint_ids)
            self.robot.set_joint_velocity_target(wheel_velocity_targets, joint_ids=self.wheel_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        obs = TensorDict({"policy": actor_obs, "critic": critic_obs}, batch_size=[self.num_envs])
        self.extras["observations"] = {"critic": critic_obs}
        self.extras["reset_env_ids"] = env_ids
        self.extras["terminal_amp_states"] = terminal_amp_states

        return obs, reward_buf, self.reset_buf, self.extras
