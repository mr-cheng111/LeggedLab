# -*- coding: utf-8 -*-
"""RB160W environment with split leg-position and wheel-velocity actions."""

from legged_lab.envs.base.wheeled_env import WheeledEnv


class RB160WEnv(WheeledEnv):
    """Use position targets for legs and velocity targets for wheel joints.

    The action vector keeps the same order and size as the robot joint list:
    leg joint actions are position offsets, while wheel joint actions are
    velocity commands in rad/s after scaling.
    """

    pass
