# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .warpauv_env import WarpAUVEnv, WarpAUVEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-WarpAUV-Direct-v1",
    entry_point="isaaclab_tasks.direct.roche-isaac-auv-env:WarpAUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVPPORunnerCfg
    },
)
