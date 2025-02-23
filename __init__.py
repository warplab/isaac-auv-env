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
from .warpauv_vel_env import WarpAUVVelEnvCfg, WarpAUVVelEnv
from .warpauv_stabilize_env import WarpAUVStabilizeEnvCfg, WarpAUVStabilizeEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-WarpAUV-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.learning-based-control:WarpAUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVPPORunnerCfg
    },
)

gym.register(
    id="Isaac-WarpAUV-Velocity-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.learning-based-control:WarpAUVVelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVVelEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVPPORunnerCfg
    },
)

gym.register(
    id="Isaac-WarpAUV-Stabilize-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.learning-based-control:WarpAUVStabilizeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVStabilizeEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVPPORunnerCfg
    },
)