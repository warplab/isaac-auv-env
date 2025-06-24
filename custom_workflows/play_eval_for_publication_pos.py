import pdb
import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--eval_name", type=str, required=True, help="Name of the eval run to store in wandb")
parser.add_argument("--custom_weights", type=str, default=None, help="Path to custom weights file")

# Eval parameters
parser.add_argument("--com_cob_offset", type=float, required=True, help="Distance of center of buoyancy from the center of mass along the X axis")
parser.add_argument("--volume", type=float, required=True, help="Volume of the robot for buoyancy force estimates")
parser.add_argument("--action_noise_std", type=float, required=True, help="Standard deviation of action noise distribution")
parser.add_argument("--observation_noise_std", type=float, required=True, help="Standard deviation of observation noise distribution")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import pandas as pd
import wandb
import numpy as np
import math

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab.utils.math import quat_from_angle_axis, quat_error_magnitude, euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_conjugate

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asymmetric_noise_cfg import *

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )

    env_cfg.domain_randomization.use_custom_randomization = False
    env_cfg.com_to_cob_offset[0] += args_cli.com_cob_offset
    env_cfg.volume = args_cli.volume

    env_cfg.use_boundaries = False
    env_cfg.cap_episode_length = False
    env_cfg.episode_length_before_reset = 300

    env_cfg.goal_spawn_radius = 5

    env_cfg.eval_mode = True

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if not args_cli.custom_weights:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.custom_weights
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    wandb.init(
        project=agent_cfg.to_dict()['wandb_project'],
        name=args_cli.eval_name,
        config=env_cfg
    )

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create dir to save logs into
    save_path = os.path.join("source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint[:-3] + "_play")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"[INFO]: Saving results into: {save_path}")

    # path for saving csv logs
    eval_csv_path = os.path.join(save_path, "logs.csv")

    # create dataframe to save results into
    log_df = pd.DataFrame(columns=[
        'des_x_vel', 
        'des_y_vel', 
        'des_z_vel', 
        'des_roll_vel', 
        'des_pitch_vel', 
        'des_yaw_vel',
        'true_x_vel',
        'true_y_vel',
        'true_z_vel',
        'true_roll_vel',
        'true_pitch_vel',
        'true_yaw_vel',
        'mse',
        'reward'
        ])

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    goal_list = [
        ([0, 0, 0], [1, 0, 0]),
        ([0, 0, 0], [-1, 0, 0]),
        ([0, 0, 0], [0, 1, 0]),
        ([0, 0, 0], [0, -1, 0]),
        ([0, 0, 0], [0, 0, 1]),
        ([0, 0, 0], [0, 0, -1]),
        ([1.0472, 0, 0], [0, 0, 0]),
        ([-1.0472, 0, 0], [0, 0, 0]),
        ([0, 1.0472, 0], [0, 0, 0]),
        ([0, -1.0472, 0], [0, 0, 0]),
        ([0, 0, 1.0472], [0, 0, 0]),
        ([0, 0, -1.0472], [0, 0, 0])
    ]

    obs, _ = env.get_observations()

    # initialize variables to track current action
    action_iter = 0
    steps_per_action = 300
    action_ix = 0

    counter = 0

    # simulate environment
    while action_ix < len(goal_list):
        counter = counter + 1

        # get next action
        goal_orientation, goal_pos = goal_list[action_ix]

        # run everything in inference mode
        with torch.inference_mode():
            des_ang_rpy = goal_orientation
            des_ang_quat = quat_from_euler_xyz(torch.Tensor([des_ang_rpy[0]]), torch.Tensor([des_ang_rpy[1]]), torch.Tensor([des_ang_rpy[2]]))
            env.unwrapped._goal[:] = des_ang_quat.to(env_cfg.sim.device)

            obs[0, 0:4] = des_ang_quat[0].to(env_cfg.sim.device)
            offset_from_origin = quat_apply(quat_conjugate(obs[0, 7:11]), torch.Tensor(goal_pos).to(env_cfg.sim.device) + env.unwrapped._default_env_origins[0] - env.unwrapped._robot.data.root_pos_w[0])
            obs[0, 4:7] = offset_from_origin

            # agent stepping
            actions = policy(obs)

            action_cost = torch.norm(actions).cpu().item()
            action_reward = 0.0 * np.exp(-1 * (action_cost ** 2))

            # env stepping
            obs, _, _, _ = env.step(actions)

            true_pos = env.unwrapped._robot.data.root_pos_w[0].cpu().numpy()
            pos_error = np.linalg.norm(true_pos - (goal_pos + env.unwrapped._default_env_origins[0].cpu().numpy()))
            pos_reward = 0.2 * np.exp(-1 * (pos_error ** 2))

            true_ang = obs[0, 7:11]
            ang_error = quat_error_magnitude(des_ang_quat[0], true_ang.cpu()).item()
            ang_reward = 0.5 * np.exp(-1 * (ang_error ** 2))

            true_ang_rpy = euler_xyz_from_quat(torch.unsqueeze(true_ang, 0))
            true_ang_rpy = np.array([true_ang_rpy[0][0].cpu().item(), true_ang_rpy[1][0].cpu().item(), true_ang_rpy[2][0].cpu().item()])
            true_ang_rpy = np.where(true_ang_rpy >= math.pi, true_ang_rpy - (2 * math.pi), true_ang_rpy)

            true_linvel = obs[0, 11:14].cpu().numpy()

            true_angvel = obs[0, 14:17].cpu().numpy()
            angvel_error = np.linalg.norm(true_angvel)
            angvel_reward = 0.0 * np.exp(-1 * (angvel_error**2))

        log_row = {
            'goal_roll': des_ang_rpy[0], 
            'goal_pitch': des_ang_rpy[1],
            'goal_yaw': des_ang_rpy[2],
            'goal_x': goal_pos[0],
            'goal_y': goal_pos[1],
            'goal_z': goal_pos[2],
            'true_roll': true_ang_rpy[0],
            'true_pitch': true_ang_rpy[1],
            'true_yaw': true_ang_rpy[2],
            'true_x': true_pos[0],
            'true_y': true_pos[1],
            'true_z': true_pos[2],
            'true_roll_vel': true_angvel[0],
            'true_pitch_vel': true_angvel[1],
            'true_yaw_vel': true_angvel[2],
            'true_x_vel': true_linvel[0],
            'true_y_vel': true_linvel[1],
            'true_z_vel': true_linvel[2],
            'action_cost': action_cost,
            'pos_error': pos_error,
            'ang_error': ang_error,
            'angvel_error': angvel_error,
            'pos_reward': pos_reward,
            'ang_reward': ang_reward,
            'angvel_reward': angvel_reward,
            'action_reward': action_reward,
            'total_reward': pos_reward + ang_reward + angvel_reward + action_reward
        }

        action_iter = action_iter + 1

        if (action_iter % steps_per_action) != 0:
            wandb.log(log_row)

        log_df = log_df._append(log_row, ignore_index=True)

        action_ix = action_iter // steps_per_action

    # save logs dataframe
    log_df.to_csv(eval_csv_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
