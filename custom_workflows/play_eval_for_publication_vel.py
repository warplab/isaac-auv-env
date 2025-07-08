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
from isaaclab.utils.math import quat_from_angle_axis, quat_error_magnitude, euler_xyz_from_quat

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asymmetric_noise_cfg import *

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )

    action_noise_std = torch.full((env_cfg.num_actions,), args_cli.action_noise_std, dtype=torch.float32, device=env_cfg.sim.device)
    env_cfg.action_noise_model = AsymmetricNoiseModelWithAdditiveBiasCfg(
        noise_cfg = AsymmetricGaussianNoiseCfg(mean=torch.zeros(env_cfg.num_actions, device=env_cfg.sim.device), std=action_noise_std, operation="add"),
        bias_noise_cfg = AsymmetricGaussianNoiseCfg(mean=torch.zeros(env_cfg.num_actions, device=env_cfg.sim.device), std=torch.zeros(env_cfg.num_actions, device=env_cfg.sim.device), operation="add"),
        dims = env_cfg.num_actions
    )

    observation_noise_std = torch.full((env_cfg.num_observations,), args_cli.observation_noise_std, dtype=torch.float32, device=env_cfg.sim.device)
    env_cfg.observation_noise_model = AsymmetricNoiseModelWithAdditiveBiasCfg(
        noise_cfg = AsymmetricGaussianNoiseCfg(mean=torch.zeros(env_cfg.num_observations, device=env_cfg.sim.device), std=observation_noise_std, operation="add"),
        bias_noise_cfg = AsymmetricGaussianNoiseCfg(mean=torch.zeros(env_cfg.num_observations, device=env_cfg.sim.device), std=torch.zeros(env_cfg.num_observations, device=env_cfg.sim.device), operation="add"),
        dims = env_cfg.num_observations
    )

    env_cfg.domain_randomization.use_custom_randomization = False
    env_cfg.com_to_cob_offset[0] += args_cli.com_cob_offset
    env_cfg.volume = args_cli.volume

    env_cfg.use_boundaries = False
    env_cfg.cap_episode_length = False
    env_cfg.episode_length_before_reset = 50

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

    axes_directions_list = [
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (2, 1),
        (2, -1),
        (5, 1),
        (5, -1),
    ]

    # reset environment
    obs, _ = env.get_observations()

    # initialize variables to track current action
    action_iter = 0
    steps_per_action = 50
    action_ix = 0

    # simulate environment
    while action_ix < len(axes_directions_list):
        
        # get next action
        axis, direction = axes_directions_list[action_ix]

        # run everything in inference mode
        with torch.inference_mode():
            obs[0, :3] = torch.Tensor([0.0 if i != axis else direction for i in range(3)])
            obs[0, 3:7] = quat_from_angle_axis(torch.Tensor([direction * 1.5708]), torch.Tensor([[0 if (i + 3) != axis else 1 for i in range(3)]]))[0]
            print("obs: ", obs)

            des_lin_vels = obs[0, :3].cpu().numpy()
            des_ang = obs[0, 3:7]

            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)

            true_lin_vels = obs[0, 7:10].cpu().numpy()
            lin_vel_error = np.linalg.norm(des_lin_vels - true_lin_vels)

            true_ang = obs[0, 10:14]
            ang_error = quat_error_magnitude(des_ang, true_ang).cpu().item()

            total_error = lin_vel_error + ang_error

            reward = (-0.01 * lin_vel_error) + (0.1 * np.exp(-1 * ang_error))

            des_ang_rpy = euler_xyz_from_quat(torch.unsqueeze(des_ang, 0))
            des_ang_rpy = np.array([des_ang_rpy[0][0].cpu().item(), des_ang_rpy[1][0].cpu().item(), des_ang_rpy[2][0].cpu().item()])
            des_ang_rpy = np.where(des_ang_rpy >= math.pi, des_ang_rpy - (2 * math.pi), des_ang_rpy)

            true_ang_rpy = euler_xyz_from_quat(torch.unsqueeze(true_ang, 0))
            true_ang_rpy = np.array([true_ang_rpy[0][0].cpu().item(), true_ang_rpy[1][0].cpu().item(), true_ang_rpy[2][0].cpu().item()])
            true_ang_rpy = np.where(true_ang_rpy >= math.pi, true_ang_rpy - (2 * math.pi), true_ang_rpy)

        log_row = {
            'des_x_vel': des_lin_vels[0],
            'des_y_vel': des_lin_vels[1],
            'des_z_vel': des_lin_vels[2],
            'des_roll': des_ang_rpy[0], 
            'des_pitch': des_ang_rpy[1],
            'des_yaw': des_ang_rpy[2],
            'true_x_vel': true_lin_vels[0],
            'true_y_vel': true_lin_vels[1],
            'true_z_vel': true_lin_vels[2],
            'true_roll': true_ang_rpy[0],
            'true_pitch': true_ang_rpy[1],
            'true_yaw': true_ang_rpy[2],
            'lin_vel_error': lin_vel_error,
            'ang_error': ang_error,
            'total_error': total_error,
            'reward': reward
        }

        wandb.log(log_row)

        log_df = log_df._append(log_row, ignore_index=True)

        action_iter = action_iter + 1
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