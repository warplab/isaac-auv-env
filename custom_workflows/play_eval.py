import argparse

from omni.isaac.lab.app import AppLauncher

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

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create dir to save logs into
    save_path = os.path.join("source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint[:-3] + "_play")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"[INFO]: Saving results into: {save_path}")

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
        (3, 1),
        (3, -1),
        (4, 1),
        (4, -1),
        (5, 1),
        (5, -1)
    ]

    # reset environment
    obs, _ = env.get_observations()

    # initialize variables to track current action
    action_iter = 0
    steps_per_action = 250

    # simulate environment
    while simulation_app.is_running():
        
        # get next action
        action_ix = action_iter // 50

        if action_ix < len(axes_directions_list):
            axis, direction = axes_directions_list[action_ix]
        else:
            if action_ix == len(axes_directions_list):
                # save the log df
                log_df.to_csv(os.path.join(save_path, 'logs.csv'))

            axis = -1

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = torch.Tensor([[0.0 if i != axis else direction for i in range(6)]])
            actions = actions.to(env_cfg.sim.device)
            print("actions: ", actions)

            # env stepping
            obs, _, _, _ = env.step(actions)

            obs = obs[0]

            des_vels = obs[:6]
            true_vels = obs[6:]
            error = torch.norm(des_vels - true_vels)
            reward = torch.exp(-1 * error)

        log_row = [vel.cpu().item() for vel in des_vels] + [vel.cpu().item() for vel in true_vels] + [error.cpu().item(), reward.cpu().item()]
        log_df[len(log_df.index)] = log_row

        action_iter = action_iter + 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()