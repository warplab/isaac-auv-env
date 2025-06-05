import argparse

from omni.isaac.lab.app import AppLauncher

import cv2
import numpy as np

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
import csv

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

    print("TEST: ", args_cli.play_checkpoint)

    resume_path = args_cli.play_checkpoint
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )   
    export_policy_as_onnx(ppo_runner.alg.policy, path=export_model_dir, filename="policy.onnx")


    des_cmd = torch.tensor([0,0,0,0,0,0])

    cv2.imshow("control here", np.zeros((100,100,3)))

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        k = cv2.waitKey(1)

        if k == ord("w"):
            des_cmd = torch.tensor([1,0,0,0,0,0])
        elif k == ord("s"):
            des_cmd = torch.tensor([-1,0,0,0,0,0])
        elif k == ord("a"):
            des_cmd = torch.tensor([0,1,0,0,0,0])
        elif k == ord("d"):
            des_cmd = torch.tensor([0,-1,0,0,0,0])
        elif k == ord("q"):
            des_cmd = torch.tensor([0,0,0,0,0,1])
        elif k == ord("e"):
            des_cmd = torch.tensor([0,0,0,0,0,-1])
        elif k == ord("o"):
            des_cmd = torch.tensor([0,0,1,0,0,0])
        elif k == ord("l"):
            des_cmd = torch.tensor([0,0,-1,0,0,0])
        elif k == ord("u"):
            des_cmd = torch.tensor([0,0,0,-1,0,0])
        elif k == ord("j"):
            des_cmd = torch.tensor([0,0,0,1,0,0])
        elif k == ord("i"):
            des_cmd = torch.tensor([0,0,0,0,-1,0])
        elif k == ord("k"):
            des_cmd = torch.tensor([0,0,0,0,1,0])
        else:
            des_cmd = torch.tensor([0,0,0,0,0,0])

        print(des_cmd)
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            print("Observation: ", obs)
            actions = policy(obs)
            # env stepping
            obs, rews, _, _ = env.step(actions)
            obs[:, :6] = des_cmd

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
