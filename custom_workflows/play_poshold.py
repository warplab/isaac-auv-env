import argparse

from omni.isaac.lab.app import AppLauncher

import cv2
import numpy as np
import omni.isaac.lab.utils.math as math_utils


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

def signed_angle_diff(angles1, angles2):
    return (angles1 - angles2 + torch.pi) % (2*torch.pi) - torch.pi

def angle_diff(angles1, angles2):
    return torch.minimum(torch.abs(angles1-angles2), 2*torch.pi - torch.abs(angles1-angles2))

def wrap_to_pi(angle_tensor):
    """
    Wraps an angle (or angles) in radians to the range [-pi, pi].
    
    Parameters:
    angle_tensor (torch.Tensor): A tensor of angles in radians.
    
    Returns:
    torch.Tensor: A tensor with angles wrapped to the range [-pi, pi].
    """
    return (angle_tensor + torch.pi) % (2 * torch.pi) - torch.pi

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Modify the configuration
    env_cfg.episode_length_s = 10000
    env_cfg.com_to_cob_offset = [0.0, 0.0, 0.1]

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
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )   
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    des_orientation = math_utils.default_orientation(1, env.unwrapped.device)
    des_euler_xyz = torch.zeros(1,3,device=env.unwrapped.device)
    prev_des_euler_xyz = des_euler_xyz.clone()
    des_orientation = math_utils.quat_from_euler_xyz(torch.tensor(des_euler_xyz[:,0], device=env.unwrapped.device),
                                                    torch.tensor(des_euler_xyz[:,1], device=env.unwrapped.device),
                                                    torch.tensor(des_euler_xyz[:,2], device=env.unwrapped.device))
    des_position_w = torch.zeros(1, 3, device=env.unwrapped.device)

    cv2.imshow("control here", np.zeros((100,100,3)))

    # reset environment

    # get the initial position
    init_pos_w = env.unwrapped._robot.data.root_pos_w
    goal_pos_w = init_pos_w

    obs, _ = env.get_observations()
    obs[:,0:4] = des_orientation
    prev_obs = obs.clone()
    # simulate environment

    while simulation_app.is_running():

        k = cv2.waitKey(1)

        # Update desired orientations
        if k == ord("i"):
            des_euler_xyz[:,1] = prev_des_euler_xyz[:,1] - torch.pi/30
        if k == ord("k"):
            des_euler_xyz[:,1] = prev_des_euler_xyz[:,1] + torch.pi/30
        if k == ord("l"):
            des_euler_xyz[:,2] = prev_des_euler_xyz[:,2] - torch.pi/30
        if k == ord("j"):
            des_euler_xyz[:,2] = prev_des_euler_xyz[:,2] + torch.pi/30
        if k == ord("o"):
            des_euler_xyz[:,0] = prev_des_euler_xyz[:,0] + torch.pi/30
        if k == ord("u"):
            des_euler_xyz[:,0] = prev_des_euler_xyz[:,0] - torch.pi/30

        # Weird debugging yaw wrapping issues
        if k == ord(","):
            des_euler_xyz[:,2] = torch.deg2rad(torch.tensor([170],device=env.unwrapped.device))
        if k == ord("."):
            des_euler_xyz[:,2] = torch.deg2rad(torch.tensor([-170],device=env.unwrapped.device))

        # Updated desired positions
        if k == ord("w"):
            goal_pos_w[:,0] += 0.1
        if k == ord("s"):
            goal_pos_w[:,0] -= 0.1
        if k == ord("a"):
            goal_pos_w[:,1] += 0.1
        if k == ord("d"):
            goal_pos_w[:,1] -= 0.1
        if k == ord("r"):
            goal_pos_w[:,2] += 0.1
        if k == ord("f"):
            goal_pos_w[:,2] -= 0.1

        if k == ord("p"):
            des_euler_xyz = wrap_to_pi(prev_des_euler_xyz)
        # Euler->Quaternion
        # des_euler_xyz = des_euler_xyz % (2.0 * torch.pi)
        # des_euler_xyz = math_utils.wrap_to_pi(des_euler_xyz)

        if k == ord("["):
            des_euler_xyz[:,2] = prev_des_euler_xyz[:,2] + 2*torch.pi
        if k == ord("]"):
            des_euler_xyz[:,2] = prev_des_euler_xyz[:,2] - 2*torch.pi

        # Todo: it is very unclear why this is necessary
        des_euler_xyz = prev_des_euler_xyz - signed_angle_diff(prev_des_euler_xyz, des_euler_xyz)
        des_orientation = math_utils.quat_from_euler_xyz(torch.tensor(des_euler_xyz[:,0], device=env.unwrapped.device),
                                                        torch.tensor(des_euler_xyz[:,1], device=env.unwrapped.device),
                                                        torch.tensor(des_euler_xyz[:,2], device=env.unwrapped.device))

        prev_des_euler_xyz = des_euler_xyz.clone()
        
        # des_orientation = math_utils.quat_unique(des_orientation)
        # des_orientation = math_utils.normalize(des_orientation)

        # Updates the visualizations only
        env.unwrapped._goal = des_orientation
        env.unwrapped._goal_pos_w = goal_pos_w

        # Compute origin offset from global desired positions
        offsets_from_origin = math_utils.quat_apply(math_utils.quat_conjugate(env.unwrapped._robot.data.root_quat_w), goal_pos_w - env.unwrapped._robot.data.root_pos_w)

        # clip offsets
        offsets_from_origin = torch.clamp(offsets_from_origin, min=-0.5, max=0.5)

        # Set the desired controls
        # if torch.sign(des_orientation[:,0]) != torch.sign(prev_obs[:,7]):
        #     des_orientation = -des_orientation

        # print(f"euler: {des_euler_xyz} | quat: {des_orientation} | meas_quat: {obs[:,7:11]}")
        obs = obs.clone()
        obs[:,0:4] = des_orientation
        obs[:,4:7] = offsets_from_origin
        # print("desired orientation: ", des_euler_xyz, ", clipped offsets: ", offsets_from_origin)
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping

            # print(rewards)

            # env stepping

            actions = policy(obs)
            rewards = env.unwrapped._get_rewards()
            obs, rews, _, _ = env.step(actions)
            # print(f"prestep goal: {prev_obs[:,0:4]} | prestep meas: {prev_obs[:,4:7]} | poststep goal: {obs[:,0:4]} | poststep meas: {obs[:,4:7]}")

            prev_obs = obs.clone()
            # print(math_utils.euler_xyz_from_quat(math_utils.quat_unique(obs[:,7:11])))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()