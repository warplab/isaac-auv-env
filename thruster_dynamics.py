"""
Thruster dynamics and model for WarpAUV

Author: Ethan Fahnestock
"""
# based on https://github.com/uuvsimulator/uuv_simulator/blob/master/uuv_gazebo_plugins/uuv_gazebo_plugins/src/Dynamics.cc

from omni.isaac.lab.utils.math import quat_from_euler_xyz
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np 
import torch

def get_thruster_com_and_orientations(device):
  """
  todo: this entire function should be handled by the USD/URDF model and Configuration files, with named actuators
  This function retrieves the thruster extrinsics for a single vehicle
  """
  def create_tf_rpy(x,y,z,rr,rp,ry):
    print(rr,rp,ry)
    shift = torch.Tensor([x, y, z])
    r = quat_from_euler_xyz(torch.Tensor([rr]), torch.Tensor([rp]), torch.Tensor([ry]))[0]
    print(rr, rp, ry, r[0], r[1], r[2], r[3])
    return shift, r

  def create_tf_quat(x,y,z,w,vx,vy,vz):
    shift = torch.Tensor([x, y, z])
    r = torch.Tensor([w, vx, vy, vz])
    return shift, r

  # TODO: think about the format of this, get rid of helper functions
  thruster_info = dict(
    drive_left=create_tf_quat(-0.4127, .1506, -0.0889, 1,0,0,0),
    drive_right = create_tf_quat(-0.4127,-.1506,-0.0889,1,0,0,0),
    rear_left = create_tf_rpy(-0.303, 0.1461, -0.1587, 0, -0.785398, 1.5708),
    rear_right = create_tf_rpy(-0.303, -0.1461, -0.1587, 0, -0.785398, -1.5708),
    front_right = create_tf_rpy(0.0585, -0.1461, -0.0540, 0, 0.785398,-1.5708),
    front_left = create_tf_rpy(0.0585, 0.1461, -0.0540, 0, 0.785398, 1.5708),
  )
  # vector pointing from com->thruster location (thruster, 3)
  # THRUSTER ORDERING IS 
  # 0 - drive_left
  # 1 - drive_right
  # 2 - rear_left
  # 3 - rear_right
  # 4 - front_left
  # 5 - front_right
  thruster_com_offsets = torch.tensor([
    [thruster_info["drive_left"][0][0], thruster_info["drive_left"][0][1], thruster_info["drive_left"][0][2]],
    [thruster_info["drive_right"][0][0], thruster_info["drive_right"][0][1], thruster_info["drive_right"][0][2]],
    [thruster_info["rear_left"][0][0], thruster_info["rear_left"][0][1], thruster_info["rear_left"][0][2]],
    [thruster_info["rear_right"][0][0], thruster_info["rear_right"][0][1], thruster_info["rear_right"][0][2]],
    [thruster_info["front_left"][0][0], thruster_info["front_left"][0][1], thruster_info["front_left"][0][2]],
    [thruster_info["front_right"][0][0], thruster_info["front_right"][0][1], thruster_info["front_right"][0][2]]
  ], dtype=torch.float32, device=device, requires_grad=False)

  # quaternions to go from COM frame to thruster frame (thruster, 4)
  thruster_quats = torch.tensor([
    [thruster_info["drive_left"][1][0], thruster_info["drive_left"][1][1], thruster_info["drive_left"][1][2], thruster_info["drive_left"][1][3]],
    [thruster_info["drive_right"][1][0], thruster_info["drive_right"][1][1], thruster_info["drive_right"][1][2], thruster_info["drive_right"][1][3]],
    [thruster_info["rear_left"][1][0], thruster_info["rear_left"][1][1], thruster_info["rear_left"][1][2], thruster_info["rear_left"][1][3]],
    [thruster_info["rear_right"][1][0], thruster_info["rear_right"][1][1], thruster_info["rear_right"][1][2], thruster_info["rear_right"][1][3]],
    [thruster_info["front_left"][1][0], thruster_info["front_left"][1][1], thruster_info["front_left"][1][2], thruster_info["front_left"][1][3]],
    [thruster_info["front_right"][1][0], thruster_info["front_right"][1][1], thruster_info["front_right"][1][2], thruster_info["front_right"][1][3]]
  ], dtype=torch.float32, device=device, requires_grad=False)
  return thruster_com_offsets, thruster_quats


class Dynamics(ABC):

  def __init__(self, numEnvs:int, num_thrusters_per_env:int, device:torch.device) -> None: 
    self.numEnvs = numEnvs
    self.num_thrusters_per_env = num_thrusters_per_env
    self.device = device
    self.reset_all()

  # maskArr is a boolean array of size (numEnvs) where envs with value=True are reset
  def reset(self, maskArr:list):
    self.state[maskArr,:] = 0.0
    self.prevTime[maskArr] = -1.0

  def reset_all(self):
    self.state = torch.zeros((self.numEnvs, self.num_thrusters_per_env), dtype=torch.float32, device=self.device, requires_grad=False)
    self.prevTime = torch.ones((self.numEnvs), dtype=torch.float32, device=self.device, requires_grad=False) * -1.0

  @abstractmethod
  def update(self, cmd:torch.tensor, t:float) -> float:
    pass

class DynamicsFirstOrder(Dynamics):

  def __init__(self, numEnvs:int, num_thrusters_per_env:int, tau:float, device:torch.device):
    super().__init__(numEnvs=numEnvs, num_thrusters_per_env=num_thrusters_per_env, device=device)
    self.tau = tau

  # cmd: torch.tensor of shape (numEnvs, num_thrusters_per_env) 
  # t: torch.tensor of shape (numEnvs) with the current times 
  # given force commands, update the state of system and report current thrusts 
  def update(self, cmd:torch.tensor, t:torch.tensor) -> float:
    # old method would return state if single time was not set yet
    #if self.prevTime < 0:
    #  self.prevTime = t
    #  return self.state

    # set previously unupdated times to the current time in those envs
    self.prevTime[self.prevTime < 0] = t[self.prevTime < 0]

    # because dt = 0 for previously unupdated times, alpha=1 and we just get the previous state 
    dt = t - self.prevTime
    alpha = torch.exp(-dt/self.tau)
    alpha = torch.zeros_like(alpha) # todo: this wipes out alpha, always just sets it to the command!
    #print(self.state.shape, cmd.shape, alpha.shape)
    #print(dt, alpha, self.state)

    self.state = self.state * alpha.unsqueeze(-1) + (1.0 - alpha).unsqueeze(-1) * cmd
    assert torch.any(self.state == cmd)

    self.prevTime = t
    return self.state

# based on https://github.com/uuvsimulator/uuv_simulator/blob/master/uuv_gazebo_plugins/uuv_gazebo_plugins/src/ThrusterConversionFcn.cc
@dataclass
class ConversionFunction(ABC):

  @abstractmethod
  def convert(self, cmd:np.ndarray) -> float:
    pass

class ConversionFunctionBasic(ConversionFunction):

  # rotorConstant: the rotor constant  
  rotorConstant: float

  def __init__(self, rotorConstant:float):
    super().__init__()
    self.rotorConstant = rotorConstant

  # cmd: np.ndarray of shape (numEnvs, num_thrusters_per_env)
  # converts velocity commands to thrust 
  def convert(self, cmd:torch.tensor) -> float:
    return self.rotorConstant * torch.abs(cmd) * cmd 
  