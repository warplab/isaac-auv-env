"""
WarpAUV environment for IsaacLabs

Author: Kevin Chang and Levi "Veevee" Cai (cail@mit.edu)
"""

from __future__ import annotations

import random
import math
import torch
from collections.abc import Sequence

from .assets.warpauv import WARPAUV_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, normalize
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers, RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul
import isaaclab.utils.math as math_utils

##
# Hydrodynamic model
##
from isaaclab.utils.math import quat_apply, quat_conjugate
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations

class WarpAUVEnvWindow(BaseEnvWindow):
    """Window manager for the warpauvenv environment."""

    def __init__(self, env: WarpAUVEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = WarpAUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # robot
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    debug_vis = True

    # env
    decimation = 2
    cap_episode_length = True
    episode_length_s = 3.0
    episode_length_before_reset = None
    num_actions = 6
    num_observations = 17
    num_states = 0
    use_boundaries = True
    max_auv_x = 7
    max_auv_y = 7
    max_auv_z = 7
    starting_depth = 8
    min_goal_steps = 100
    goal_completion_radius = 0.01
    goal_dims = 4
    eval_mode = False

    goal_spawn_radius = 2.0
    init_guidance_rate = 0.1
    init_vel_max = 1.0

    # rewards
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 1000

    rew_scale_pos = 0.2
    rew_scale_ang = 0.5
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.0
    rew_scale_lin_vel = 0.0
    rew_scale_actions = 0.2

    # dynamics
    com_to_cob_offset = [0.0, 0.0, 0.01] # in meters, add this (xyz) to COM to get COB location
    water_rho = 997.0 # kg/m^3
    water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
    rotor_constant = 0.1 / 100.0 # rotor constant used in Gazebo, note /10 because 0.04 is "10x bigger than it should be"
    dyn_time_constant = 0.05 # time constant for linear dynamics for each rotor 
    volume = 0.022747843530591776 # assuming cubic meters - NEUTRALLY BOUYANT. In orignal sim file volume = 0.0223
    # volume = 0.03
    mass = 2.2701e+01 # kg

     # domain randomization
    # todo: isaaclabs has a built-in method somehow
    class domain_randomization:
        use_custom_randomization = True
        # com_to_cob_offset_radius = 0 # uniform from sphere around predicted com_to_cob_offset
        # volume_range = [0.022747843530591776, 0.022747843530591776] # uniform [lowerbound, upperbound]
        # mass_range = [2.2701e+0,2.2701e+0] # uniform [lowerbound, upperbound]
        com_to_cob_offset_radius = 0.05 # uniform from sphere around predicted com_to_cob_offset
        volume_range = [0.019747843530591773, 0.02574784353059178] # uniform [loierbound, upperbound]
        mass_range = [2.2701e+01,2.2701e+01] # uniform [lowerbound, upperbound]


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins # just for visualizations at the moment
        self._step_count = 0
        
        # Get thruster configurations
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1)

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

        # Get specific information about the AUV
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # todo: get inertias from the model or physx view
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)

        # estimated inertial values from a solid rect. prism model (with estimated side lengths of 0.7m, 0.4m, and 0.2m):
        # fake inertial values for warpauv, based on I_ii = (1/12) * mass * (len_j**2 + len_k**2)
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        # todo: cleaner way to handle this
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.copy()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.copy()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True) 

        # Initialize dynamics calculators
        self._init_thruster_dynamics()
        
        # Set initial goals
        self._reset_idx(self._robot._ALL_INDICES)

    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
          self.cfg.com_to_cob_offset = torch.tensor(self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False).reshape(1,3).repeat(self.num_envs, 1)

        # get force calculation functions and rotor dynamics models
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, 6, self.cfg.dyn_time_constant, self.device)
        self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._debug: print("original actions vec: ", actions)
        if self._debug: print("concatenated actions shape: ", self._actions)

        self._actions[:] = actions
        self._actions[:] = torch.clip(self._actions, -1, 1).to(self.device)

    def _apply_action(self) -> None:
        self._thrust[:,0,:], self._moment[:,0,:] = self._compute_dynamics(self._actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

    def _get_observations(self) -> dict:
        #desired_pos_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._goal - self._robot.data.root_pos_w)
        offset_from_origin_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._default_env_origins - self._robot.data.root_pos_w)

        # Uniquefy and normalize all quaternions
        # goal = self._goal
        # root_quat_w = self._robot.data.root_quat_w
        # goal = math_utils.normalize(math_utils.quat_unique(self._goal))
        # root_quat_w = math_utils.normalize(math_utils.quat_unique(self._robot.data.root_quat_w))

        obs = torch.cat(
            [
                self._goal,
                offset_from_origin_b,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        offsets_from_origin = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._default_env_origins - self._robot.data.root_pos_w)

        total_reward = _compute_rewards(
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_actions,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self.reset_terminated,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._goal,
            offsets_from_origin,
            self._completed_envs,
            self._actions
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs)

        self._step_count = self._step_count + 1

        if self.cfg.episode_length_before_reset:
            if self._step_count == self.cfg.episode_length_before_reset:
                time_out = torch.ones(self.num_envs)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]

        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        if not self.cfg.eval_mode:
            # Randomize initial position relative to the origin
            self._default_root_state[env_ids, :3] += self._sample_from_sphere(len(env_ids), self.cfg.goal_spawn_radius)

            # Randomize initial orientation relative to the origin
            # self._default_root_state[env_ids, 3:7] = math_utils.random_orientation(len(env_ids), device=self.device)
            
            # Randomize initial linear and rotational velocities
            # self._default_root_state[env_ids, 7:13] = math_utils.sample_uniform(-self.cfg.init_vel_max, self.cfg.init_vel_max, (len(env_ids), 6), device=self.device)

        self._step_count = 0
        
        # Apply domain randomization
        self._reset_domain(env_ids)

        # Reset goals
        self._reset_goal(env_ids)

        if not self.cfg.eval_mode:
            # Apply guidance (set to goal position and orientation)
            envs_to_guide = math_utils.sample_uniform(0, 1, len(env_ids), self.device) < self.cfg.init_guidance_rate
            env_ids_to_guide = env_ids[envs_to_guide]
            self._default_root_state[env_ids_to_guide, :3] = self._default_env_origins[env_ids_to_guide, :3]
            self._default_root_state[env_ids_to_guide, 3:7] = self._goal[env_ids_to_guide, 0:4]

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)


    # OVERRIDE THIS FUNC TO CHANGE GOAL
    def _reset_goal(self, env_ids: Sequence[int]):
        # Get random orientation
        self._goal[env_ids, 0:4] = math_utils.random_orientation(len(env_ids), device=self.device)

        # Get random yaw orientation with 0 pitch and roll
        # self._goal[env_ids,0:4] = math_utils.random_yaw_orientation(len(env_ids), device=self.device)

        # Get fix RPY
        # rs = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ps = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ys = torch.zeros(len(env_ids), device=self.device) + 0.0
        # self._goal[env_ids,0:4] = math_utils.quat_from_euler_xyz(rs, ps, ys)

    def _reset_domain(self, env_ids: Sequence[int]):
        self.masses[env_ids] = self.masses[env_ids]

        # Randomize COM to COB offset
        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)

        # Randomize volume
        if self.cfg.domain_randomization.use_custom_randomization:
            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(vol_lower, vol_upper, self.volumes[env_ids].shape, self.device)

    def _sample_from_circle(self, num_env_ids, r):
        sampled_radius = r * torch.sqrt(torch.rand((num_env_ids), device=self.device))
        sampled_theta = torch.rand((num_env_ids), device=self.device) * 2 * 3.14159
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return (sampled_x, sampled_y)

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        norms = torch.norm(coords, dim=1).unsqueeze(1)
        coords /= norms

        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1/3)

        return radii * coords

    def _compute_dynamics(self, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute dynamics from actions.
        actions are -1 for full reverse thrust, 1 for full forward thrust - THESE REPRESENT PWM VALUES
        BASED ON LINE 91 of https://gitlab.com/warplab/ros/warpauv/warpauv_simulation/-/blob/master/src/robot_sim_interface.py
        Args:
            actions (torch.Tensor): Actions shape (num_envs, num_actions)

        Returns:
            [torch.Tensor]: Forces sent to the simulation
            [torch.Tensor]: Torques sent to the simulation
        """

        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, 6, 3), device=self.device, dtype=torch.float)
        motorValues = torch.clone(actions) # at this point these are PWM commands between -1 and 1

        if self._debug: print("motorValues: ", motorValues)

        # convert the PWM commands to rad/s using method in https://gitlab.com/warplab/ros/warpauv/warpauv_simulation/-/blob/master/src/robot_sim_interface.py
        motorValues[torch.abs(motorValues) < 0.08] = 0 
        motorValues[motorValues >= 0.08] = -139.0 * (torch.pow(motorValues[motorValues >= 0.08], 2.0)) + 500 * motorValues[motorValues >= 0.08] + 8.28
        motorValues[motorValues <= -0.08] = 161.0 * (torch.pow(motorValues[motorValues <= -0.08], 2.0)) + 517.86 * motorValues[motorValues <= -0.08] - 5.72

        # get the current motor velocities using thruster dynamics
        # TODO: CHECK THAT SIM DT IS CORRECT HERE
        motorValues = self.thruster_dynamics.update(motorValues, self.episode_length_buf * self.sim.cfg.dt)

        # get thruster forces from their speeds using the thruster conversion function 
        motorValues = self.thruster_conversion.convert(motorValues)

        # TODO: this could be taken out of the physics step
        thruster_forces[..., 0] = 1.0 # start with forces in the x direction
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces) # rotate the forces into the thruster's frame

        # apply the force magnitudes to the thruster forces
        thruster_forces = thruster_forces * motorValues.unsqueeze(-1) # make motorValues shape (num_envs, 6, 1))

        # calculate the thruster torques 
        # T = r x F
        # T (num_envs, num_thrusters_per_env, 3)
        # r (num_thrusters_per_env, 3)
        # F (num_envs, num_thrusters_per_env, 3)
        # it should broadcast r to be (num_envs, num_thrusters_per_env, 3)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        # now sum together all the forces/torques on each robot
        thruster_forces = torch.sum(thruster_forces, dim=-2) # sum over the thruster indices
        thruster_torques = torch.sum(thruster_torques, dim=-2) # sum over the thruster indices

        ## Calculate hydrodynamics
        if self._debug: print("gravity magnitude: ", self._gravity_magnitude) 
        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(self._robot.data.root_quat_w, self.cfg.water_rho, self.volumes, abs(self._gravity_magnitude), self.com_to_cob_offsets)

        density_forces, density_torques, viscosity_forces, viscosity_torques = self.force_calculation_functions.calculate_density_and_viscosity_forces(
          self._robot.data.root_quat_w, self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w, self.inertia_tensors, self.inertia_tensors_mean, self.cfg.water_beta, self.cfg.water_rho, self.masses
        )

        if self._debug: print("density forces: ", density_forces)
        if self._debug: print("density torques: ", density_torques)

        if self._debug: print("viscosity forces: ", viscosity_forces)
        if self._debug: print("viscosity torques: ", viscosity_torques)

        if self._debug: print("buoyancy forces: ", buoyancy_forces)
        if self._debug: print("buoyancy torques: ", buoyancy_torques)

        if self._debug: print("thruster forces: ", thruster_forces)
        if self._debug: print("thruster torques: ", thruster_torques)

        forces = density_forces + buoyancy_forces + viscosity_forces + thruster_forces
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques

        if self._debug: print("final forces", forces)
        if self._debug: print("final torques", torques)

        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_ang_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_z_ang_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_z_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_z_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "x_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/x_b"
                self.x_b_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "z_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/z_b"
                self.z_b_visualizer = VisualizationMarkers(marker_cfg)
            
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_ang_visualizer.set_visibility(True)
            self.goal_z_ang_visualizer.set_visibility(True)
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

            if hasattr(self, "goal_ang_visualizer"):
                self.goal_ang_visualizer.set_visibility(False)

            if hasattr(self, "goal_z_ang_visualizer"):
                self.goal_z_ang_visualizer.set_visibility(False)

            if hasattr(self, "x_b_visualizer"):
                self.x_b_visualizer.set_visibility(False)
            
            if hasattr(self, "z_b_visualizer"):
                self.z_b_visualizer.set_visibility(False)

    def _rotate_quat_by_euler_xyz(self, q: torch.tensor, x: float|torch.tensor, y: float|torch.tensor, z: float|torch.tensor, device=None):
        # Assumes q has shape [num_envs, 4]
        num_envs = q.shape[0]
        if device == None:
            device = self.device

        if type(x) == float:
            x = torch.zeros(num_envs, device=device) + x

        if type(y) == float:
            y = torch.zeros(num_envs, device=device) + y
        
        if type(z) == float:
            z = torch.zeros(num_envs, device=device) + z

        iq = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(q, iq)


    def _debug_vis_callback(self, event):
        # Visualize the goal positions
        # self.goal_pos_visualizer.visualize(translations = self._default_env_origins)
        self.goal_pos_visualizer.visualize(translations = self._goal_pos_w)

        # Visualize goal orientations
        goal_quats_w = self._goal
        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_quats_w, scales=ang_marker_scales)

        # Visualize goal orientations via another axis
        goal_z_quat = self._rotate_quat_by_euler_xyz(goal_quats_w, 0.0, -torch.pi/2, 0.0)
        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_z_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_z_quat, scales=ang_marker_scales)

        # Visualize current X-direction
        x_w = self._robot.data.root_quat_w
        x_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        x_w_marker_scales[:, 0] = 1
        self.x_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=x_w, scales=x_w_marker_scales)

        # Visualize current Z-direction
        z_w_quat = self._rotate_quat_by_euler_xyz(self._robot.data.root_quat_w, 0.0, -torch.pi/2, 0.0)
        z_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        z_w_marker_scales[:, 0] = 1
        self.z_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=z_w_quat, scales=z_w_marker_scales)


@torch.jit.script
def quat_dist(q1, q2):
    return 1 - torch.sum(q1*q2, dim=-1)**2

@torch.jit.script
def _compute_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_actions: float,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    goal: torch.Tensor,
    offsets_from_origin: torch.Tensor,
    completed_envs: torch.Tensor,
    actions: torch.Tensor,
):

    # Reward position accuracy, todo: scale the gaussian std appropriately
    rew_pos = rew_scale_pos * torch.exp(-1 * torch.norm(offsets_from_origin, dim=1)**2)

    # Reward angular accuracy, todo: scale the gaussian std appropriately
    # Uniquefy and normalize all quaternions
    rew_ang = rew_scale_ang * torch.exp(-1 * math_utils.quat_error_magnitude(goal[:,:], root_quat[:,:]))

    # Penalize angular velocity
    rew_ang_vel = rew_scale_ang_vel * torch.exp(-1 * torch.norm(ang_vel, dim=1)**2)

    # # Penalize energy consumption
    rew_action = rew_scale_actions * torch.exp(-1 * torch.norm(actions, dim=1)**2)

    total_rew = rew_ang + rew_action + rew_pos + rew_ang_vel
    return total_rew
