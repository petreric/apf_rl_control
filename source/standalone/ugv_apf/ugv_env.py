from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg, AssetBase
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from .simple_room import JETBOT_CFG  # isort:skip
@configclass
class UgvSceneCfg(InteractiveSceneCfg):



    room = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/Room")


@configclass
class UgvEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 15
    state_space = 0
    debug_vis = True

    # ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = UgvSceneCfg(num_envs=4000, env_spacing=11, replicate_physics=True)
    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0

class UgvEnv(DirectRLEnv):
    cfg: UgvEnvCfg

    def __init__(self, cfg: UgvEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # "ang_vel_penalty",
                "apf_difference",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # def _get_observations(self) -> dict:
    #     desired_pos_b, _ = subtract_frame_transforms(
    #         self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
    #     )
    #     robot_position = desired_pos_b
    #     print(f"robot_position: {robot_position}")
    #     obs = torch.cat(
    #         [
    #             self._robot.data.root_lin_vel_b,
    #             self._robot.data.root_ang_vel_b,
    #             self._robot.data.projected_gravity_b,
    #             desired_pos_b,
    #         ],
    #         dim=-1,
    #     )
    #     print(f"obs: {obs}")
    #     observations = {"policy": obs}
    #     return observations

    def _get_observations(self) -> dict:
        """
        Calculate and return the observation dictionary with normalized potential field values.
        """
        # Constants for the attractive potential field
        radius = 0.3  # Radius for surrounding points
        num_points = 11  # Number of points around the robot
        mu = 1.0  # Parabolic constant
        U_max = 100.0  # Max potential field value
        U_min = 0.0  # Min potential field value

        # Ensure all tensors are on the same device (use self.device or the desired device)
        device = self._robot.device  # Assuming the robot's device is either 'cpu' or 'cuda'

        # Robot position (Center of Mass) and desired (goal) position
        # desired_pos_b is the goal in the robot's local frame, convert it to the world frame
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3].to(device), 
            self._robot.data.root_state_w[:, 3:7].to(device),
            self._desired_pos_w.to(device)
        )
        robot_position = self._robot.data.root_state_w[:, :3].to(device)  # CoM position (world frame)
        goal_position = desired_pos_b  # Goal position (robot's local frame)

        # Compute distance from robot CoM to the goal (in robot's body frame)
        dist_robot_to_goal = torch.norm(goal_position, dim=-1) # distance in the body frame
        normalized_distance_to_goal = dist_robot_to_goal / 10.0
        U_robot = 0.5 * mu * dist_robot_to_goal**2  # Potential field value at robot CoM

        # Normalize CoM potential field
        U_robot_norm = (U_robot - U_max) / (U_min - U_max)

        # Compute surrounding points on a circle around the robot (in world frame)
        angles = torch.linspace(0.0, float(2 * torch.pi), num_points, dtype=torch.float32, device=device)  # angles for the points
        # surrounding_points = torch.stack([
        #     robot_position[:, 0].unsqueeze(-1) + radius * torch.cos(angles),  # x positions
        #     robot_position[:, 1].unsqueeze(-1) + radius * torch.sin(angles),  # y positions
        #     robot_position[:, 2].unsqueeze(-1).expand(-1, num_points)  # z positions (constant for all points)
        # ], dim=-1)  # shape: (num_envs, num_points, 3)
        surrounding_points = torch.stack([
            radius * torch.cos(angles),  # x positions
            radius * torch.sin(angles),  # y positions
            torch.zeros_like(angles)  # z positions (elevation = 0)        
            ], dim=-1)  # shape: (num_envs, num_points, 3)

        # Compute distances of surrounding points to the goal (in world frame)
        dist_points_to_goal = torch.norm(surrounding_points - goal_position.unsqueeze(1), dim=-1)  # (num_envs, num_points)
        U_points = 0.5 * mu * dist_points_to_goal**2  # Potential field values for surrounding points

        # Normalize surrounding point potential fields
        U_points_norm = (U_points - U_max) / (U_min - U_max)

        goal_direction = self._desired_pos_w - self._robot.data.root_pos_w
        goal_direction = goal_direction / torch.norm(goal_direction, dim=1, keepdim=True)

        movement_direction = self._robot.data.root_lin_vel_b / torch.norm(self._robot.data.root_lin_vel_b, dim=1, keepdim=True)
        relative_angle = torch.acos(torch.sum(movement_direction * goal_direction, dim=1))
        normalized_relative_angle = relative_angle / torch.pi


        # Concatenate CoM and surrounding points into observation vector
        obs = torch.cat([U_robot.unsqueeze(-1), U_points,self._robot.data.root_ang_vel_b],
                         dim=-1)  # shape: (num_envs, 12)
        # Prepare the observation dictionary
        observations = {"policy": obs}
        return observations

    
    # def _get_rewards(self) -> torch.Tensor:
    #     lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
    #     ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
    #     distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
    #     distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
    #     rewards = {
    #         "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
    #         "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
    #         "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
    #     }
    #     reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
    #     # Logging
    #     for key, value in rewards.items():
    #         self._episode_sums[key] += value
    #     return reward

    def _get_rewards(self) -> torch.Tensor:

        # Angular velocity penalty (sum of squares of each component in body frame)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # Distance to goal reward (mapped with tanh)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        # Attractive Potential Field (APF) difference reward
        mu = 1.0  # APF coefficient
        current_apf = 0.5 * mu * distance_to_goal**2  # APF at the current step

        # Compute the APF difference (initialize to zero for the first step)
        if not hasattr(self, "previous_apf"):
            self.previous_apf = torch.zeros_like(current_apf)  # Initialize previous APF
        apf_difference = self.previous_apf - current_apf
        self.previous_apf = current_apf  # Update for the next step

        # Reward components with scaling factors
        rewards = {
            # "ang_vel_penalty": -ang_vel * 0.09 * self.step_dt,
            "apf_difference": apf_difference 
        }

        # Total reward as the sum of all components
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging rewards for monitoring
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward
    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)