"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
import random
import torch
import math
import numpy as np

##
# Pre-defined configs
##
from custom_task.jetbot import JETBOT_CFG  # isort:skip





@configclass
class JetbotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # room
    room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/simpleGround.usd"))

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    goal_marker = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/goal_marker", spawn=sim_utils.UsdFileCfg(usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/goal_marker.usd"),init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0,0.0,0.0)))


    # articulation
    jetbot: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        data_types=["rgb"],
        prim_path="{ENV_REGEX_NS}/Robot/chassis/rgb_camera/jetbot_camera",
        spawn=None,
        height=224,
        width=224,
        update_period=.1
    )

@configclass 
class JetbotEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    num_actions = 2
    action_scale = 100.0
    observation_space = 10

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Room
    room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/simpleGround.usd"))
    goal_marker_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/goal_marker", spawn=sim_utils.UsdFileCfg(usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/goal_marker.usd"),init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0,0.0,0.0)))

    # Scene
    scene: InteractiveSceneCfg = JetbotSceneCfg(num_envs=1, env_spacing=15.0,replicate_physics=True)

    # Reset
    max_robot_x = 4.8
    min_robot_x = -4.8
    max_robot_y = 4.8
    min_robot_y = -4.8
    goal_distance = 0.2

    # num_channels = 3
    # num_observations = num_channels * scene.camera.height * scene.camera.width

class JetbotEnv(DirectRLEnv):
    cfg: JetbotEnvCfg

    def __init__(self, cfg: JetbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.cfg.robot_cfg
        # self._set_goal_position() 
        self.action_scale = self.cfg.action_scale
        self.previous_apf_value = None  # To store the APF value from the previous step
        self.common_step_counter = 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        #add elements to scene
        light_cfg = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
        room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/simpleGround.usd"))
        self.goal_marker = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/goal_marker", spawn=sim_utils.UsdFileCfg(usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/goal_marker.usd"),init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0,0.0,0.0)))
        self.scene.articulations["robot"] = self.robot

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions)
    def _get_observations(self) -> dict:
        # Robot and goal positions (in 3D)
        ROOM_DIMENSION = 10.0
        RADIUS = 0.3
        robot_position_3d = self.robot.data.root_pos_w  # numpy array [x, y, z]
        goal_position_3d, orientations = self.goal_marker.get_world_poses()

        # Extract 2D positions (assume z-plane is irrelevant for calculations)
        robot_position = robot_position_3d[:2]  # [x, y]
        goal_position = goal_position_3d[:2]  # [x, y]

        # 1. Distance to the goal, normalized
        distance_to_goal = np.linalg.norm(robot_position - goal_position)
        normalized_distance = distance_to_goal / ROOM_DIMENSION

        # 2. Relative angle
        robot_direction = self.robot.data.heading_vector[:2]  # Normalize the 2D heading vector
        goal_direction = goal_position - robot_position
        goal_direction /= np.linalg.norm(goal_direction)  # Normalize goal direction
        cos_theta = np.dot(robot_direction, goal_direction)
        relative_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical issues

        # 3. APF at the robot's position
        Ucom = np.linalg.norm(robot_position - goal_position)**2  # Parabolic APF
        Umax = (0**2 + 0**2)  # APF at (0, 0)
        Umin = (ROOM_DIMENSION**2 + ROOM_DIMENSION**2)  # APF at farthest point
        normalized_apf_com = (Ucom - Umax) / (Umax - Umin)

        # 4. APF at 7 surrounding points
        angles = np.linspace(-math.pi / 2, math.pi / 2, 7)
        apf_values = []
        for angle in angles:
            # Calculate surrounding point position
            offset = RADIUS * np.array([np.cos(angle), np.sin(angle)])
            point_position = robot_position + offset

            # APF value at the point
            Upoint = np.linalg.norm(point_position - goal_position)**2
            normalized_apf = (Upoint - Umax) / (Umax - Umin)
            apf_values.append(normalized_apf)

        # Combine all observations into a tensor
        obs = torch.tensor(
            [normalized_distance, relative_angle, normalized_apf_com] + apf_values,
            dtype=torch.float32
        )

        # Return observations in a dictionary
        observations = {"policy": obs}

        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        # Robot and goal positions (in 3D)
        robot_position_3d = self.robot.data.root_pos_w  # numpy array [x, y, z]
        goal_position_3d = self.goal_marker.data.root_pos_w  # numpy array [x, y, z]

        # Extract 2D positions (project onto the same plane)
        robot_position = robot_position_3d[:2]  # [x, y]
        goal_position = goal_position_3d[:2]  # [x, y]

        # Calculate the current APF value at the robot's position
        current_apf_value = np.linalg.norm(robot_position - goal_position)**2

        # Calculate the change in APF value
        if self.previous_apf_value is None:
            # If this is the first step, assume no APF change
            apf_reward = 0.0
        else:
            apf_reward = self.previous_apf_value - current_apf_value

        # Update the previous APF value
        self.previous_apf_value = current_apf_value

        # Calculate the angle penalty
        robot_direction = self.robot.data.heading_vector[:2]  # Robot's 2D heading vector
        goal_direction = goal_position - robot_position
        goal_direction /= np.linalg.norm(goal_direction)  # Normalize goal direction

        cos_theta = np.dot(robot_direction, goal_direction)
        angle_diff = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle difference in radians
        angle_penalty = -0.4 * (angle_diff**2)  # Penalty proportional to square of angle difference

        # Combine rewards
        total_reward = apf_reward + angle_penalty

        # Optional debug print
        if self.common_step_counter % 10 == 0:
            print(f"Step {self.common_step_counter}: APF Reward = {apf_reward}, Angle Penalty = {angle_penalty}, Total Reward = {total_reward}")

        # Increment the step counter
        self.common_step_counter += 1

        # Return the reward as a torch tensor
        return torch.tensor(total_reward, dtype=torch.float32)


    def _set_goal_position(self):
        robot_orientation = self.robot.data.root_quat_w
        marker = self.scene["goal_marker"]
        # forward_vector = get_basis_vector_z(robot_orientation)
        positions, orientations = marker.get_world_poses()
        random_offset = torch.tensor(random.uniform(-3.0, 3.0))
        positions[:, 1] = random_offset
        marker.set_world_poses(positions, orientations) 
        forward_distance = 1
        # point_in_front = self.robot.data.root_pow_w + forward_distance * forward_vector

        return

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    goal_marker = scene["goal_circle"]
    positions, orientations = goal_marker.get_world_poses()
    # positions[:, 1] = torch.tensor([0.0])
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            random_offset = torch.tensor(random.uniform(-3.0, 3.0))
            positions[:, 1] = random_offset
            goal_marker.set_world_poses(positions, orientations)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
            print(f"position {positions}")
            # print("Received Lidar point cloud shape: ", scene["lidar"].data.output["point_cloud"].shape)

        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=15.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()