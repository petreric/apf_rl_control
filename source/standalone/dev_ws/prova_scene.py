# import argparse
# import random

# from omni.isaac.lab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import torch

# import omni.isaac.core.utils.prims as prim_utils

# import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.assets import Articulation, AssetBaseCfg
# from omni.isaac.lab.sim import SimulationContext

# ##
# # Pre-defined configs
# ##
# from custom_task.jetbot import JETBOT_CFG  # isort:skip

# def design_scene() -> tuple[dict, list[list[float]]]:
#     """Designs the scene."""
#     # Ground-plane
#     cfg = sim_utils.GroundPlaneCfg()
#     cfg.func("/World/defaultGroundPlane", cfg)
#     # Lights
#     cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     cfg.func("/World/Light", cfg)

#     # Random origins for the two robots
#     origin1_x = random.uniform(-1.0, 1.0)
#     origin2_x = random.uniform(2.0, 3.0)

#     origins = [[origin1_x, 0.0, 0.0], [origin2_x, 0.0, 0.0]]

#     # Create separate groups called "Origin1", "Origin2"
#     prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
#     prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

#     # Articulation
#     cartpole_cfg = JETBOT_CFG.copy()
#     cartpole_cfg.prim_path = "/World/Origin.*/Robot"
#     cartpole = Articulation(cfg=cartpole_cfg)

#     # return the scene information
#     scene_entities = {"cartpole": cartpole}

#     return scene_entities, origins

# def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
#     """Runs the simulation loop."""
#     # Extract scene entities
#     robot = entities["cartpole"]
#     # Define simulation stepping
#     sim_dt = sim.get_physics_dt()
#     count = 0
#     # Simulation loop
#     while simulation_app.is_running():
#         # Reset
#         if count % 500 == 0:
#             # reset counter
#             count = 0
#             # reset the scene entities
#             root_state = robot.data.default_root_state.clone()
#             root_state[:, :3] += origins
#             robot.write_root_state_to_sim(root_state)
#             # set joint positions with some noise
#             joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
#             joint_pos += torch.rand_like(joint_pos) * 0.1
#             robot.write_joint_state_to_sim(joint_pos, joint_vel)
#             # clear internal buffers
#             robot.reset()
#             print("[INFO]: Resetting robot state...")
#         # Apply random action
#         efforts = torch.randn_like(robot.data.joint_pos) * 5.0
#         robot.set_joint_effort_target(efforts)
#         robot.write_data_to_sim()
#         # Perform step
#         sim.step()
#         # Increment counter
#         count += 1
#         # Update buffers
#         robot.update(sim_dt)

# def main():
#     """Main function."""
#     # Load kit helper
#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim = SimulationContext(sim_cfg)
#     # Set main camera
#     sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
#     # Design scene
#     scene_entities, scene_origins = design_scene()
#     scene_origins = torch.tensor(scene_origins, device=sim.device)
#     # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # Run the simulator
#     run_simulator(sim, scene_entities, scene_origins)

# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from custom_task.jetbot import JETBOT_CFG  # isort:skip





@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    # ground = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"))
    ground = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/simpleGround.usd"))

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    cartpole: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["cartpole"]
    # Define simulation stepping
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
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
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
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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