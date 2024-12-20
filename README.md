# Project Status: In Progress

This project is currently under development. While functional, it is not yet complete. Future updates will include additional features and detailed documentation on how to set up, configure, and use the project.

If you use this repository, please cite the following paper:

```
@INPROCEEDINGS{10424064,
  author={Ricioppo, Petre and Celestini, Davide and Capello, Elisa},
  booktitle={2023 IEEE International Workshop on Metrology for Agriculture and Forestry (MetroAgriFor)},
  title={Generalization of Reinforcement Learning through Artificial Potential Fields for agricultural UGVs},
  year={2023},
  volume={},
  number={},
  pages={386-391},
  keywords={Deep learning;Smart agriculture;Reinforcement learning;Pulse width modulation;Numerical models;Collision avoidance;Robots;UGV;Trajectory Planning;Robotics;Autonomous;Navigation;Reinforcement Learning;Artificial Potential Field},
  doi={10.1109/MetroAgriFor58484.2023.10424064}
}
```

# APF-Based Reinforcement Learning for UGV Control in Isaac Sim

This project aims to replicate the implementation of a Reinforcement Learning (RL) agent combined with Artificial Potential Fields (APF) to control an Unmanned Ground Vehicle (UGV) in NVIDIA's Isaac Sim. This work builds on concepts presented in the cited paper.

## Getting Started

To get started with this repository, follow these steps:

### 1. Install IsaacLab
Before using this project, set up IsaacLab by following the installation guide available at the following link:
[IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

### 2. Clone the Repository
Once IsaacLab is installed, clone this repository into the `~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct` directory:

```bash
cd ~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct
git clone https://github.com/petreric/apf_rl_control.git
```

## Current Progress

The development so far includes:

- **Custom Environment**: The `dev_ws` directory contains the implementation of an environment that includes a custom USD environment and a robot model.
- **Training Implemented**: The project implements the training of an RL agent to control the linear and angular velocity of the UGV in an environment with no obstacles. Future updates will introduce obstacle configurations and additional features.
- **Interactive Scene**: The project utilizes the interactive scene from the IsaacLab tutorial, which serves as a foundation for further development.

## Training and Testing the Agent

To train the RL agent using IsaacLab, run the following command:

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-UgvApf-Direct-v0
```

After training is completed, you can test the agent using the following command:

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-UgvApf-Direct-v0 --num_envs 1
```

Stay tuned for updates!

