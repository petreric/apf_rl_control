# APF-Based Reinforcement Learning for UGV Control in Isaac Sim

This project aims to replicate the implementation of a Reinforcement Learning (RL) agent combined with Artificial Potential Fields (APF) to control an Unmanned Ground Vehicle (UGV) in NVIDIA's Isaac Sim. This work builds on concepts presented in the following paper:

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

## Getting Started

To get started with this repository, follow these steps:

### 1. Install IsaacLab
Before using this project, set up IsaacLab by following the installation guide available at the following link:
[IsaacLab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

### 2. Clone the Repository
Once IsaacLab is installed, clone this repository into the `IsaacLab` directory:

```bash
cd ~/IsaacLab
git clone https://github.com/petreric/apf_rl_control.git
```

## Current Progress

The development so far includes:

- **Custom Environment**: The `dev_ws` directory contains the implementation of an environment that includes a custom USD environment and a robot model.
- **Interactive Scene**: The project is currently at the stage of using the interactive scene from the IsaacLab tutorial, which forms the basis for further development.

Devo lavorare su ugv_apf in IsaacLab non questo (è uguale).Ho aggiunto l'ambiente della stanza. Ora devo provare prima a vedere se il training con il drone funziona, poi cambiare gli states e la reward.



