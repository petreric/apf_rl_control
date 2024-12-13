import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


NOVACARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/petre/IsaacLab/apf_rl_control/source/standalone/dev_ws/custom_task/nova_carter.usd"
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    }
)