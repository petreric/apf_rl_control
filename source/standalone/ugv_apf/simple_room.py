import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


JETBOT_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/petre/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ugv_apf/models/simpleGround.usd"
    )
)

