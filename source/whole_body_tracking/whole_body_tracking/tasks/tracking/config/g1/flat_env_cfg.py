from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    """G1 专用的轨迹跟踪环境。

    基础环境定义通用 tracking MDP；这里填入 Unitree G1 资产、action scale、
    anchor body，以及用于 reward 跟踪的 body 子集和顺序。
    """

    def __post_init__(self):
        super().__post_init__()

        # 用基于 G1 URDF 的 articulation 替换通用机器人占位符。
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 每个关节的 action scale 来自 robots/g1.py 中的 actuator effort/stiffness。
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        # torso_link 是用于 root 对齐的 tracking anchor。
        self.commands.motion.anchor_body_name = "torso_link"
        # tracked body 列表比完整 URDF 更小，用于稳定 reward 和 observation。
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    """去掉无状态估计时不可用 observation 的环境变体。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    """低频控制环境变体，需要搭配对应 PPO 配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
