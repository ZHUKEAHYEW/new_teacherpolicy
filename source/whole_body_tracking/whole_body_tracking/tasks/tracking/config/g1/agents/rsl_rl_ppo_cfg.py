"""G1 tracking 任务的 RSL-RL PPO 超参数。

训练脚本会通过 Isaac Lab registry 读取这个配置。常调参数包括：
- `num_steps_per_env`: 每个 PPO rollout 的步数。
- `max_iterations`: 默认最大训练迭代数，可被 `--max_iterations` 覆盖。
- `save_interval`: checkpoint 保存间隔。
- `policy`: actor/critic 网络结构。
- `algorithm`: PPO 损失、学习率、KL 和 GAE 参数。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """50 Hz G1 tracking 的默认 PPO runner 配置。"""

    # 24 steps * 0.02s 控制周期 = 每次 PPO 更新前，每个环境收集 0.48s rollout。
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        # 训练初期加在动作上的探索噪声。
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        # 标准 clipped PPO 设置；adaptive 学习率会根据 desired_kl 调整。
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    """低控制频率环境配套使用的 PPO 配置。"""

    def __post_init__(self):
        super().__post_init__()
        # 改变控制频率后，尽量保持相近的真实时间 rollout 长度。
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        # 调整折扣因子，尽量保持近似的时间常数。
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
