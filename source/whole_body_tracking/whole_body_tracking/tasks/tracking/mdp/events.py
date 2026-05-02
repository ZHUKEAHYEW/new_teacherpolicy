"""训练中使用的 domain randomization 事件。

EventTerm 会在 startup 或 interval 时调用这些函数。这里的随机化用于提升策略鲁棒性，
例如关节零位偏差、质心偏差、摩擦和外部推扰。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    随机化关节默认位置。真实机器人可能因为标定误差，与 URDF 默认值略有不同。

    这个函数还会同步更新 joint position action 的 offset，否则动作中心仍是旧默认角度。
    """
    # 取出 articulation，方便类型提示和后续访问。
    asset: Articulation = env.scene[asset_cfg.name]

    # 保存名义默认关节角，导出 ONNX metadata 时会用到。
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # 解析需要随机化的环境 id。
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 解析需要随机化的关节索引。
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # 用 slice(None) 可以避免不必要的索引开销
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # action offset 不会自动同步，所以这里手动更新。
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """通过叠加随机偏移来随机化刚体质心 CoM。

    .. note::
        这个函数使用 CPU tensor 设置 CoM，建议只在环境初始化时调用。
    """
    # 取出 articulation，方便类型提示和后续访问。
    asset: Articulation = env.scene[asset_cfg.name]
    # 解析环境 id。
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 解析 body 索引。
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 在 CPU 上采样 CoM 偏移，因为这里 PhysX 的 CoM setter 需要 CPU 侧数据。
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # 获取当前 body 质心，形状大致为 (num_assets, num_bodies)。
    coms = asset.root_physx_view.get_coms().clone()

    # 在给定范围内随机化 CoM。
    coms[:, body_ids, :3] += rand_samples

    # 写回新的 CoM。
    asset.root_physx_view.set_coms(coms, env_ids)
