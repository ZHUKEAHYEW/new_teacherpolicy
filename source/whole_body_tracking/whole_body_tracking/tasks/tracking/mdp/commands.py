"""训练和 play 使用的 motion command 实现。

这个文件是轨迹跟踪任务的核心：
1. `MotionLoader` 读取单个 `.npz` 参考轨迹，并按机器人关节/body 顺序重排。
2. `MotionCollection` 管理一组轨迹，支持多轨迹训练时每个并行环境独立选择轨迹。
3. `MotionCommand` 在每个仿真 step 生成目标关节、目标 body 位姿、误差指标和 reset 初始状态。

训练中机器人不是直接执行 `.npz`，而是把 `.npz` 作为 command/reward 参考，让 PPO 学出可控策略。
"""

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _indices_from_names(source_names: Sequence[str], target_names: Sequence[str], label: str) -> list[int]:
    source_index = {name: i for i, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_index]
    if missing:
        raise KeyError(f"Motion source {label} names are missing required names: {missing}")
    return [source_index[name] for name in target_names]


class MotionLoader:
    """读取并采样单个 motion `.npz` 文件。

    `.npz` 中的关键数组包括 joint_pos/joint_vel/body_pos_w/body_quat_w/body velocity。
    训练时控制频率和轨迹 fps 不一定一致，所以这里提供线性插值采样。
    """

    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        joint_indexes: Sequence[int] | None = None,
        position_offset: Sequence[float] | None = None,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        if joint_indexes is not None:
            joint_pos = joint_pos[:, joint_indexes]
            joint_vel = joint_vel[:, joint_indexes]
        self.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        if position_offset is not None:
            self._body_pos_w += torch.tensor(position_offset, dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    def _sample_linear(self, values: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        """在小数帧位置对 tensor 做线性插值。"""
        frame_times = torch.clamp(frame_times, 0.0, float(self.time_step_total - 1))
        frame_floor = frame_times.floor().long()
        frame_ceil = torch.clamp(frame_floor + 1, max=self.time_step_total - 1)
        blend = (frame_times - frame_floor.float()).view(-1, *([1] * (values.dim() - 1)))
        return values[frame_floor] * (1.0 - blend) + values[frame_ceil] * blend

    def _sample_quat(self, values: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        """插值四元数，并对结果归一化。

        这里使用 normalized linear interpolation，并在插值前处理四元数正负号，
        避免同一个旋转因为符号相反导致插值绕远路。
        """
        frame_times = torch.clamp(frame_times, 0.0, float(self.time_step_total - 1))
        frame_floor = frame_times.floor().long()
        frame_ceil = torch.clamp(frame_floor + 1, max=self.time_step_total - 1)
        q0 = values[frame_floor]
        q1 = values[frame_ceil]
        q1 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -q1, q1)
        blend = (frame_times - frame_floor.float()).view(-1, *([1] * (values.dim() - 1)))
        return torch.nn.functional.normalize(q0 * (1.0 - blend) + q1 * blend, dim=-1)

    def sample_joint_pos(self, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample_linear(self.joint_pos, frame_times)

    def sample_joint_vel(self, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample_linear(self.joint_vel, frame_times)

    def sample_body_pos_w(self, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample_linear(self.body_pos_w, frame_times)

    def sample_body_quat_w(self, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample_quat(self.body_quat_w, frame_times)

    def sample_body_lin_vel_w(self, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample_linear(self.body_lin_vel_w, frame_times)

    def sample_body_ang_vel_w(self, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample_linear(self.body_ang_vel_w, frame_times)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCollection:
    """一个或多个兼容 motion 文件的容器。

    多轨迹训练时 `active_motion_ids[env_id]` 指明每个并行环境正在跟踪哪条轨迹。
    所有轨迹必须有相同的关节维度和 tracked body 维度，否则 observation/reward 维度会变化。
    """

    def __init__(
        self,
        motion_files: Sequence[str],
        body_indexes: Sequence[int],
        device: str = "cpu",
        joint_indexes: Sequence[int] | None = None,
        position_offsets: Sequence[Sequence[float] | None] | None = None,
    ):
        if not motion_files:
            raise ValueError("At least one motion file must be provided.")
        if position_offsets is None:
            position_offsets = [None] * len(motion_files)
        if len(position_offsets) != len(motion_files):
            raise ValueError("motion_position_offsets must have the same length as motion_files.")

        self.motions = [
            MotionLoader(
                motion_file,
                body_indexes,
                device=device,
                joint_indexes=joint_indexes,
                position_offset=position_offset,
            )
            for motion_file, position_offset in zip(motion_files, position_offsets)
        ]
        self.num_motions = len(self.motions)
        self.time_step_totals = torch.tensor(
            [motion.time_step_total for motion in self.motions], dtype=torch.long, device=device
        )
        self.fps = torch.tensor([motion.fps for motion in self.motions], dtype=torch.float32, device=device)
        self.max_time_step_total = int(self.time_step_totals.max().item())
        self.max_duration_s = max(motion.time_step_total / motion.fps for motion in self.motions)

        joint_dim = self.motions[0].joint_pos.shape[1]
        body_dim = self.motions[0].body_pos_w.shape[1]
        for motion in self.motions:
            if motion.joint_pos.shape[1] != joint_dim:
                raise ValueError("All motions must have the same joint dimension after reordering.")
            if motion.body_pos_w.shape[1] != body_dim:
                raise ValueError("All motions must have the same tracked body dimension after reordering.")

    # 这些属性用于兼容旧的单轨迹工具，特别是 ONNX 导出。
    # 多轨迹时把第一条轨迹作为代表轨迹暴露出去。
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motions[0].joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motions[0].joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motions[0].body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motions[0].body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motions[0].body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motions[0].body_ang_vel_w

    def _sample(self, method_name: str, motion_ids: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        """让每个并行环境从自己当前 active motion 中采样。"""
        sample0 = getattr(self.motions[0], method_name)(frame_times[:1])
        output = torch.empty((len(frame_times), *sample0.shape[1:]), dtype=sample0.dtype, device=sample0.device)
        for motion_id, motion in enumerate(self.motions):
            env_ids = torch.where(motion_ids == motion_id)[0]
            if len(env_ids) == 0:
                continue
            output[env_ids] = getattr(motion, method_name)(frame_times[env_ids])
        return output

    def sample_joint_pos(self, motion_ids: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample("sample_joint_pos", motion_ids, frame_times)

    def sample_joint_vel(self, motion_ids: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample("sample_joint_vel", motion_ids, frame_times)

    def sample_body_pos_w(self, motion_ids: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample("sample_body_pos_w", motion_ids, frame_times)

    def sample_body_quat_w(self, motion_ids: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample("sample_body_quat_w", motion_ids, frame_times)

    def sample_body_lin_vel_w(self, motion_ids: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample("sample_body_lin_vel_w", motion_ids, frame_times)

    def sample_body_ang_vel_w(self, motion_ids: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        return self._sample("sample_body_ang_vel_w", motion_ids, frame_times)


class MotionCommand(CommandTerm):
    """Isaac Lab command term：把参考动作转成跟踪目标。"""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        if self.cfg.source_body_names is not None:
            motion_body_indexes = torch.tensor(
                _indices_from_names(self.cfg.source_body_names, self.cfg.body_names, "body"),
                dtype=torch.long,
                device=self.device,
            )
        else:
            motion_body_indexes = self.body_indexes

        motion_joint_indexes = None
        if self.cfg.source_joint_names is not None:
            motion_joint_indexes = _indices_from_names(
                self.cfg.source_joint_names,
                self.robot.data.joint_names,
                "joint",
            )

        # `motion_file` 保留给单轨迹兼容；`motion_files` 用于多轨迹训练。
        motion_files = list(self.cfg.motion_files) if self.cfg.motion_files is not None else [self.cfg.motion_file]
        position_offsets = (
            list(self.cfg.motion_position_offsets)
            if self.cfg.motion_position_offsets is not None
            else [self.cfg.motion_position_offset] * len(motion_files)
        )
        self.motion = MotionCollection(
            motion_files,
            motion_body_indexes,
            device=self.device,
            joint_indexes=motion_joint_indexes,
            position_offsets=position_offsets,
        )
        if self.cfg.fixed_start_frame is not None:
            min_motion_length = int(self.motion.time_step_totals.min().item())
            if self.cfg.fixed_start_frame < 0 or self.cfg.fixed_start_frame >= min_motion_length:
                raise ValueError(
                    f"fixed_start_frame={self.cfg.fixed_start_frame} is outside motion length "
                    f"{min_motion_length}."
                )
        self.motion_frame_steps = self.motion.fps * env.cfg.decimation * env.cfg.sim.dt
        self.active_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_step_f = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        print(
            "[INFO]: Motion timing: "
            f"num_motions={self.motion.num_motions}, control_dt={env.cfg.decimation * env.cfg.sim.dt:.4f}s, "
            f"fps_range=({self.motion.fps.min().item():.3f}, {self.motion.fps.max().item():.3f}), "
            f"frame_step_range=({self.motion_frame_steps.min().item():.3f}, {self.motion_frame_steps.max().item():.3f})"
        )
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.max_duration_s) + 1
        self.bin_failed_count = torch.zeros(
            self.motion.num_motions, self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_motion"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO: 之后可以评估这是否仍是最合适的 policy 输入形式。
        # policy 接收当前参考帧的目标关节位置和目标关节速度。
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.sample_joint_pos(self.active_motion_ids, self.time_step_f)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.sample_joint_vel(self.active_motion_ids, self.time_step_f)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.sample_body_pos_w(self.active_motion_ids, self.time_step_f)
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.sample_body_quat_w(self.active_motion_ids, self.time_step_f)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.sample_body_lin_vel_w(self.active_motion_ids, self.time_step_f)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.sample_body_ang_vel_w(self.active_motion_ids, self.time_step_f)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.sample_body_pos_w(self.active_motion_ids, self.time_step_f)[:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.sample_body_quat_w(self.active_motion_ids, self.time_step_f)[
            :, self.motion_anchor_body_index
        ]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.sample_body_lin_vel_w(self.active_motion_ids, self.time_step_f)[
            :, self.motion_anchor_body_index
        ]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.sample_body_ang_vel_w(self.active_motion_ids, self.time_step_f)[
            :, self.motion_anchor_body_index
        ]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        """保存可解释的跟踪误差，供日志和 debug 使用。"""
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """为 reset 选择起始帧。

        统计每条 motion 的每个时间 bin 的失败次数，并优先从失败较多的
        (motion_id, bin_id) 开始新 episode。
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_env_ids = env_ids[episode_failed]
            failed_motion_ids = self.active_motion_ids[failed_env_ids]
            failed_motion_lengths = self.motion.time_step_totals[failed_motion_ids].float()
            current_bin_index = torch.clamp(
                self.time_step_f[failed_env_ids] * self.bin_count / torch.clamp(failed_motion_lengths, min=1.0),
                0,
                self.bin_count - 1,
            )
            failed_flat_bins = failed_motion_ids * self.bin_count + current_bin_index.long()
            self._current_bin_failed[:] = torch.bincount(
                failed_flat_bins, minlength=self.motion.num_motions * self.bin_count
            ).view(self.motion.num_motions, self.bin_count)

        # 采样起始 bin
        total_bins = self.motion.num_motions * self.bin_count
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(total_bins)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(1),
            (0, self.cfg.adaptive_kernel_size - 1),  # 非因果卷积核
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(self.motion.num_motions, self.bin_count)

        sampling_probabilities = sampling_probabilities.view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_flat_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        sampled_motion_ids = torch.div(sampled_flat_bins, self.bin_count, rounding_mode="floor")
        sampled_bins = sampled_flat_bins % self.bin_count

        self.active_motion_ids[env_ids] = sampled_motion_ids
        sampled_motion_lengths = self.motion.time_step_totals[sampled_motion_ids].float()
        self.time_step_f[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (sampled_motion_lengths - 1.0)
        )
        self.time_steps[env_ids] = self.time_step_f[env_ids].long()

        # 记录采样分布指标
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(total_bins) if total_bins > 1 else torch.ones_like(H)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_motion"][:] = torch.div(imax, self.bin_count, rounding_mode="floor").float()
        self.metrics["sampling_top1_bin"][:] = (imax % self.bin_count).float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        """把选中的环境 reset 到参考姿态，并叠加配置中的随机扰动。"""
        if len(env_ids) == 0:
            return
        if self.cfg.fixed_start_frame is None:
            self._adaptive_sampling(env_ids)
        else:
            if self.motion.num_motions > 1:
                self.active_motion_ids[env_ids] = torch.randint(
                    self.motion.num_motions, (len(env_ids),), dtype=torch.long, device=self.device
                )
            self.time_steps[env_ids] = self.cfg.fixed_start_frame
            self.time_step_f[env_ids] = float(self.cfg.fixed_start_frame)
            self.metrics["sampling_entropy"][:] = 0.0
            self.metrics["sampling_top1_prob"][:] = 1.0
            self.metrics["sampling_top1_bin"][:] = float(self.cfg.fixed_start_frame) / self.motion.max_time_step_total

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """推进参考帧，并刷新相对目标 body 位姿。"""
        self.time_step_f += self.motion_frame_steps[self.active_motion_ids]
        motion_lengths = self.motion.time_step_totals[self.active_motion_ids]
        self.time_steps = torch.minimum(self.time_step_f.long(), motion_lengths - 1)
        env_ids = torch.where(self.time_step_f >= motion_lengths.float() - 1.0)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """motion command 的配置。

    `motion_file`/`motion_position_offset` 是单轨迹接口；
    `motion_files`/`motion_position_offsets` 是多轨迹接口，长度必须互相匹配。
    """

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_files: list[str] | None = None
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    source_joint_names: list[str] | None = None
    source_body_names: list[str] | None = None
    motion_position_offset: tuple[float, float, float] | None = None
    motion_position_offsets: list[tuple[float, float, float] | None] | None = None
    fixed_start_frame: int | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
