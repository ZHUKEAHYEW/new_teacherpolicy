from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """把 RSL-RL 相关命令行参数加入 parser。

    参数：
        parser: 需要追加参数的 argparse parser。
    """
    # 把 RSL-RL 参数放到单独分组里，方便 train/play/replay 复用。
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # 实验日志参数
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # checkpoint 加载参数
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # 日志后端参数
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """根据 task 名称和命令行参数读取 RSL-RL 配置。

    参数：
        task_name: Isaac Lab 任务名称。
        args_cli: 命令行参数。

    返回：
        已经应用命令行覆盖项的 RSL-RL 配置。
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # 先从 Isaac Lab registry 读取任务默认配置，再应用命令行覆盖。
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """用命令行参数更新 RSL-RL 配置。

    参数：
        agent_cfg: RSL-RL agent 配置。
        args_cli: 命令行参数。

    返回：
        更新后的 RSL-RL agent 配置。
    """
    # 只覆盖用户显式传入的值，其他保持配置文件默认值。
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # 设置 neptune 项目名；本地训练一般不使用。
    if agent_cfg.logger == "neptune" and args_cli.log_project_name:
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
