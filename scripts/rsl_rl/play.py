"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

G1_CANONICAL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_CANONICAL_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=5000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--motion_source_order",
    type=str,
    default="g1_canonical",
    choices=["g1_canonical", "robot"],
    help="Source joint/body order used by the local motion npz file.",
)
parser.add_argument("--manifest_file", type=str, default=None, help="Path to the local batch_manifest.json file.")
parser.add_argument("--terrain_file", type=str, default=None, help="Path to the local terrain USD file.")
parser.add_argument(
    "--terrain_use_manifest_pose",
    action="store_true",
    help="Spawn --terrain_file at manifest terrain_world_pose instead of the origin.",
)
parser.add_argument("--motion_root_body_idx", type=int, default=0, help="Root body index used for manifest alignment.")
parser.add_argument(
    "--fixed_start_frame",
    type=int,
    default=0,
    help="Motion frame used for every episode reset. Use -1 to keep adaptive random start-frame sampling.",
)
parser.add_argument("--env_spacing", type=float, default=None, help="Environment spacing override.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    yaw_rad = math.radians(float(yaw_deg))
    half_yaw = 0.5 * yaw_rad
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def _add_height_scan_terrain_target(env_cfg: ManagerBasedRLEnvCfg):
    height_scanner = getattr(env_cfg.scene, "height_scanner", None)
    if height_scanner is None or not hasattr(height_scanner.__class__, "RaycastTargetCfg"):
        return

    terrain_target = "{ENV_REGEX_NS}/Terrain"
    existing_targets = []
    for target in height_scanner.mesh_prim_paths:
        existing_targets.append(target if isinstance(target, str) else target.prim_expr)
    if terrain_target in existing_targets:
        return

    height_scanner.mesh_prim_paths.append(
        height_scanner.__class__.RaycastTargetCfg(prim_expr=terrain_target, track_mesh_transforms=False)
    )
    print(f"[INFO]: Added height-scan terrain target: {terrain_target}")


def _load_manifest_entry(manifest_file: str | None, motion_file: str) -> dict | None:
    if manifest_file is None:
        return None

    manifest_path = os.path.abspath(manifest_file)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Invalid manifest file path: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        payload = json.load(f)

    trajectories = payload.get("trajectories", [])
    if not trajectories:
        raise ValueError(f"No trajectories found in manifest: {manifest_path}")

    motion_name = os.path.basename(motion_file)
    motion_stem = os.path.splitext(motion_name)[0]
    for item in trajectories:
        trajectory_path = item.get("trajectory_path", "")
        trajectory_name = item.get("trajectory_name", "")
        if os.path.basename(trajectory_path) == motion_name:
            return item
        if trajectory_name and motion_stem == trajectory_name:
            return item

    if len(trajectories) == 1:
        print(f"[WARN] No exact manifest match for {motion_name}; using the only trajectory entry.")
        return trajectories[0]

    raise ValueError(f"Could not find manifest trajectory entry matching motion file: {motion_name}")


def _get_skill_output_start_frame(manifest_entry: dict) -> int:
    for segment in manifest_entry.get("segments", []):
        if segment.get("mode") == "skill_execution":
            return int(segment["output_start_frame"])
    raise ValueError("Manifest entry has no skill_execution segment.")


def _rotate_xyz_by_yaw_deg(xyz: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    return np.array(
        [
            cos_yaw * float(xyz[0]) - sin_yaw * float(xyz[1]),
            sin_yaw * float(xyz[0]) + cos_yaw * float(xyz[1]),
            float(xyz[2]),
        ],
        dtype=np.float32,
    )


def _compute_manifest_motion_offset(
    manifest_entry: dict, motion_file: str, root_body_idx: int, use_manifest_terrain_pose: bool
) -> tuple[float, float, float]:
    data = np.load(motion_file)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    skill_output_start = _get_skill_output_start_frame(manifest_entry)
    if skill_output_start >= len(body_pos_w):
        raise ValueError(
            f"skill_execution.output_start_frame={skill_output_start} exceeds motion length {len(body_pos_w)}."
        )

    skill_anchor = manifest_entry["skill_anchor"]
    target_root = np.asarray(skill_anchor["root_translation"], dtype=np.float32)
    target_root_frame = "manifest skill world"

    if (
        use_manifest_terrain_pose
        and not bool(skill_anchor.get("fixed_world", False))
        and "terrain_world_pose" in manifest_entry
    ):
        terrain_pose = manifest_entry["terrain_world_pose"]
        terrain_translation = np.asarray(terrain_pose.get("translation", (0.0, 0.0, 0.0)), dtype=np.float32)
        terrain_yaw_deg = float(terrain_pose.get("yaw_deg", 0.0))
        target_root = terrain_translation + _rotate_xyz_by_yaw_deg(target_root, terrain_yaw_deg)
        target_root_frame = "manifest terrain world"

    observed_root = body_pos_w[skill_output_start, root_body_idx]
    offset = target_root - observed_root
    print(f"[INFO]: Target motion root ({target_root_frame}): {target_root.tolist()}")
    print(f"[INFO]: Applying manifest motion offset: {offset.tolist()}")
    return tuple(float(x) for x in offset)


def _add_manifest_terrain(
    env_cfg: ManagerBasedRLEnvCfg, terrain_file: str, manifest_entry: dict | None, use_manifest_pose: bool
):
    terrain_path = os.path.abspath(terrain_file)
    if not os.path.isfile(terrain_path):
        raise FileNotFoundError(f"Invalid terrain file path: {terrain_path}")

    terrain_pose = manifest_entry.get("terrain_world_pose", {}) if use_manifest_pose and manifest_entry is not None else {}
    translation = terrain_pose.get("translation", (0.0, 0.0, 0.0))
    yaw_deg = terrain_pose.get("yaw_deg", 0.0)
    terrain_pos = tuple(float(x) for x in translation)
    terrain_rot = _yaw_quat_wxyz(float(yaw_deg))

    env_cfg.scene.terrain_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Terrain",
        spawn=sim_utils.UsdFileCfg(
            usd_path=terrain_path,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=terrain_pos, rot=terrain_rot),
        collision_group=0,
    )
    _add_height_scan_terrain_target(env_cfg)
    print(f"[INFO]: Using local terrain USD: {terrain_path}")
    if use_manifest_pose:
        print(f"[INFO]: Terrain pose from manifest: pos={terrain_pos}, yaw_deg={float(yaw_deg)}")
    else:
        print(f"[INFO]: Terrain pose: pos={terrain_pos}, yaw_deg={float(yaw_deg)}")


def _use_fixed_motion_start(env_cfg: ManagerBasedRLEnvCfg, start_frame: int):
    env_cfg.commands.motion.fixed_start_frame = start_frame
    env_cfg.commands.motion.pose_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.commands.motion.velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    print(f"[INFO]: Fixed motion reset frame: {start_frame}")


def _disable_debug_visuals(env_cfg: ManagerBasedRLEnvCfg):
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.debug_vis = False
    if hasattr(env_cfg.scene, "contact_forces") and env_cfg.scene.contact_forces is not None:
        env_cfg.scene.contact_forces.debug_vis = False
    print("[INFO]: Disabled debug visualization for play.")


def _configure_motion(env_cfg: ManagerBasedRLEnvCfg, motion_file: str):
    motion_path = os.path.abspath(motion_file)
    if not os.path.isfile(motion_path):
        raise FileNotFoundError(f"Invalid motion file path: {motion_path}")
    print(f"[INFO]: Using motion file: {motion_path}")
    env_cfg.commands.motion.motion_file = motion_path
    if args_cli.motion_source_order == "g1_canonical":
        env_cfg.commands.motion.source_joint_names = G1_CANONICAL_JOINT_NAMES
        env_cfg.commands.motion.source_body_names = G1_CANONICAL_BODY_NAMES


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing if args_cli.env_spacing is not None else env_cfg.scene.env_spacing

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if args_cli.motion_file is not None:
        _configure_motion(env_cfg, args_cli.motion_file)

    if not isinstance(env_cfg.commands.motion.motion_file, str):
        raise ValueError("No motion file is configured. Pass --motion_file for local checkpoint evaluation.")

    manifest_entry = _load_manifest_entry(args_cli.manifest_file, env_cfg.commands.motion.motion_file)
    if manifest_entry is not None:
        env_cfg.commands.motion.motion_position_offset = _compute_manifest_motion_offset(
            manifest_entry,
            env_cfg.commands.motion.motion_file,
            args_cli.motion_root_body_idx,
            args_cli.terrain_use_manifest_pose,
        )
        print(f"[INFO]: Manifest trajectory: {manifest_entry.get('trajectory_name', '')}")

    if args_cli.terrain_file is not None:
        _add_manifest_terrain(env_cfg, args_cli.terrain_file, manifest_entry, args_cli.terrain_use_manifest_pose)

    if args_cli.fixed_start_frame >= 0:
        _use_fixed_motion_start(env_cfg, args_cli.fixed_start_frame)

    _disable_debug_visuals(env_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        policy_nn,
        normalizer=normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, "local", export_model_dir)
    # reset environment
    observations = env.get_observations()
    obs = observations[0] if isinstance(observations, tuple) else observations
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
