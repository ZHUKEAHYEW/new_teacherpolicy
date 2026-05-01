import argparse
import json
import pathlib
import sys
import time
import numpy as np
import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ===== npz 导出时的 canonical joint 顺序 =====
NPZ_JOINT_NAMES = [
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


def reorder_joint_array(joint_arr: np.ndarray, src_joint_names: list[str], dst_joint_names: list[str]) -> np.ndarray:
    src_index = {name: i for i, name in enumerate(src_joint_names)}
    out = np.zeros((joint_arr.shape[0], len(dst_joint_names)), dtype=joint_arr.dtype)

    missing = []
    for j, name in enumerate(dst_joint_names):
        if name not in src_index:
            missing.append(name)
            continue
        out[:, j] = joint_arr[:, src_index[name]]

    if missing:
        raise KeyError(f"这些目标关节在 npz 源顺序中找不到: {missing}")

    return out


def quat_wxyz_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    half = 0.5 * yaw_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def rotate_xyz_by_yaw_deg(xyz: np.ndarray, yaw_deg: float) -> np.ndarray:
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


def load_manifest_entry(manifest_path: pathlib.Path, motion_file: pathlib.Path):
    if not manifest_path.exists():
        print(f"[WARN] Manifest not found: {manifest_path}")
        return None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectories = payload.get("trajectories", [])

    motion_name = motion_file.name
    motion_path_str = str(motion_file)

    for item in trajectories:
        traj_path = item.get("trajectory_path", "")
        traj_name = item.get("trajectory_name", "")

        if pathlib.Path(traj_path).name == motion_name:
            return item
        if traj_name and motion_name.startswith(traj_name):
            return item
        if traj_path and motion_path_str.endswith(traj_path):
            return item

    print(f"[WARN] Could not find matching trajectory entry in manifest for {motion_name}")
    return None


def compute_replay_root_offset(
    manifest_entry, body_pos_w_np: np.ndarray, root_body_idx: int, use_manifest_terrain_pose: bool
) -> np.ndarray:
    """
    用 manifest 里的 skill_anchor + skill_execution.output_start_frame
    把 npz 轨迹整体平移到 manifest 世界坐标系。
    """
    if manifest_entry is None:
        return np.zeros(3, dtype=np.float32)

    if "skill_anchor" not in manifest_entry or "segments" not in manifest_entry:
        return np.zeros(3, dtype=np.float32)

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
        target_root = terrain_translation + rotate_xyz_by_yaw_deg(target_root, terrain_yaw_deg)
        target_root_frame = "manifest terrain world"

    skill_output_start = None
    for seg in manifest_entry["segments"]:
        if seg.get("mode") == "skill_execution":
            skill_output_start = int(seg["output_start_frame"])
            break

    if skill_output_start is None:
        print("[WARN] No skill_execution segment found in manifest; robot root offset not applied.")
        return np.zeros(3, dtype=np.float32)

    if skill_output_start >= len(body_pos_w_np):
        print("[WARN] skill_execution.output_start_frame exceeds trajectory length; robot root offset not applied.")
        return np.zeros(3, dtype=np.float32)

    observed_root = np.asarray(body_pos_w_np[skill_output_start, root_body_idx], dtype=np.float32)
    offset = target_root - observed_root

    print(f"[INFO] skill_output_start_frame = {skill_output_start}")
    print(f"[INFO] target replay root ({target_root_frame}) = {target_root.tolist()}")
    print(f"[INFO] observed replay root at skill start = {observed_root.tolist()}")
    print(f"[INFO] applying replay root offset = {offset.tolist()}")

    return offset


def set_xformable_prim_pose(stage, prim_path: str, translation_xyz: np.ndarray, quat_wxyz: np.ndarray):
    """
    直接改 USD prim 的位姿。
    """
    from pxr import UsdGeom, Gf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()

    # 清掉已有 xform op，避免叠加乱掉
    if ops:
        xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()

    translate_op.Set(Gf.Vec3d(float(translation_xyz[0]), float(translation_xyz[1]), float(translation_xyz[2])))
    orient_op.Set(
        Gf.Quatd(
            float(quat_wxyz[0]),
            Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
        )
    )


parser = argparse.ArgumentParser(description="Replay local npz trajectory in Isaac Sim.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. Tracking-Flat-G1-v0")
parser.add_argument("--motion_file", type=str, required=True, help="Local npz motion file")
parser.add_argument("--manifest_file", type=str, default="", help="Optional batch_manifest.json path")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 env")
parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS")
parser.add_argument("--loop", action="store_true", help="Loop playback")
parser.add_argument("--root_body_idx", type=int, default=0, help="Which body index in body_pos_w/body_quat_w to use as root")
parser.add_argument("--z_offset", type=float, default=0.0, help="Additional z offset for replay")
parser.add_argument("--terrain_file", type=str, default="", help="Optional local terrain USD file")
parser.add_argument(
    "--terrain_use_manifest_pose",
    action="store_true",
    help="Spawn --terrain_file at manifest terrain_world_pose instead of the origin.",
)
parser.add_argument("--apply_manifest_terrain_pose", action="store_true", help="Apply terrain pose from manifest")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


def add_terrain_asset(env_cfg: ManagerBasedRLEnvCfg, terrain_file: str, manifest_entry, use_manifest_pose: bool):
    terrain_path = pathlib.Path(terrain_file).resolve()
    if not terrain_path.exists():
        raise FileNotFoundError(f"Terrain file not found: {terrain_path}")

    terrain_translation = np.zeros(3, dtype=np.float32)
    terrain_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if use_manifest_pose and manifest_entry is not None and "terrain_world_pose" in manifest_entry:
        terrain_pose = manifest_entry["terrain_world_pose"]
        terrain_translation = np.asarray(terrain_pose.get("translation", terrain_translation), dtype=np.float32)
        terrain_quat = quat_wxyz_from_yaw_deg(float(terrain_pose.get("yaw_deg", 0.0)))
    elif manifest_entry is not None and "terrain_world_pose" in manifest_entry:
        print("[INFO] Terrain USD is loaded at the origin. Pass --terrain_use_manifest_pose to use manifest pose.")

    env_cfg.scene.terrain_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Terrain",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(terrain_path),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(x) for x in terrain_translation),
            rot=tuple(float(x) for x in terrain_quat),
        ),
        collision_group=0,
    )
    print(f"[INFO] Loaded terrain USD: {terrain_path}")
    print(f"[INFO] Terrain init pose: pos={terrain_translation.tolist()}, quat_wxyz={terrain_quat.tolist()}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    motion_path = pathlib.Path(args_cli.motion_file).resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    if args_cli.manifest_file:
        manifest_path = pathlib.Path(args_cli.manifest_file).resolve()
    else:
        manifest_path = motion_path.parent / "batch_manifest.json"

    manifest_entry = load_manifest_entry(manifest_path, motion_path)

    data = np.load(motion_path, allow_pickle=True)
    print("NPZ keys:", data.files)

    joint_pos_np = np.asarray(data["joint_pos"], dtype=np.float32)
    joint_vel_np = np.asarray(data["joint_vel"], dtype=np.float32)
    body_pos_w_np = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w_np = np.asarray(data["body_quat_w"], dtype=np.float32)
    body_lin_vel_w_np = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
    body_ang_vel_w_np = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

    num_frames = min(
        len(joint_pos_np),
        len(joint_vel_np),
        len(body_pos_w_np),
        len(body_quat_w_np),
        len(body_lin_vel_w_np),
        len(body_ang_vel_w_np),
    )

    print(f"[INFO] joint_pos shape     : {joint_pos_np.shape}")
    print(f"[INFO] joint_vel shape     : {joint_vel_np.shape}")
    print(f"[INFO] body_pos_w shape    : {body_pos_w_np.shape}")
    print(f"[INFO] body_quat_w shape   : {body_quat_w_np.shape}")
    print(f"[INFO] root_body_idx       : {args_cli.root_body_idx}")
    print(f"[INFO] num_frames          : {num_frames}")

    if manifest_entry is not None:
        print(f"[INFO] Manifest trajectory_name : {manifest_entry.get('trajectory_name', '')}")
        print(f"[INFO] Manifest terrain_path    : {manifest_entry.get('terrain_path', '')}")
        if "terrain_world_pose" in manifest_entry:
            print(f"[INFO] Manifest terrain pose    : {manifest_entry['terrain_world_pose']}")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = str(motion_path)
    if args_cli.terrain_file:
        add_terrain_asset(env_cfg, args_cli.terrain_file, manifest_entry, args_cli.terrain_use_manifest_pose)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if not args_cli.headless else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env.reset()

    robot = env.unwrapped.scene["robot"]
    device = robot.device

    robot_joint_names = list(robot.data.joint_names)
    print("[INFO] Robot joint order:")
    for i, name in enumerate(robot_joint_names):
        print(f"  {i:02d}: {name}")

    joint_pos_np = reorder_joint_array(joint_pos_np, NPZ_JOINT_NAMES, robot_joint_names)
    joint_vel_np = reorder_joint_array(joint_vel_np, NPZ_JOINT_NAMES, robot_joint_names)
    print("[INFO] Reordered joint_pos/joint_vel from NPZ canonical order to IsaacLab robot order.")

    # ===== 关键：把整条机器人轨迹对齐到 manifest 世界坐标 =====
    replay_root_offset = compute_replay_root_offset(
        manifest_entry,
        body_pos_w_np,
        args_cli.root_body_idx,
        args_cli.terrain_use_manifest_pose or args_cli.apply_manifest_terrain_pose,
    )

    # ===== 关键：直接改 obstacles prim，而不是 scene entity =====
    if args_cli.apply_manifest_terrain_pose and manifest_entry is not None and "terrain_world_pose" in manifest_entry:
        terrain_pose = manifest_entry["terrain_world_pose"]
        terrain_translation = np.asarray(terrain_pose["translation"], dtype=np.float32)
        terrain_quat = quat_wxyz_from_yaw_deg(float(terrain_pose["yaw_deg"]))

        stage = omni.usd.get_context().get_stage()

        candidate_prim_paths = ["/World/envs/env_0/terrain", "/World/envs/env_0/obstacles"]
        if args_cli.terrain_use_manifest_pose:
            candidate_prim_paths.insert(0, "/World/envs/env_0/Terrain")

        applied = False
        for prim_path in candidate_prim_paths:
            try:
                set_xformable_prim_pose(stage, prim_path, terrain_translation, terrain_quat)
                print(f"[INFO] Applied manifest terrain pose to prim '{prim_path}': {terrain_pose}")
                applied = True
                break
            except Exception:
                pass

        if not applied:
            print("[WARN] Could not find terrain/obstacles prim to apply manifest terrain pose.")
    elif args_cli.apply_manifest_terrain_pose:
        print("[WARN] --apply_manifest_terrain_pose enabled, but manifest entry or terrain_world_pose missing.")

    replay_fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else float(args_cli.fps)
    sleep_dt = 1.0 / float(args_cli.fps)

    frame_idx = 0
    print("[INFO] Start replay...")

    while simulation_app.is_running():
        root_pos = body_pos_w_np[frame_idx, args_cli.root_body_idx].copy()
        root_quat = body_quat_w_np[frame_idx, args_cli.root_body_idx].copy()
        root_lin_vel = body_lin_vel_w_np[frame_idx, args_cli.root_body_idx].copy()
        root_ang_vel = body_ang_vel_w_np[frame_idx, args_cli.root_body_idx].copy()

        # 把整条轨迹移到 manifest 世界系
        root_pos += replay_root_offset
        root_pos[2] += args_cli.z_offset

        jp = torch.tensor(joint_pos_np[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        jv = torch.tensor(joint_vel_np[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)

        rp = torch.tensor(root_pos, dtype=torch.float32, device=device).unsqueeze(0)
        rq = torch.tensor(root_quat, dtype=torch.float32, device=device).unsqueeze(0)
        rlv = torch.tensor(root_lin_vel, dtype=torch.float32, device=device).unsqueeze(0)
        rav = torch.tensor(root_ang_vel, dtype=torch.float32, device=device).unsqueeze(0)

        root_pose = torch.cat([rp, rq], dim=-1)
        root_vel = torch.cat([rlv, rav], dim=-1)

        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_vel)
        robot.write_joint_state_to_sim(jp, jv)

        env.unwrapped.sim.step(render=True)

        if frame_idx % 30 == 0:
            print(
                f"[FRAME {frame_idx:04d}] "
                f"root_pos=({rp[0,0].item(): .3f}, {rp[0,1].item(): .3f}, {rp[0,2].item(): .3f}) "
                f"replay_fps={replay_fps}"
            )

        frame_idx += 1
        if frame_idx >= num_frames:
            if args_cli.loop:
                frame_idx = 0
            else:
                break

        time.sleep(sleep_dt)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
