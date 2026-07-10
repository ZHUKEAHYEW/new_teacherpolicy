# Training Run Log

## Round 1 - climb_08_z_scale_1.0 multi-trajectory baseline

- Date: 2026-07-09
- Dataset: `data/climb_08_z_scale_1.0`
- Motions: 15 `.npz` files from `tracking/*.npz`
- Manifest: `data/climb_08_z_scale_1.0/tracking/batch_manifest.json`
- Terrain: `data/climb_08_z_scale_1.0/terrain/multi_boxes_z_scale_1.0.usd`
- Initial `num_envs`: 8096
- Goal: verify multi-trajectory skill training starts cleanly and does not diverge or run out of memory.
- Notes before launch:
  - Dataset path resolution succeeded.
  - Manifest trajectory count matches motion file count.
  - GPU detected: NVIDIA GeForce RTX 4090 Laptop GPU, 16GB VRAM.
  - 8096 envs may be high for 16GB with USD terrain and height scan; first action is to try it as requested, then reduce if memory or instability appears.

Command:

```bash
conda run -n env_isaaclab python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --dataset_dir data/climb_08_z_scale_1.0 \
  --num_envs 8096 \
  --headless \
  --logger tensorboard \
  --run_name climb_08_multi_8096 \
  --max_iterations 10000
```

Result:

- Failed before Isaac/Training startup.
- Error: `DirectoryNotACondaEnvironmentError` for `/home/zhukeahyew/miniconda3/envs/env_isaaclab`.
- Diagnosis: available conda env is named `isaaclab`; `env_isaaclab` exists as an empty directory but is not a valid conda environment.
- No training parameter changes yet.

## Round 2 - retry with valid conda environment

- Change from Round 1: use `conda run -n isaaclab` instead of `conda run -n env_isaaclab`.
- Training parameters unchanged:
  - `num_envs`: 8096
  - `max_iterations`: 10000
  - dataset: `data/climb_08_z_scale_1.0`

Command:

```bash
conda run -n isaaclab python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --dataset_dir data/climb_08_z_scale_1.0 \
  --num_envs 8096 \
  --headless \
  --logger tensorboard \
  --run_name climb_08_multi_8096 \
  --max_iterations 10000
```

## Round 5 - add shortcut launchers for multi/simple training

- Change: add `--multi DATASET` and `--simple DATASET` to `scripts/rsl_rl/train.py`.
- Behavior:
  - `--multi climb_08_z_scale_1.0` resolves to `data/climb_08_z_scale_1.0/` and loads all `tracking/*.npz`.
  - `--simple climb_08_z_scale_1.0` resolves to the same dataset but uses the first sorted `tracking/*.npz`.
- Default values applied by the shortcut:
  - `--task Tracking-Flat-G1-v0`
  - `--max_iterations 20000`
  - `--headless`
  - `--logger tensorboard`
- Existing lower-level arguments are still honored when explicitly provided.
- Related defaults:
  - `env_spacing` now defaults to `10.0` in both `train.py` and the task config.

## Round 4 - user-confirmed live training

- Status: starting now.
- Dataset: `data/climb_08_z_scale_1.0`
- Motions: 15 trajectories.
- Initial `num_envs`: 8096.
- Run name: `climb_08_multi_8096`
- Goal: keep the run stable, stop and tune parameters if it blows up.

Command:

```bash
conda run -n env_isaaclab python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-v0 \
    --dataset_dir data/climb_08_z_scale_1.0 \
    --num_envs 8096 \
    --headless \
    --logger tensorboard \
    --run_name climb_08_multi_8096 \
    --max_iterations 10000
```

## Round 4 status update

- Status: running.
- Latest observed step: `22`.
- Latest metrics:
  - `Train/mean_reward`: `1.042896032333374`
  - `Train/mean_episode_length`: `34.310001373291016`
  - `Perf/total_fps`: `15753.0`
  - `Perf/collection time`: `11.96917724609375`
  - `Perf/learning_time`: `0.36522626876831055`
  - `Loss/value_function`: `0.018127374351024628`
  - `Loss/surrogate`: `-0.012928947806358337`
  - `Loss/entropy`: `37.1058235168457`
- GPU usage at this point:
  - `10318 MiB / 16376 MiB`
  - utilization about `88%`
- Interpretation:
  - Training is alive and producing scalars.
  - No immediate OOM or divergence signal yet.
  - `num_envs=8096` is aggressive but currently stable enough to continue monitoring.

Result:

- Failed during Isaac environment creation, before PPO training iterations.
- Dataset parsing succeeded and all 15 motion files were loaded into the config.
- Manifest alignment offsets were computed for all 15 trajectories.
- Terrain USD was selected correctly.
- Error:
  - `CUDA error: no CUDA-capable device is detected`
  - `RuntimeError: No CUDA GPUs are available`
- Diagnosis:
  - The command ran inside the tool sandbox. Isaac/PhysX could not create a CUDA context from inside the sandbox, even though `nvidia-smi` sees the RTX 4090 outside the training process.
  - This is an execution environment issue, not a training instability or `num_envs` failure.

## Round 3 - run outside sandbox so Isaac/PhysX can access CUDA

- Change from Round 2: run the same training command with escalated execution outside the sandbox.
- Training parameters unchanged:
  - `num_envs`: 8096
  - `max_iterations`: 10000
  - dataset: `data/climb_08_z_scale_1.0`

Command:

```bash
conda run -n isaaclab python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --dataset_dir data/climb_08_z_scale_1.0 \
  --num_envs 8096 \
  --headless \
  --logger tensorboard \
  --run_name climb_08_multi_8096 \
  --max_iterations 10000
```
