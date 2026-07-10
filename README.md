# TeacherPolicyTraining

## 核心脚本介绍：

whole\_body\_tracking/scripts/rsl\_rl/train.py ——— Training
whole\_body\_tracking/scripts/rsl\_rl/play.py ——— Play Policy To test
whole\_body\_tracking/scripts/rsl\_rl/replay\_npz.py ——— Replay NPZ To Check out

## install:

1.激活isaacsim环境

- conda activate env\_isaaclab

2.切换至仓库目录

- cd /home/user\_name/whole\_body\_tracking   （需要替换为你的项目路径）

3.安装该仓库

- python -m pip install -e source/whole\_body\_tracking

## Start To Train

### 注意事项

默认并行环境数在 `tracking_env_cfg.py` 中是 `16384`。如果显存不够，可通过 `--num_envs` 覆盖
日志默认保存路径：
logs/rsl_rl/g1_flat/<timestamp>_<run_name>/

### Quick Start Training

```bash
cd /home/user_name/whole_body_tracking

python scripts/rsl_rl/train.py --multi climb_15_z_scale_1.0
```
or
```bash
python scripts/rsl_rl/train.py --simple climb_15_z_scale_1.0
```
默认参数：
`task=Tracking-Flat-G1-v0`（G1Task）
`--max_iterations 20000`（20k iterations）
`--headless` （无头）
`--logger tensorboard` （数据可视化）



### 查看 TensorBoard 曲线

训练命令中使用了：

```bash
--logger tensorboard
```

训练过程中的 reward、loss、learning rate、KL、episode length 等日志会写入：

```text
logs/rsl_rl/g1_flat/
```

启动 TensorBoard：

```bash
cd /home/user_name/whole_body_tracking
tensorboard --logdir logs/rsl_rl/g1_flat
```

启动后在浏览器打开终端提示的地址，通常是：

```text
http://localhost:6006
```

如果遇到：

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

在 Isaac Lab 环境中执行：

```bash
pip install "setuptools<81" --force-reinstall
```

然后重新启动 TensorBoard。

关键参数解析：

- `--dataset_dir`: 结构化数据集目录，或包含多个结构化数据集的根目录。
- `--dataset_name`: 当 `--dataset_dir` 指向数据集根目录时，选择其中一个数据集。
- `--dataset_motion_index`: 只读取排序后的第 N 条 `.npz` 轨迹，`--simple` 会默认等价于它的 `0`。
- `--motion_file`: 本地参考动作 `.npz`。
- `--motion_files`: 一次传入多个本地参考动作 `.npz`，训练时每个并行环境会随机选择一条轨迹。
- `--motion_dir`: 传入一个目录，训练时会自动读取目录下所有 `.npz` 轨迹。
- `--manifest_file`: 用于读取 skill anchor、terrain pose 等对齐信息。
- `--terrain_file`: 本地 terrain USD。
- `--terrain_use_manifest_pose`: 使用 manifest 中的 terrain 世界位姿。
- `--num_envs`: 覆盖默认并行环境数。

追加参数：

- **从已有 checkpoint 继续训练**：

```bash
--resume True \
--load_run 2026-05-01_15-36-06_climb_15_high_jump \
--checkpoint model_10000.pt
```

温馨提醒： `--load_run` 是 `logs/rsl_rl/g1_flat/` 下的 run 文件夹名，`--checkpoint` 是该目录中的 checkpoint 文件名。

### 结构化数据集训练

现在也支持直接读取如下结构的数据集：

```text
data/climb_15_z_scale_1.0/
├── terrain/
│   ├── multi_boxes_z_scale_1.0.usd
│   └── configuration/
└── tracking/
    ├── batch_manifest.json
    ├── climb_15_z_scale_1.0_0000.npz
    └── ...
```

快捷入口：

```bash
python scripts/rsl_rl/train.py --multi climb_15_z_scale_1.0
python scripts/rsl_rl/train.py --simple climb_15_z_scale_1.0
```

其中 `--multi` 会读取 `data/轨迹文件夹名/tracking/` 下全部 `.npz`，`--simple` 只取排序后的第一个 `.npz`。这两个入口默认补上 `--max_iterations 20000`、`--headless`、`--logger tensorboard`，并把任务固定到 `Tracking-Flat-G1-v0`。

如果不使用快捷入口，也可以显式指定数据集目录：

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --dataset_dir data \
  --dataset_name climb_15_z_scale_1.0 \
  --headless \
  --logger tensorboard \
  --run_name climb_15_dataset \
  --max_iterations 10000
```

也可以直接把 `--dataset_dir` 指向某一个数据集目录：

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --dataset_dir data/climb_15_z_scale_1.0 \
  --headless \
  --logger tensorboard \
  --run_name climb_15_dataset \
  --max_iterations 10000
```

结构化数据集会自动推导：

- `tracking/*.npz` 作为 motion 文件。
- `tracking/batch_manifest.json` 作为 manifest。
- `terrain/*.usd` 作为 terrain。
- 自动启用 manifest 中的 `terrain_world_pose` 对齐。

如果只想训练其中一条轨迹，也可以直接加 `--dataset_motion_index 0`，效果和 `--simple` 一致。

## Play

播放本地 checkpoint：

```bash
cd /home/user_name/whole_body_tracking

python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 \
  --num_envs=1 \
  --dataset_dir data \
  --dataset_name climb_15_z_scale_1.0 \
  --dataset_motion_index 0 \
  --load_run 2026-05-01_15-36-06_climb_15_high_jump \
  --checkpoint model_10000.pt
```

播放结构化数据集中的某条轨迹：

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 \
  --num_envs=1 \
  --dataset_dir data \
  --dataset_name climb_15_z_scale_1.0 \
  --dataset_motion_index 0 \
  --load_run 2026-05-01_15-36-06_climb_15_high_jump \
  --checkpoint model_10000.pt
```

## 如需要录制视频

\--video \
\--video\_length 5000

- 视频路径：
  `logs/rsl_rl/g1_flat/<load_run>/videos/play/`

## 代码结构

```
whole_body_tracking/
│
├── scripts/rsl_rl/
│   ├── train.py                 # 训练脚本
│   ├── play.py                  # 可视化/播放脚本
│   ├── replay_local_npz.py      # NPZ 回放检查脚本
│   └── cli_args.py              # 命令行参数定义
│
├── source/whole_body_tracking/whole_body_tracking/
│   │
│   ├── tasks/tracking/
│   │   ├── tracking_env_cfg.py              # 环境配置
│   │   ├── mdp/                             # 马尔可夫决策过程模块
│   │   │   ├── commands.py                  # 命令生成
│   │   │   ├── observations.py              # 观测空间
│   │   │   ├── rewards.py                   # 奖励函数
│   │   │   ├── terminations.py              # 终止条件
│   │   │   └── events.py                    # 随机事件
│   │   └── config/g1/
│   │       ├── __init__.py
│   │       ├── flat_env_cfg.py              # 平地环境配置
│   │       └── agents/
│   │           └── rsl_rl_ppo_cfg.py        # PPO 算法配置
│   │
│   ├── robots/
│   │   └── g1.py                            # G1 机器人定义
│   │
│   └── utils/
│       ├── exporter.py                      # 导出工具
│       └── my_on_policy_runner.py           # 自定义 on-policy 执行器
```
