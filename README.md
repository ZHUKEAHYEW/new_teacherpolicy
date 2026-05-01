# 项目仓库使用说明

## 核心文件：

whole\_body\_tracking/scripts/rsl\_rl/train.py ——— 训练代码
whole\_body\_tracking/scripts/rsl\_rl/play.py ——— 可视化代码
whole\_body\_tracking/scripts/rsl\_rl/replay\_npz.py ——— 回放检查npz代码

## 安装流程:

1.激活isaaacsim环境

- conda activate env\_isaaclab

2.切换至仓库目录

- cd /home/user\_name/whole\_body\_tracking   （需要替换为你的项目路径）

3.安装该仓库

- python -m pip install -e source/whole\_body\_tracking

## 进行训练

### 注意事项

默认并行环境数在 `tracking_env_cfg.py` 中是 `16384`。如果显存不够，可通过 `--num_envs` 覆盖。

### 快速启动训练

```bash
cd /home/user_name/whole_body_tracking

python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --motion_file data/climb_15_z_scale_1.0_0000.npz \
  --manifest_file data/batch_manifest.json \
  --terrain_file climb_15/multi_boxes_z_scale_1.0/multi_boxes_z_scale_1.0.usd \
  --terrain_use_manifest_pose \
  --headless \
  --logger tensorboard \
  --run_name climb_15_local \
  --max_iterations 10000
```

训练日志默认保存到：

```
logs/rsl_rl/g1_flat/<timestamp>_<run_name>/
```

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

- `--motion_file`: 本地参考动作 `.npz`。
- `--manifest_file`: 用于读取 skill anchor、terrain pose 等对齐信息。
- `--terrain_file`: 本地 terrain USD。
- `--terrain_use_manifest_pose`: 使用 manifest 中的 terrain 世界位姿。
- `--fixed_start_frame -1`: 使用 adaptive random start-frame sampling。训练默认就是 `-1`。
- `--fixed_start_frame 0`: 每次 episode 从第 0 帧开始 （train.py默认是随机采样开始）
- `--num_envs`: 覆盖默认并行环境数。

追加参数：

- **从已有 checkpoint 继续训练**：

```bash
--resume True \
--load_run 2026-05-01_15-36-06_climb_15_high_jump \
--checkpoint model_10000.pt
```

温馨提醒： `--load_run` 是 `logs/rsl_rl/g1_flat/` 下的 run 文件夹名，`--checkpoint` 是该目录中的 checkpoint 文件名。

## Play

播放本地 checkpoint：

```bash
cd /home/user_name/whole_body_tracking

python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 \
  --num_envs=1 \
  --motion_file data/climb_15_z_scale_1.0_0000.npz \
  --manifest_file data/batch_manifest.json \
  --terrain_file climb_15/multi_boxes_z_scale_1.0/multi_boxes_z_scale_1.0.usd \
  --terrain_use_manifest_pose \
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
