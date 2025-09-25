# Data Distillation with Reinforcement Learning

## 使用说明

### 基本用法

```bash
python main.py --dataset=CIFAR10 --ipc=10
```

### IPC=50 (P=5) 的命令示例

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=CIFAR10 --ipc=50 --num_episodes=50 --state_batch_size=256 --k_steps=20 --lr_policy=1e-4 --lr_student=1e-3 --distillation_ratio=0.5 --mini_batch_size=64 --cuda_device=0 --seed=42 --num_intervals=5 --root_log_dir=logged_files
```

### 参数说明

- `--dataset`: 数据集名称 (默认: CIFAR10)
- `--ipc`: 每个类别的图像数量 (Images Per Class) (默认: 10)
- `--num_episodes`: 总训练轮数 (默认: 50)
- `--state_batch_size`: 状态批次大小 (默认: 256)
- `--k_steps`: 学生网络训练步数 (默认: 20)
- `--lr_policy`: 策略网络学习率 (默认: 1e-4)
- `--lr_student`: 学生网络学习率 (默认: 1e-3)
- `--distillation_ratio`: 蒸馏比例 (默认: 0.5)
- `--mini_batch_size`: 学生网络训练的mini-batch大小 (默认: 64)
- `--cuda_device`: CUDA设备ID (默认: 0)
- `--seed`: 随机种子 (默认: 42)
- `--num_intervals`: 评估间隔 (默认: 5)
- `--root_log_dir`: 日志文件目录 (默认: logged_files)

### 示例命令

#### IPC=10 (默认)
```bash
python main.py
```

#### IPC=50
```bash
python main.py --ipc=50
```

#### IPC=50 with custom parameters
```bash
python main.py --ipc=50 --num_episodes=100 --lr_policy=5e-5 --lr_student=5e-4
```