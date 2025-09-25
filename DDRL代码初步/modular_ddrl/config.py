import torch

# 超参数定义
STATE_BATCH_SIZE = 256        # 状态批次大小（St）
K_STEPS = 20                  # 学生网络训练步数
NUM_EPISODES = 50             # 总训练轮数
LEARNING_RATE_POLICY = 1e-4   # 策略网络学习率
LEARNING_RATE_STUDENT = 1e-3  # 学生网络学习率
DISTILLATION_RATIO = 0.5      # 蒸馏比例（期望选择的样本比例）
MINI_BATCH_SIZE = 64          # 学生网络训练的mini-batch大小
IPC = 10                      # 每个类别的图像数量 (Images Per Class)

# 检查GPU可用性
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")