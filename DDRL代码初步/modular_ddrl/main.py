import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from torch.distributions import Bernoulli

# 导入自定义模块
from models.student_network import StudentNetwork
from models.policy_network import PolicyNetwork
from data.data_processing import get_data_transforms, load_cifar10_dataset, create_data_loaders
from utils.evaluation import evaluate_student, evaluate_policy
from utils.training import train_student, train_policy

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 超参数定义
STATE_BATCH_SIZE = 256        # 状态批次大小（St）
K_STEPS = 20                  # 学生网络训练步数
NUM_EPISODES = 50             # 总训练轮数
LEARNING_RATE_POLICY = 1e-4   # 策略网络学习率
LEARNING_RATE_STUDENT = 1e-3  # 学生网络学习率
DISTILLATION_RATIO = 0.5      # 蒸馏比例（期望选择的样本比例）
MINI_BATCH_SIZE = 64          # 学生网络训练的mini-batch大小

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # 获取数据预处理变换
    transform_train, transform_test = get_data_transforms()
    
    # 加载数据集
    trainset, testset = load_cifar10_dataset(transform_train, transform_test)
    
    # 划分训练集和验证集（90%训练，10%验证）
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, testset, STATE_BATCH_SIZE)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(testset)}")
    
    print("开始强化学习数据蒸馏训练...")
    print("=" * 50)
    
    # 初始化网络
    policy_net = PolicyNetwork().to(device)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE_POLICY)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 记录奖励历史
    reward_history = []
    distilled_sizes = []
    
    # 创建一个固定的训练数据加载器迭代器
    train_loader_iter = iter(train_loader)
    
    for episode in range(NUM_EPISODES):
        start_time = time.time()
        
        # 第一步：获取状态 (Get State)
        try:
            state_data, state_labels = next(train_loader_iter)
        except StopIteration:
            # 如果迭代器耗尽，重新创建
            train_loader_iter = iter(train_loader)
            state_data, state_labels = next(train_loader_iter)
            
        state_data, state_labels = state_data.to(device), state_labels.to(device)
        
        # 训练策略网络
        student_net = StudentNetwork().to(device)
        reward, avg_loss, selected_indices, log_probs = train_policy(
            policy_net, student_net, policy_optimizer, None,
            state_data, state_labels, val_loader, criterion,
            LEARNING_RATE_STUDENT, K_STEPS, DISTILLATION_RATIO, MINI_BATCH_SIZE, device
        )
        
        # 记录奖励和蒸馏批次大小
        reward_history.append(reward)
        distilled_sizes.append(len(selected_indices))
        
        # 打印日志
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
            avg_distilled_size = np.mean(distilled_sizes[-10:]) if len(distilled_sizes) >= 10 else np.mean(distilled_sizes)
            elapsed_time = time.time() - start_time
            print(f'轮数 [{episode+1}/{NUM_EPISODES}], 奖励: {reward:.2f}%, 平均奖励: {avg_reward:.2f}%, '
                  f'蒸馏批次大小: {len(selected_indices)} (平均: {avg_distilled_size:.1f}), '
                  f'学生网络损失: {avg_loss:.4f}, 耗时: {elapsed_time:.2f}秒')
    
    print("=" * 50)
    print("训练完成!")
    
    # 评估最终策略
    print("评估最终策略...")
    evaluate_policy(policy_net, test_loader, device, DISTILLATION_RATIO)
    
    return policy_net, reward_history

if __name__ == "__main__":
    policy_net, reward_history = main()