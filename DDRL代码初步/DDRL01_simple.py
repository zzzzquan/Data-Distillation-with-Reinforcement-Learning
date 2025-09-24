import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 简化版超参数定义
STATE_BATCH_SIZE = 64         # 状态批次大小（St）
K_STEPS = 5                   # 学生网络训练步数
NUM_EPISODES = 10             # 总训练轮数
LEARNING_RATE_POLICY = 1e-4   # 策略网络学习率
LEARNING_RATE_STUDENT = 1e-3  # 学生网络学习率
DISTILLATION_RATIO = 0.5      # 蒸馏比例（期望选择的样本比例）
MINI_BATCH_SIZE = 32          # 学生网络训练的mini-batch大小

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建数据目录
os.makedirs('./data', exist_ok=True)

# 简化的数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集（简化版）
print("正在加载CIFAR-10数据集...")
try:
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    print("数据集加载完成")
except Exception as e:
    print(f"数据集加载失败: {e}")
    # 创建模拟数据集用于测试
    print("创建模拟数据集...")
    trainset = torchvision.datasets.FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=transform)
    testset = torchvision.datasets.FakeData(size=500, image_size=(3, 32, 32), num_classes=10, transform=transform)

# 使用小部分数据进行快速测试
trainset = torch.utils.data.Subset(trainset, range(1000))
testset = torch.utils.data.Subset(testset, range(500))

# 划分训练集和验证集（80%训练，20%验证）
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=STATE_BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(testset)}")

# 简化版学生网络定义
class StudentNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 简化版策略网络定义
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)  # 输出每个样本的选取概率
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 使用sigmoid输出概率
        return x

# 在验证集上评估学生网络
def evaluate_student(student_net, val_loader):
    student_net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = student_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

# 训练学生网络
def train_student(student_net, distilled_data_loader, optimizer, criterion, k_steps):
    student_net.train()
    total_loss = 0.0
    num_batches = 0
    
    for step in range(k_steps):
        for i, data in enumerate(distilled_data_loader):
            if i >= 1:  # 每步只训练一个batch以加快速度
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = student_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
                
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

# 主训练循环
def main():
    print("开始简化版强化学习数据蒸馏训练...")
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
        # 第一步：获取状态 (Get State)
        try:
            state_data, state_labels = next(train_loader_iter)
        except StopIteration:
            # 如果迭代器耗尽，重新创建
            train_loader_iter = iter(train_loader)
            state_data, state_labels = next(train_loader_iter)
            
        state_data, state_labels = state_data.to(device), state_labels.to(device)
        
        # 第二步：生成动作 (Generate Action / Distill Data)
        # 策略网络输出每个样本的选取概率
        selection_probs = policy_net(state_data)
        
        # 为避免所有概率都接近0或1，添加一个小的epsilon
        selection_probs = torch.clamp(selection_probs, min=1e-3, max=1-1e-3)
        
        # 创建伯努利分布并采样
        m = Bernoulli(selection_probs)
        actions = m.sample()  # 0或1，表示是否选择该样本
        
        # 计算对数概率（用于REINFORCE算法）
        log_probs = m.log_prob(actions)
        
        # 获取被选中的样本索引
        selected_indices = (actions.squeeze() > 0.5).nonzero(as_tuple=True)[0]
        
        # 如果没有选中任何样本或选中太多样本，则调整选择
        if len(selected_indices) == 0:
            selected_indices = torch.randperm(len(state_data))[:max(1, int(len(state_data) * DISTILLATION_RATIO))]
        elif len(selected_indices) > len(state_data) * 0.8:  # 如果选中太多
            perm = torch.randperm(len(selected_indices))
            selected_indices = selected_indices[perm[:int(len(state_data) * DISTILLATION_RATIO)]]
        
        # 根据动作获取蒸馏数据批次
        distilled_data = state_data[selected_indices]
        distilled_labels = state_labels[selected_indices]
        
        # 记录蒸馏批次大小
        distilled_sizes.append(len(selected_indices))
        
        # 创建蒸馏数据加载器
        if len(distilled_data) > 0:
            distilled_dataset = torch.utils.data.TensorDataset(distilled_data, distilled_labels)
            distilled_loader = torch.utils.data.DataLoader(distilled_dataset, batch_size=min(MINI_BATCH_SIZE, len(distilled_data)), shuffle=True)
        else:
            # 如果没有蒸馏数据，使用原始数据
            distilled_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(state_data, state_labels), 
                                                          batch_size=MINI_BATCH_SIZE, shuffle=True)
        
        # 第三步：训练学生网络 (Train Student Network)
        # 重新初始化学生网络
        student_net = StudentNetwork().to(device)
        student_optimizer = optim.Adam(student_net.parameters(), lr=LEARNING_RATE_STUDENT)
        
        # 训练k步
        avg_loss = train_student(student_net, distilled_loader, student_optimizer, criterion, K_STEPS)
        
        # 第四步：计算奖励 (Calculate Reward)
        reward = evaluate_student(student_net, val_loader)
        
        # 第五步：更新策略网络 (Update Policy Network)
        # 使用REINFORCE算法更新策略网络
        policy_optimizer.zero_grad()
        
        # 损失函数：L = -Rt * logπϕ(at|St)
        policy_loss = -torch.mean(torch.mean(log_probs, dim=1) * reward)
        
        # 添加熵正则化项，鼓励探索
        entropy = torch.mean(torch.sum(m.entropy(), dim=1))
        policy_loss -= 0.01 * entropy  # 熵正则化系数
        
        # 反向传播和优化
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        policy_optimizer.step()
        
        # 记录奖励
        reward_history.append(reward)
        
        # 打印日志
        if (episode + 1) % 2 == 0:
            avg_reward = np.mean(reward_history[-2:]) if len(reward_history) >= 2 else np.mean(reward_history)
            avg_distilled_size = np.mean(distilled_sizes[-2:]) if len(distilled_sizes) >= 2 else np.mean(distilled_sizes)
            print(f'轮数 [{episode+1}/{NUM_EPISODES}], 奖励: {reward:.2f}%, 平均奖励: {avg_reward:.2f}%, '
                  f'蒸馏批次大小: {len(selected_indices)} (平均: {avg_distilled_size:.1f}), '
                  f'学生网络损失: {avg_loss:.4f}')
    
    print("=" * 50)
    print("简化版训练完成!")
    
    return policy_net, reward_history

if __name__ == "__main__":
    policy_net, reward_history = main()