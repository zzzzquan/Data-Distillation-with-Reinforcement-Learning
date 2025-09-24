import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np
import time

def train_student(student_net, distilled_data_loader, optimizer, criterion, k_steps, device):
    """
    训练学生网络
    
    Args:
        student_net: 学生网络模型
        distilled_data_loader: 蒸馏数据加载器
        optimizer: 优化器
        criterion: 损失函数
        k_steps: 训练步数
        device: 设备 (CPU/GPU)
    
    Returns:
        float: 平均损失
    """
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


def train_policy(policy_net, student_net, policy_optimizer, student_optimizer, 
                 state_data, state_labels, val_loader, criterion, 
                 LEARNING_RATE_STUDENT, K_STEPS, DISTILLATION_RATIO, MINI_BATCH_SIZE, device):
    """
    训练策略网络
    
    Args:
        policy_net: 策略网络模型
        student_net: 学生网络模型
        policy_optimizer: 策略网络优化器
        student_optimizer: 学生网络优化器
        state_data: 状态数据
        state_labels: 状态标签
        val_loader: 验证集数据加载器
        criterion: 损失函数
        LEARNING_RATE_STUDENT: 学生网络学习率
        K_STEPS: 训练步数
        DISTILLATION_RATIO: 蒸馏比例
        MINI_BATCH_SIZE: 批次大小
        device: 设备 (CPU/GPU)
    
    Returns:
        tuple: (reward, avg_loss, selected_indices, log_probs)
    """
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
    student_net = student_net.to(device)
    student_optimizer = optim.Adam(student_net.parameters(), lr=LEARNING_RATE_STUDENT)
    
    # 训练k步
    avg_loss = train_student(student_net, distilled_loader, student_optimizer, criterion, K_STEPS, device)
    
    # 第四步：计算奖励 (Calculate Reward)
    reward = evaluate_student(student_net, val_loader, device)
    
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
    
    return reward, avg_loss, selected_indices, log_probs