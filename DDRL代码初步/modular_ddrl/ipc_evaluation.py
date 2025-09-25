import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Bernoulli

# 导入自定义模块
from models.student_network import StudentNetwork
from models.policy_network import PolicyNetwork
from data.data_processing import get_data_transforms, load_cifar10_dataset, create_data_loaders, select_data_by_ipc
from utils.evaluation import evaluate_student, evaluate_policy
from utils.training import train_student, train_policy

# 导入配置参数
from config import STATE_BATCH_SIZE, K_STEPS, NUM_EPISODES, LEARNING_RATE_POLICY, LEARNING_RATE_STUDENT
from config import DISTILLATION_RATIO, MINI_BATCH_SIZE, IPC, DEVICE

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 使用配置中的设备
device = DEVICE
print(f"Using device: {device}")

def ipc_evaluation():
    """
    按IPC=10标准浓缩数据集并在验证集上测试准确度
    """
    print("开始IPC=10数据浓缩和验证评估...")
    print("=" * 50)
    
    # 获取数据预处理变换
    transform_train, transform_test = get_data_transforms()
    
    # 加载数据集
    trainset, testset = load_cifar10_dataset(transform_train, transform_test)
    
    # 按IPC=10标准选择训练数据
    print(f"按IPC={IPC}标准选择训练数据...")
    condensed_trainset = select_data_by_ipc(trainset, ipc=IPC)
    print(f"浓缩数据集大小: {len(condensed_trainset)} (每类{IPC}张图像)")
    
    # 划分训练集和验证集（90%训练，10%验证）
    train_size = int(0.9 * len(condensed_trainset))
    val_size = len(condensed_trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(condensed_trainset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, testset, STATE_BATCH_SIZE)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(testset)}")
    
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
    
    print("开始强化学习数据蒸馏训练...")
    for episode in range(NUM_EPISODES):
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
            print(f'轮数 [{episode+1}/{NUM_EPISODES}], 奖励: {reward:.2f}%, 平均奖励: {avg_reward:.2f}%, '
                  f'蒸馏批次大小: {len(selected_indices)} (平均: {avg_distilled_size:.1f}), '
                  f'学生网络损失: {avg_loss:.4f}')
    
    print("=" * 50)
    print("训练完成!")
    
    # 评估最终策略
    print("评估最终策略...")
    evaluate_policy(policy_net, test_loader, device, DISTILLATION_RATIO)
    
    # 在验证集上测试最终学生网络的准确度
    print("在验证集上测试最终学生网络的准确度...")
    final_student_net = StudentNetwork().to(device)
    # 重新训练最终学生网络以获得更好的性能评估
    final_optimizer = optim.Adam(final_student_net.parameters(), lr=LEARNING_RATE_STUDENT)
    
    # 使用策略网络选择浓缩数据进行训练
    with torch.no_grad():
        policy_net.eval()
        # 获取一批训练数据用于蒸馏
        state_data, state_labels = next(iter(train_loader))
        state_data, state_labels = state_data.to(device), state_labels.to(device)
        
        # 策略网络选择样本
        selection_probs = policy_net(state_data)
        selection_probs = torch.clamp(selection_probs, min=1e-3, max=1-1e-3)
        m = Bernoulli(selection_probs)
        actions = m.sample()
        selected_indices = (actions.squeeze() > 0.5).nonzero(as_tuple=True)[0]
        
        # 获取被选中的样本
        distilled_data = state_data[selected_indices]
        distilled_labels = state_labels[selected_indices]
        
        if len(distilled_data) > 0:
            distilled_dataset = torch.utils.data.TensorDataset(distilled_data, distilled_labels)
            distilled_loader = torch.utils.data.DataLoader(distilled_dataset, batch_size=min(MINI_BATCH_SIZE, len(distilled_data)), shuffle=True)
        else:
            distilled_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(state_data, state_labels), 
                                                          batch_size=MINI_BATCH_SIZE, shuffle=True)
    
    # 训练最终学生网络并测试验证集准确度
    print("训练最终学生网络并测试验证集准确度...")
    val_accuracies = []
    for epoch in range(10):  # 训练10个epoch进行评估
        avg_loss = train_student(final_student_net, distilled_loader, final_optimizer, criterion, K_STEPS, device)
        val_accuracy = evaluate_student(final_student_net, val_loader, device)
        val_accuracies.append(val_accuracy)
        print(f'最终学生网络训练 Epoch [{epoch+1}/10], 损失: {avg_loss:.4f}, 验证集准确度: {val_accuracy:.2f}%')
    
    # 测试集准确度
    test_accuracy = evaluate_student(final_student_net, test_loader, device)
    print(f"最终测试集准确度: {test_accuracy:.2f}%")
    
    # 输出统计信息
    print("\n" + "=" * 50)
    print("评估结果总结:")
    print(f"IPC设置: {IPC} (每类{IPC}张图像)")
    print(f"浓缩数据集大小: {len(condensed_trainset)}")
    print(f"最终验证集准确度: {val_accuracies[-1]:.2f}%")
    print(f"最终测试集准确度: {test_accuracy:.2f}%")
    print(f"平均验证集准确度: {np.mean(val_accuracies):.2f}% ± {np.std(val_accuracies):.2f}%")
    
    return policy_net, final_student_net, val_accuracies[-1], test_accuracy

if __name__ == "__main__":
    policy_net, student_net, val_acc, test_acc = ipc_evaluation()