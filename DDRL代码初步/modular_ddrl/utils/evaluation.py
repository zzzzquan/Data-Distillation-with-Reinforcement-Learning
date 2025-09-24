import torch
from torch.distributions import Bernoulli

def evaluate_student(student_net, val_loader, device):
    """
    在验证集上评估学生网络
    
    Args:
        student_net: 学生网络模型
        val_loader: 验证集数据加载器
        device: 设备 (CPU/GPU)
    
    Returns:
        float: 准确率
    """
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
    accuracy = 100 * correct / total
    return accuracy


def evaluate_policy(policy_net, test_loader, device, DISTILLATION_RATIO):
    """
    在测试集上评估策略网络
    
    Args:
        policy_net: 策略网络模型
        test_loader: 测试集数据加载器
        device: 设备 (CPU/GPU)
        DISTILLATION_RATIO: 蒸馏比例
    
    Returns:
        float: 选择比例
    """
    policy_net.eval()
    selected_count = 0
    total_count = 0
    
    # 限制测试样本数量以加快评估
    max_test_samples = 2000
    current_samples = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            if current_samples >= max_test_samples:
                break
                
            data, labels = data.to(device), labels.to(device)
            current_samples += len(data)
            
            # 策略网络输出每个样本的选取概率
            selection_probs = policy_net(data)
            
            # 为避免所有概率都接近0或1，添加一个小的epsilon
            selection_probs = torch.clamp(selection_probs, min=1e-4, max=1-1e-4)
            
            # 创建伯努利分布并采样
            m = Bernoulli(selection_probs)
            actions = m.sample()  # 0或1，表示是否选择该样本
            
            # 计算选中的样本数
            selected_count += actions.sum().item()
            total_count += len(data)
            
            # 如果选中了过多的样本，则随机移除一些
            if selected_count > total_count * 0.8:
                selected_indices = (actions.squeeze() > 0.5).nonzero(as_tuple=True)[0]
                perm = torch.randperm(len(selected_indices))
                # 保留大约DISTILLATION_RATIO比例的样本
                keep_count = int(total_count * DISTILLATION_RATIO)
                selected_indices = selected_indices[perm[:keep_count]]
                selected_count = len(selected_indices)
    
    selection_ratio = selected_count / total_count if total_count > 0 else 0
    print(f"在测试集上选中的样本数: {selected_count}/{total_count} ({selection_ratio*100:.2f}%)")
    return selection_ratio