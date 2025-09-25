import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

def get_data_transforms():
    """
    获取数据预处理变换
    
    Returns:
        tuple: (训练集变换, 测试集变换)
    """
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return transform_train, transform_test


def load_cifar10_dataset(transform_train, transform_test):
    """
    加载CIFAR-10数据集
    
    Args:
        transform_train: 训练集变换
        transform_test: 测试集变换
    
    Returns:
        tuple: (训练集, 测试集)
    """
    # 创建数据目录
    os.makedirs('./data', exist_ok=True)
    
    # 加载CIFAR-10数据集
    print("正在加载CIFAR-10数据集...")
    try:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        print("数据集加载完成")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        # 创建模拟数据集用于测试
        print("创建模拟数据集...")
        trainset = torchvision.datasets.FakeData(size=10000, image_size=(3, 32, 32), num_classes=10, transform=transform_train)
        testset = torchvision.datasets.FakeData(size=2000, image_size=(3, 32, 32), num_classes=10, transform=transform_test)
        
    return trainset, testset


def select_data_by_ipc(dataset, ipc=10):
    """
    按每个类别的图像数量(IPC)选择数据
    
    Args:
        dataset: 原始数据集
        ipc: 每个类别的图像数量
    
    Returns:
        torch.utils.data.Dataset: 选择后的数据集
    """
    # 获取所有标签
    labels = np.array(dataset.targets)
    
    # 为每个类别选择指定数量的样本
    selected_indices = []
    for class_idx in range(10):  # CIFAR-10有10个类别
        class_indices = np.where(labels == class_idx)[0]
        # 随机选择ipc个样本
        if len(class_indices) >= ipc:
            selected_class_indices = np.random.choice(class_indices, ipc, replace=False)
        else:
            # 如果某个类别的样本不足ipc个，则选择所有样本
            selected_class_indices = class_indices
        selected_indices.extend(selected_class_indices)
    
    # 创建子集
    selected_indices = torch.tensor(selected_indices)
    subset = torch.utils.data.Subset(dataset, selected_indices)
    return subset


def create_data_loaders(train_dataset, val_dataset, testset, STATE_BATCH_SIZE, batch_size_val=100, batch_size_test=100):
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        testset: 测试数据集
        STATE_BATCH_SIZE: 训练批次大小
        batch_size_val: 验证批次大小
        batch_size_test: 测试批次大小
    
    Returns:
        tuple: (训练加载器, 验证加载器, 测试加载器)
    """
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=STATE_BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, val_loader, test_loader