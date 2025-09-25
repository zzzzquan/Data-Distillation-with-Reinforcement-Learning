import torch
import numpy as np
from data.data_processing import get_data_transforms, load_cifar10_dataset, select_data_by_ipc

def simple_ipc_test():
    """
    简化版IPC测试，仅验证数据浓缩功能
    """
    print("开始简化版IPC=10测试...")
    print("=" * 40)
    
    # 获取数据预处理变换
    transform_train, transform_test = get_data_transforms()
    
    # 加载数据集
    print("加载CIFAR-10数据集...")
    trainset, testset = load_cifar10_dataset(transform_train, transform_test)
    
    print(f"原始训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    
    # 显示各类别样本数量
    train_labels = np.array(trainset.targets)
    unique, counts = np.unique(train_labels, return_counts=True)
    print("\n原始训练集中各类别样本数量:")
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"类别 {label}: {count} 样本")
    
    # 按IPC=10标准选择训练数据
    print(f"\n按IPC=10标准选择训练数据...")
    condensed_trainset = select_data_by_ipc(trainset, ipc=10)
    print(f"浓缩数据集大小: {len(condensed_trainset)} (每类10张图像)")
    
    # 显示浓缩后各类别样本数量
    condensed_labels = np.array([trainset.targets[i] for i in condensed_trainset.indices])
    unique, counts = np.unique(condensed_labels, return_counts=True)
    print("\n浓缩训练集中各类别样本数量:")
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"类别 {label}: {count} 样本")
    
    print("\n" + "=" * 40)
    print("简化版IPC测试完成!")
    print(f"数据浓缩比例: {len(condensed_trainset)}/{len(trainset)} = {len(condensed_trainset)/len(trainset)*100:.2f}%")
    
    return condensed_trainset

if __name__ == "__main__":
    condensed_dataset = simple_ipc_test()