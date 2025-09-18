#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制动作数据的7个子图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_action_data(csv_file):
    """
    读取CSV文件并绘制7个子图
    """
    # 读取CSV文件（无列名）
    data = pd.read_csv(csv_file, header=None)
    
    # 为列命名
    column_names = ['Action_1', 'Action_2', 'Action_3', 'Action_4', 'Action_5', 'Action_6', 'Action_7']
    data.columns = column_names
    
    # 创建时间步索引
    time_steps = np.arange(len(data))
    
    # 设置图像参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (16, 12)
    # 设置中文字体，如果没有中文字体则使用英文
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Action Data Visualization - 7 Action Dimensions', fontsize=16, fontweight='bold')
    
    # 绘制前7个子图
    for i in range(7):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 绘制线图
        ax.plot(time_steps, data.iloc[:, i], linewidth=2, color=f'C{i}')
        ax.set_title(f'{column_names[i]}', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = data.iloc[:, i].mean()
        std_val = data.iloc[:, i].std()
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_val:.3f}')
        ax.legend(fontsize=10)
        
        # 设置y轴范围
        y_min, y_max = data.iloc[:, i].min(), data.iloc[:, i].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # 隐藏多余的子图
    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = '/ML-vePFS/tangyinzhou/yinuo/ManiSkill_evaluation/action_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    
    # 显示图像
    plt.show()
    
    # 打印数据统计信息
    print("\nData Statistics:")
    print("=" * 50)
    for i, col in enumerate(column_names):
        stats = data.iloc[:, i].describe()
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Range: {stats['max'] - stats['min']:.6f}")

if __name__ == "__main__":
    csv_file = "/ML-vePFS/tangyinzhou/yinuo/ManiSkill_evaluation/debug/action.csv"
    plot_action_data(csv_file)
