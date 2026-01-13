import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties

def compare_attention_maps(file1_path, file2_path, output_path=None, title="Attention Difference"):
    """
    比较两个注意力CSV文件并可视化差异
    """
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        print(f"文件不存在: {file1_path}")
        return None
    if not os.path.exists(file2_path):
        print(f"文件不存在: {file2_path}")
        return None
        
    try:
        # 读取CSV文件
        df1 = pd.read_csv(file1_path, index_col=0)
        df2 = pd.read_csv(file2_path, index_col=0)
        
        # 确保两个矩阵具有相同的形状
        if df1.shape != df2.shape:
            print(f"注意力矩阵形状不匹配: {df1.shape} vs {df2.shape}")
            min_rows = min(df1.shape[0], df2.shape[0])
            min_cols = min(df1.shape[1], df2.shape[1])
            df1 = df1.iloc[:min_rows, :min_cols]
            df2 = df2.iloc[:min_rows, :min_cols]
        
        # 计算差值
        diff_df = df1 - df2
        
        # 统计信息
        max_diff = np.abs(diff_df.values).max()
        mean_diff = np.abs(diff_df.values).mean()
        
        # 设置matplotlib使用Agg后端以避免字体问题
        plt.switch_backend('Agg')
        
        # 创建热力图
        plt.figure(figsize=(10, 8), dpi=300)
        
        # 对称的色标范围
        vmax = max(abs(diff_df.values.min()), abs(diff_df.values.max()))
        
        # 绘制热力图 - 使用RdBu_r调色板
        ax = sns.heatmap(diff_df, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        xticklabels=True, yticklabels=True)
        
        # 调整标签
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        # 使用英文标题避免字体问题
        plt.title(f"{title}\nMax diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        plt.tight_layout()
        
        if output_path:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            print(f"已保存差异热力图到 {output_path}")
        else:
            plt.show()
        
        plt.close()  # 关闭图形，避免内存泄漏
        
        return diff_df
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

# 主函数
def main():
    # 基础路径
    base_dir1 = "/home/bd/data/LLaVA/FastV/output_example_backdoor/attention_weights"
    base_dir2 = "/home/bd/data/LLaVA/FastV/output_example/attention_weights"
    output_dir = "/home/bd/data/LLaVA/attention_diff"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理第1层到第32层
    for layer in range(0, 32):
        print(f"正在处理第 {layer} 层...")
        
        # 构建文件路径
        file1_path = os.path.join(base_dir1, f"layer_{layer}_attention.csv")
        file2_path = os.path.join(base_dir2, f"layer_{layer}_attention.csv")
        output_path = os.path.join(output_dir, f"layer_{layer}_attention_diff.png")
        
        # 设置标题
        title = f"Layer {layer} Attention Difference"
        
        # 比较注意力地图
        compare_attention_maps(file1_path, file2_path, output_path, title)

if __name__ == "__main__":
    main()