import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

def load_mm_projection_data(file_path):
    """加载mm-projector输出数据"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_tsne(file1, file2, output_path="tsne_visualization.png"):
    """对两个mm-projector输出文件进行t-SNE降维可视化"""
    # 加载数据
    data1 = load_mm_projection_data(file1)
    data2 = load_mm_projection_data(file2)
    
    # 提取特征数据
    features1 = np.array(data1[0]["data"][0])  # shape: [576, 4096]
    features2 = np.array(data2[0]["data"][0])  # shape: [576, 4096]
    
    # 创建标签和合并数据
    labels = np.concatenate([np.zeros(features1.shape[0]), np.ones(features2.shape[0])])
    combined_features = np.vstack([features1, features2])
    
    print(f"特征形状: {combined_features.shape}, 标签形状: {labels.shape}")
    
    # 使用t-SNE进行降维
    print("正在执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded_features = tsne.fit_transform(combined_features)
    
    # 可视化
    plt.figure(figsize=(10, 8), dpi=300)
    
    # 分离两个数据集的降维结果
    embedded_features1 = embedded_features[:features1.shape[0]]
    embedded_features2 = embedded_features[features1.shape[0]:]
    
    # 绘制散点图
    plt.scatter(embedded_features1[:, 0], embedded_features1[:, 1], 
                c='blue', alpha=0.5, s=10, label='File 1')
    plt.scatter(embedded_features2[:, 0], embedded_features2[:, 1], 
                c='red', alpha=0.5, s=10, label='File 2')
    
    # 添加标题和标签
    plt.title('t-SNE Visualization of MM-Projector Outputs')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存图像
    plt.savefig(output_path, bbox_inches='tight')
    print(f"t-SNE可视化已保存至: {output_path}")
    
    # 绘制热图分析
    plt.figure(figsize=(12, 10), dpi=300)
    
    # 计算两组特征的平均值差异
    mean_diff = np.abs(np.mean(features1, axis=0) - np.mean(features2, axis=0))
    
    # 绘制差异的热图（只取前100个维度作为示例）
    top_dims = np.argsort(mean_diff)[-100:]
    sns.heatmap(np.vstack([
        np.mean(features1, axis=0)[top_dims], 
        np.mean(features2, axis=0)[top_dims]
    ]), cmap='coolwarm', 
       yticklabels=['File 1', 'File 2'], 
       xticklabels=top_dims)
    plt.title('Feature Differences Between Files (Top 100 dimensions)')
    plt.savefig(output_path.replace('.png', '_heatmap.png'), bbox_inches='tight')
    print(f"特征差异热图已保存至: {output_path.replace('.png', '_heatmap.png')}")
    
    return embedded_features, labels

if __name__ == "__main__":
    # 两个文件路径
    file1 = "/home/bd/data/LLaVA/FastV/output_example/mm_projection_output.json"
    file2 = "/home/bd/data/LLaVA/FastV/output_example_backdoor/mm_projection_output.json"
    
    # 执行t-SNE可视化
    embedded_features, labels = visualize_tsne(file1, file2)
    
    # 额外分析：计算特征间的余弦相似度
    features1 = np.array(load_mm_projection_data(file1)[0]["data"][0])
    features2 = np.array(load_mm_projection_data(file2)[0]["data"][0])
    
    # 对应patch之间的余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    patch_similarities = np.diagonal(cosine_similarity(features1, features2))
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(patch_similarities, bins=50, alpha=0.75)
    plt.title('Cosine Similarity Between Corresponding Patches')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('patch_similarity_histogram.png', bbox_inches='tight')
    print(f"Patch相似度直方图已保存至: patch_similarity_histogram.png")