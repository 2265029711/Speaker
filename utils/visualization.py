"""
可视化工具模块
包含：DET曲线、特征可视化等
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from typing import List, Optional


def plot_det_curve(scores: List[float], 
                   labels: List[int],
                   title: str = "DET Curve",
                   save_path: Optional[str] = None):
    """
    绘制检测误差权衡曲线 (Detection Error Trade-off Curve)
    
    Args:
        scores: 相似度得分列表
        labels: 真实标签列表 (1=同一说话人, 0=不同说话人)
        title: 图表标题
        save_path: 保存路径（可选）
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 计算FPR和FNR
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # 创建DET曲线（使用对数坐标）
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, linewidth=2, label='DET Curve')
    
    # 设置对数坐标
    plt.xscale('log')
    plt.yscale('log')
    
    # 设置坐标轴范围和标签
    plt.xlim([0.001, 1])
    plt.ylim([0.001, 1])
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
    plt.title(title, fontsize=14)
    
    # 添加EER点（对角线交点）
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    plt.scatter([eer], [eer], color='red', s=100, zorder=5, label=f'EER = {eer:.4f}')
    
    # 添加对角线
    plt.plot([0.001, 1], [0.001, 1], 'k--', alpha=0.3, label='EER Line')
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DET曲线已保存至: {save_path}")
    
    plt.show()


def plot_score_distribution(target_scores: List[float],
                            non_target_scores: List[float],
                            title: str = "Score Distribution",
                            save_path: Optional[str] = None):
    """
    绘制得分分布直方图
    
    Args:
        target_scores: 目标说话人得分
        non_target_scores: 非目标说话人得分
        title: 图表标题
        save_path: 保存路径（可选）
    """
    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    bins = np.linspace(-1, 1, 50)
    plt.hist(target_scores, bins=bins, alpha=0.7, label='Target', color='green')
    plt.hist(non_target_scores, bins=bins, alpha=0.7, label='Non-target', color='red')
    
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"得分分布图已保存至: {save_path}")
    
    plt.show()


def plot_embedding_tsne(embeddings: np.ndarray,
                        labels: List[int],
                        title: str = "Embedding t-SNE",
                        save_path: Optional[str] = None):
    """
    使用t-SNE可视化声纹嵌入向量
    
    Args:
        embeddings: 嵌入向量矩阵 [N, dim]
        labels: 说话人标签
        title: 图表标题
        save_path: 保存路径（可选）
    """
    from sklearn.manifold import TSNE
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], label=f'Speaker {label}', alpha=0.7)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE可视化已保存至: {save_path}")
    
    plt.show()
