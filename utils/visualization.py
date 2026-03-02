"""
可视化工具模块
包含：DET曲线、特征可视化、训练指标可视化等
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score
from typing import List, Optional, Dict
import os


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


def plot_epoch_metrics(metrics: Dict[str, float],
                       epoch: int,
                       save_path: Optional[str] = None):
    """
    绘制单轮训练指标的柱状图
    
    Args:
        metrics: 包含各项指标的字典，如 {'loss': 0.5, 'accuracy': 0.85, 'f1': 0.82, 'eer': 0.12}
        epoch: 当前轮次
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 6))
    
    # 分离不同类型的指标
    loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
    rate_metrics = {k: v for k, v in metrics.items() if k.lower() not in ['loss'] and 'loss' not in k.lower()}
    
    # 创建子图
    if loss_metrics:
        plt.subplot(1, 2, 1)
        bars = plt.bar(loss_metrics.keys(), loss_metrics.values(), color=['#ff6b6b', '#ffa502'][:len(loss_metrics)])
        plt.ylabel('Loss Value', fontsize=12)
        plt.title(f'Epoch {epoch} - Loss Metrics', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        # 在柱子上显示数值
        for bar, val in zip(bars, loss_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
    
    if rate_metrics:
        plt.subplot(1, 2, 2)
        colors = ['#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9'][:len(rate_metrics)]
        bars = plt.bar(rate_metrics.keys(), rate_metrics.values(), color=colors)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'Epoch {epoch} - Performance Metrics', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        # 在柱子上显示数值
        for bar, val in zip(bars, rate_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        # 设置y轴范围0-1（针对比率指标）
        if all(0 <= v <= 1 for v in rate_metrics.values()):
            plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Epoch {epoch} 指标图已保存至: {save_path}")
    
    plt.close()


def plot_training_trends(history: Dict[str, List[float]],
                         save_path: Optional[str] = None):
    """
    绘制训练过程的总体趋势图
    
    Args:
        history: 包含各轮指标历史记录的字典
                 如 {'loss': [0.8, 0.6, ...], 'accuracy': [0.6, 0.7, ...], ...}
        save_path: 保存路径（可选）
    """
    if not history or not any(history.values()):
        print("警告: 没有训练历史数据可供绘制")
        return
    
    epochs = list(range(1, len(next(iter(history.values()))) + 1))
    
    # 计算需要多少个子图
    n_metrics = len(history)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 为不同指标定义颜色和样式
    colors = {
        'loss': '#ff6b6b',
        'val_loss': '#ffa502',
        'accuracy': '#4ecdc4',
        'val_accuracy': '#45b7d1',
        'f1': '#96ceb4',
        'val_f1': '#ffeaa7',
        'eer': '#e17055',
        'val_eer': '#fdcb6e',
        'min_dcf': '#a29bfe',
        'val_min_dcf': '#6c5ce7',
        'precision': '#00b894',
        'recall': '#00cec9'
    }
    
    for idx, (metric_name, values) in enumerate(history.items()):
        ax = axes[idx]
        color = colors.get(metric_name, '#74b9ff')
        
        ax.plot(epochs, values, marker='o', linewidth=2, markersize=6, 
                color=color, label=metric_name)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Trend', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 设置x轴为整数刻度
        ax.set_xticks(epochs)
        
        # 标注最小/最大值
        if values:
            if 'loss' in metric_name.lower() or 'eer' in metric_name.lower() or 'dcf' in metric_name.lower():
                # 这些指标越小越好
                best_idx = np.argmin(values)
                best_val = values[best_idx]
                ax.scatter([epochs[best_idx]], [best_val], color='red', s=100, zorder=5)
                ax.annotate(f'Best: {best_val:.4f}', 
                           xy=(epochs[best_idx], best_val),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=9, color='red')
            else:
                # 这些指标越大越好
                best_idx = np.argmax(values)
                best_val = values[best_idx]
                ax.scatter([epochs[best_idx]], [best_val], color='green', s=100, zorder=5)
                ax.annotate(f'Best: {best_val:.4f}', 
                           xy=(epochs[best_idx], best_val),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=9, color='green')
    
    # 隐藏多余的子图
    for idx in range(len(history), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Training Metrics Trends', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练趋势图已保存至: {save_path}")
    
    plt.close()


def plot_pretrain_comparison(pretrain_metrics: Dict[str, float],
                             finetuned_metrics: Dict[str, float],
                             save_path: Optional[str] = None):
    """
    绘制预训练模型与微调后模型的性能对比图
    
    Args:
        pretrain_metrics: 预训练模型的指标
        finetuned_metrics: 微调后模型的指标
        save_path: 保存路径（可选）
    """
    # 找出两者共有的指标
    common_metrics = sorted(set(pretrain_metrics.keys()) & set(finetuned_metrics.keys()))
    
    if not common_metrics:
        print("警告: 没有共同的指标可供对比")
        return
    
    x = np.arange(len(common_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pretrain_vals = [pretrain_metrics[m] for m in common_metrics]
    finetuned_vals = [finetuned_metrics[m] for m in common_metrics]
    
    bars1 = ax.bar(x - width/2, pretrain_vals, width, label='Pretrained', color='#74b9ff', alpha=0.8)
    bars2 = ax.bar(x + width/2, finetuned_vals, width, label='Finetuned', color='#00b894', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Pretrained vs Finetuned Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in common_metrics], fontsize=10)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for bar, val in zip(bars1, pretrain_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, finetuned_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 添加性能提升标注
    improvements = []
    for i, m in enumerate(common_metrics):
        if 'loss' in m.lower() or 'eer' in m.lower() or 'dcf' in m.lower():
            # 越小越好
            improvement = (pretrain_vals[i] - finetuned_vals[i]) / pretrain_vals[i] * 100
            if improvement > 0:
                improvements.append(f"{m}: ↓{improvement:.1f}%")
        else:
            # 越大越好
            improvement = (finetuned_vals[i] - pretrain_vals[i]) / pretrain_vals[i] * 100 if pretrain_vals[i] != 0 else 0
            if improvement > 0:
                improvements.append(f"{m}: ↑{improvement:.1f}%")
    
    if improvements:
        improvement_text = "Performance Improvement:\n" + "\n".join(improvements)
        ax.text(0.02, 0.98, improvement_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"模型对比图已保存至: {save_path}")
    
    plt.close()


def compute_classification_metrics(predictions: List[int], 
                                   labels: List[int]) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        predictions: 预测标签
        labels: 真实标签
        
    Returns:
        包含accuracy, precision, recall, f1的字典
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    accuracy = np.mean(predictions == labels)
    
    # 使用macro平均处理多分类
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
