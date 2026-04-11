"""
可视化工具模块
包含：DET曲线、特征可视化、训练指标可视化等
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score
import os

from utils.plot_config import setup_matplotlib_for_chinese


_EXACT_METRIC_LABELS = {
    'train_loss': '训练损失',
    'train_acc': '训练准确率',
    'train_accuracy': '训练准确率',
    'train_f1': '训练F1值',
    'train_precision': '训练精确率',
    'train_recall': '训练召回率',
    'val_loss': '验证损失',
    'val_acc': '验证准确率',
    'val_accuracy': '验证准确率',
    'val_f1': '验证F1值',
    'val_precision': '验证精确率',
    'val_recall': '验证召回率',
    'valid_loss': '验证损失',
    'valid_acc': '验证准确率',
    'valid_accuracy': '验证准确率',
    'valid_f1': '验证F1值',
    'valid_precision': '验证精确率',
    'valid_recall': '验证召回率',
    'test_loss': '测试损失',
    'test_acc': '测试准确率',
    'test_accuracy': '测试准确率',
    'test_f1': '测试F1值',
    'test_precision': '测试精确率',
    'test_recall': '测试召回率',
    'eer': 'EER',
    'min_dcf': '最小DCF',
}


def _format_metric_name(metric_name: str) -> str:
    """将内部指标名称转换为适合展示的标签。"""
    normalized = metric_name.lower().replace('-', '_')
    if normalized in _EXACT_METRIC_LABELS:
        return _EXACT_METRIC_LABELS[normalized]

    tokens = normalized.split('_')
    prefix_labels = {
        'train': '训练',
        'val': '验证',
        'valid': '验证',
        'test': '测试',
    }
    metric_labels = {
        'acc': '准确率',
        'accuracy': '准确率',
        'f1': 'F1值',
        'eer': 'EER',
        'min': '最小',
        'dcf': 'DCF',
        'auc': 'AUC',
        'loss': '损失',
        'precision': '精确率',
        'recall': '召回率',
    }

    formatted = []
    for token in tokens:
        formatted.append(metric_labels.get(token, token.upper() if token in {'eer', 'dcf', 'auc'} else token.title()))

    if tokens and tokens[0] in prefix_labels:
        formatted[0] = prefix_labels[tokens[0]]

    return ''.join(formatted)


def _metric_family(metric_name: str) -> str:
    """将指标归类为损失类、性能类或误差类。"""
    lower = metric_name.lower()
    if 'loss' in lower:
        return 'loss'
    if 'eer' in lower or 'dcf' in lower:
        return 'error'
    return 'score'


def _metric_base_name(metric_name: str) -> str:
    """移除 `train_`、`val_` 等数据集前缀，保留指标主体名称。"""
    tokens = metric_name.lower().replace('-', '_').split('_')
    if tokens and tokens[0] in {'train', 'val', 'valid', 'test'}:
        tokens = tokens[1:]
    return '_'.join(tokens) if tokens else metric_name.lower()


def _series_style(metric_name: str) -> Dict[str, str]:
    """为指标曲线返回稳定的颜色和线型配置。"""
    base_name = _metric_base_name(metric_name)
    palette = {
        'loss': '#C44E52',
        'accuracy': '#4C72B0',
        'f1': '#55A868',
        'precision': '#8172B2',
        'recall': '#CCB974',
        'eer': '#DD8452',
        'min_dcf': '#64B5CD',
    }
    lower = metric_name.lower()
    if lower.startswith('val_') or lower.startswith('valid_'):
        line_style = '--'
    elif lower.startswith('test_'):
        line_style = ':'
    else:
        line_style = '-'

    return {
        'color': palette.get(base_name, '#5F6B7A'),
        'linestyle': line_style,
    }


def _style_axis(ax: Any) -> None:
    """为坐标轴应用更适合学术图表的轻量样式。"""
    ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#B8C0CC')
    ax.spines['bottom'].set_color('#B8C0CC')
    ax.tick_params(labelsize=10)


def _annotate_bar_values(
    ax: Any,
    bars: Sequence[Any],
    value_fmt: str = '{:.4f}',
    inside_threshold: float = 0.18,
) -> None:
    """为水平条形图添加数值标注，并尽量避免与条形边界重叠。"""
    x_min, x_max = ax.get_xlim()
    span = max(x_max - x_min, 1e-8)
    for bar in bars:
        value = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        if span > 0 and value > inside_threshold * span:
            ax.text(value - 0.02 * span, y, value_fmt.format(value),
                    va='center', ha='right', fontsize=10, color='white', fontweight='bold')
        else:
            ax.text(value + 0.015 * span, y, value_fmt.format(value),
                    va='center', ha='left', fontsize=10, color='#2F3B4A')


def _place_inline_labels(ax: Any, label_specs: List[Dict[str, Any]], x_position: float) -> None:
    """在线尾放置标签，并做简单的碰撞规避。"""
    if not label_specs:
        return

    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1e-8)
    min_gap = span * 0.08
    lower_bound = y_min + span * 0.04
    upper_bound = y_max - span * 0.04

    ordered = sorted(
        [
            {'index': idx, **spec}
            for idx, spec in enumerate(label_specs)
        ],
        key=lambda item: item['y']
    )

    adjusted = [item['y'] for item in ordered]
    for idx in range(1, len(adjusted)):
        adjusted[idx] = max(adjusted[idx], adjusted[idx - 1] + min_gap)

    overflow = adjusted[-1] - upper_bound
    if overflow > 0:
        adjusted = [value - overflow for value in adjusted]

    if adjusted[0] < lower_bound:
        shift = lower_bound - adjusted[0]
        adjusted = [value + shift for value in adjusted]

    for item, adjusted_y in zip(ordered, adjusted):
        ax.text(
            x_position,
            adjusted_y,
            item['label'],
            color=item['color'],
            fontsize=10,
            va='center',
            ha='left',
            clip_on=False,
        )
        ax.plot(
            [item['x'], x_position - 0.04 * max(x_position, 1)],
            [item['y'], adjusted_y],
            color=item['color'],
            linewidth=1.0,
            alpha=0.65,
            clip_on=False,
        )


def plot_det_curve(scores: List[float], 
                   labels: List[int],
                   title: str = "DET曲线",
                   save_path: Optional[str] = None) -> None:
    """
    绘制检测误差权衡曲线（DET 曲线）。
    
    Args:
        scores: 相似度得分序列。
        labels: 二分类标签，`1` 表示同一说话人。
        title: 图表标题。
        save_path: 可选的保存路径。
    """
    setup_matplotlib_for_chinese()

    scores = np.array(scores)
    labels = np.array(labels)
    
    # 计算误报率（FPR）和漏拒率（FNR）。
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # 使用对数坐标绘制 DET 曲线。
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, linewidth=2, label='DET曲线')
    
    # 设置对数坐标轴。
    plt.xscale('log')
    plt.yscale('log')
    
    # 设置坐标轴范围和标签。
    plt.xlim([0.001, 1])
    plt.ylim([0.001, 1])
    plt.xlabel('误接受率（FAR）', fontsize=12)
    plt.ylabel('误拒绝率（FRR）', fontsize=12)
    plt.title(title, fontsize=14)
    
    # 标出 EER 对应的近似交点。
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    plt.scatter([eer], [eer], color='red', s=100, zorder=5, label=f'EER = {eer:.4f}')
    
    # 添加参考对角线。
    plt.plot([0.001, 1], [0.001, 1], 'k--', alpha=0.3, label='EER参考线')
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DET曲线已保存至: {save_path}")
    
    plt.show()


def plot_score_distribution(target_scores: List[float],
                            non_target_scores: List[float],
                            title: str = "分数分布",
                            save_path: Optional[str] = None) -> None:
    """
    绘制目标分数与非目标分数的重叠直方图。
    
    Args:
        target_scores: 同一说话人配对的得分序列。
        non_target_scores: 不同说话人配对的得分序列。
        title: 图表标题。
        save_path: 可选的保存路径。
    """
    setup_matplotlib_for_chinese()

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制分数分布直方图。
    bins = np.linspace(-1, 1, 50)
    plt.hist(target_scores, bins=bins, alpha=0.7, label='同说话人', color='green')
    plt.hist(non_target_scores, bins=bins, alpha=0.7, label='不同说话人', color='red')
    
    plt.xlabel('相似度分数', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"得分分布图已保存至: {save_path}")
    
    plt.show()


def plot_embedding_tsne(embeddings: np.ndarray,
                        labels: List[int],
                        title: str = "嵌入向量 t-SNE 可视化",
                        save_path: Optional[str] = None) -> None:
    """
    使用 t-SNE 将嵌入向量可视化到二维平面。
    
    Args:
        embeddings: 形状为 `[N, dim]` 的嵌入向量矩阵。
        labels: 用于区分散点颜色的说话人标签。
        title: 图表标题。
        save_path: 可选的保存路径。
    """
    from sklearn.manifold import TSNE
    setup_matplotlib_for_chinese()
    
    # 使用 t-SNE 进行二维降维。
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # 按说话人标签绘制散点图。
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], label=f'说话人 {label}', alpha=0.7)
    
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE可视化已保存至: {save_path}")
    
    plt.show()


def plot_epoch_metrics(metrics: Dict[str, float],
                       epoch: int,
                       save_path: Optional[str] = None) -> None:
    """
    绘制单轮训练结果摘要图，并将损失与性能指标分面展示。
    
    Args:
        metrics: 单轮训练指标字典。
        epoch: 当前轮次，从 1 开始计数。
        save_path: 可选的保存路径。
    """
    setup_matplotlib_for_chinese()

    loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
    non_loss_metrics = {k: v for k, v in metrics.items() if 'loss' not in k.lower()}

    panels = []
    if loss_metrics:
        panels.append(('loss', loss_metrics))
    if non_loss_metrics:
        panels.append(('score', non_loss_metrics))

    if not panels:
        return

    width_ratios = [1.0 if panel_name == 'loss' else 1.45 for panel_name, _ in panels]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(12.5, 5.2),
        gridspec_kw={'width_ratios': width_ratios},
    )
    if len(panels) == 1:
        axes = [axes]

    fig.patch.set_facecolor('white')
    fig.suptitle(f'第 {epoch} 轮训练指标概览', fontsize=16, fontweight='bold', y=0.98)

    for ax, (panel_name, panel_metrics) in zip(axes, panels):
        labels = [_format_metric_name(name) for name in panel_metrics.keys()]
        values = list(panel_metrics.values())
        y_pos = np.arange(len(labels))

        if panel_name == 'loss':
            colors = [_series_style(name)['color'] for name in panel_metrics.keys()]
            ax.hlines(y_pos, 0, values, color=colors, linewidth=2.5, alpha=0.85)
            ax.scatter(values, y_pos, s=110, color=colors, edgecolors='white', linewidths=1.4, zorder=3)
            max_value = max(values) if values else 1.0
            ax.set_xlim(0, max_value * 1.22 if max_value > 0 else 1.0)
            ax.set_xlabel('损失值', fontsize=11)
            ax.set_title('损失指标', fontsize=13, fontweight='bold')
            for x_value, y_value in zip(values, y_pos):
                ax.text(x_value + max(ax.get_xlim()[1] * 0.02, 0.01), y_value, f'{x_value:.4f}',
                        va='center', ha='left', fontsize=10, color='#2F3B4A')
        else:
            order = np.argsort(values)[::-1]
            sorted_values = [values[idx] for idx in order]
            sorted_labels = [labels[idx] for idx in order]
            sorted_metric_names = [list(panel_metrics.keys())[idx] for idx in order]
            y_pos = np.arange(len(sorted_labels))
            colors = [_series_style(name)['color'] for name in sorted_metric_names]

            bars = ax.barh(y_pos, sorted_values, color=colors, edgecolor='none', alpha=0.9, height=0.62)
            if all(0.0 <= value <= 1.0 for value in sorted_values):
                ax.set_xlim(0, 1.0)
                ax.set_xlabel('指标值', fontsize=11)
                ax.set_xticks(np.linspace(0, 1.0, 6))
            else:
                max_value = max(sorted_values) if sorted_values else 1.0
                ax.set_xlim(0, max_value * 1.12 if max_value > 0 else 1.0)
                ax.set_xlabel('指标值', fontsize=11)

            ax.set_title('性能指标', fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            _annotate_bar_values(ax, bars)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels if panel_name == 'loss' else sorted_labels, fontsize=10)
        _style_axis(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Epoch {epoch} 指标图已保存至: {save_path}")
    
    plt.close(fig)


def plot_training_trends(history: Dict[str, List[float]],
                         save_path: Optional[str] = None) -> None:
    """
    按指标类别绘制训练过程中的整体趋势图。
    
    Args:
        history: 按指标名称组织的历史序列。
        save_path: 可选的保存路径。
    """
    setup_matplotlib_for_chinese()

    if not history or not any(history.values()):
        print("警告: 没有训练历史数据可供绘制")
        return

    metric_order = list(history.keys())
    epochs = list(range(1, len(next(iter(history.values()))) + 1))

    grouped_metrics = {
        'loss': [name for name in metric_order if _metric_family(name) == 'loss'],
        'score': [name for name in metric_order if _metric_family(name) == 'score'],
        'error': [name for name in metric_order if _metric_family(name) == 'error'],
    }

    panel_specs = []
    if grouped_metrics['loss']:
        panel_specs.append(('损失变化', '损失值', grouped_metrics['loss']))
    if grouped_metrics['score']:
        panel_specs.append(('性能变化', '指标值', grouped_metrics['score']))
    if grouped_metrics['error']:
        panel_specs.append(('错误率变化', '错误率', grouped_metrics['error']))

    if not panel_specs:
        return

    fig, axes = plt.subplots(
        len(panel_specs),
        1,
        figsize=(11.5, 3.6 * len(panel_specs)),
        sharex=True,
    )
    if len(panel_specs) == 1:
        axes = [axes]

    fig.patch.set_facecolor('white')
    fig.suptitle('训练趋势', fontsize=17, fontweight='bold', y=0.985)

    max_epoch = epochs[-1]
    extra_space = max(1, int(np.ceil(max_epoch * 0.12)))
    label_x = max_epoch + extra_space * 0.22
    x_ticks = epochs if len(epochs) <= 12 else np.unique(np.linspace(1, max_epoch, 8, dtype=int))

    for ax, (panel_title, y_label, metric_names) in zip(axes, panel_specs):
        label_specs = []
        for metric_name in metric_names:
            values = history.get(metric_name, [])
            if not values:
                continue

            style = _series_style(metric_name)
            display_name = _format_metric_name(metric_name)
            marker_step = max(1, len(epochs) // 8)

            ax.plot(
                epochs,
                values,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2.4,
                solid_capstyle='round',
            )
            ax.plot(
                epochs[::marker_step],
                values[::marker_step],
                linestyle='None',
                marker='o',
                markersize=3.8,
                color=style['color'],
                alpha=0.75,
            )

            if _metric_family(metric_name) in {'loss', 'error'}:
                best_idx = int(np.argmin(values))
            else:
                best_idx = int(np.argmax(values))

            ax.scatter(
                epochs[best_idx],
                values[best_idx],
                s=42,
                facecolors='white',
                edgecolors=style['color'],
                linewidths=1.6,
                zorder=4,
            )

            label_specs.append({
                'label': display_name,
                'x': epochs[-1],
                'y': values[-1],
                'color': style['color'],
            })

        if all(
            all(0.0 <= value <= 1.0 for value in history.get(metric_name, []))
            for metric_name in metric_names
            if history.get(metric_name)
        ):
            ax.set_ylim(0, 1.02)

        ax.set_title(panel_title, fontsize=13, fontweight='bold', loc='left')
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlim(1, max_epoch + extra_space)
        ax.set_xticks(x_ticks)
        _style_axis(ax)
        _place_inline_labels(ax, label_specs, label_x)

    axes[-1].set_xlabel('轮次', fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.94, 0.965])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"训练趋势图已保存至: {save_path}")
    
    plt.close(fig)


def plot_pretrain_comparison(pretrain_metrics: Dict[str, float],
                             finetuned_metrics: Dict[str, float],
                             save_path: Optional[str] = None) -> None:
    """
    使用分组柱状图对比预训练模型与微调模型的指标。
    
    Args:
        pretrain_metrics: 预训练模型的指标字典。
        finetuned_metrics: 微调后模型的指标字典。
        save_path: 可选的保存路径。
    """
    setup_matplotlib_for_chinese()

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
    
    bars1 = ax.bar(x - width/2, pretrain_vals, width, label='预训练模型', color='#74b9ff', alpha=0.8)
    bars2 = ax.bar(x + width/2, finetuned_vals, width, label='微调后模型', color='#00b894', alpha=0.8)
    
    ax.set_xlabel('指标', fontsize=12)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('预训练与微调模型性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_format_metric_name(m) for m in common_metrics], fontsize=10)
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
            if pretrain_vals[i] != 0:
                improvement = (pretrain_vals[i] - finetuned_vals[i]) / pretrain_vals[i] * 100
                if improvement > 0:
                    improvements.append(f"{_format_metric_name(m)}：下降 {improvement:.1f}%")
        else:
            # 越大越好
            if pretrain_vals[i] != 0:
                improvement = (finetuned_vals[i] - pretrain_vals[i]) / pretrain_vals[i] * 100
                if improvement > 0:
                    improvements.append(f"{_format_metric_name(m)}：提升 {improvement:.1f}%")
    
    if improvements:
        improvement_text = "性能提升：\n" + "\n".join(improvements)
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
    根据预测结果与真实标签计算分类指标。
    
    Args:
        predictions: 预测类别编号。
        labels: 真实类别编号。
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    accuracy = np.mean(predictions == labels)
    
    # 多分类任务下采用宏平均，避免样本量较大的类别占据过高权重。
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
