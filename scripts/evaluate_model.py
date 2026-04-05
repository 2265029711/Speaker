"""
模型评估脚本
对训练好的模型进行完整评估，生成详细报告和可视化
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml
import pandas as pd
from tqdm import tqdm
import csv
from speechbrain.inference.speaker import SpeakerRecognition
from torch.utils.data import DataLoader, Dataset

from utils.losses import AAMSoftmax


SpeakerSample = Dict[str, Any]
SpeakerBatch = Dict[str, Any]


class SpeakerDataset(Dataset):
    """从 CSV 加载音频路径与说话人标签，并返回评估所需张量。"""
    
    def __init__(self, csv_path: str, audio_dir: str, sample_rate: int = 16000) -> None:
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.speaker_ids = sorted(self.df['speaker_id'].unique())
        self.speaker_to_idx = {sid: idx for idx, sid in enumerate(self.speaker_ids)}
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> SpeakerSample:
        row = self.df.iloc[idx]
        speaker_id = row['speaker_id']
        audio_path = os.path.join(self.audio_dir, row['audio_path'])
        
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return {
            'waveform': waveform.squeeze(0),
            'speaker_id': torch.tensor(self.speaker_to_idx[speaker_id]),
            'audio_path': audio_path,
            'speaker_id_str': speaker_id
        }


def collate_fn(batch: Sequence[SpeakerSample]) -> SpeakerBatch:
    """将批次内变长音频补齐到统一长度。"""
    waveforms = [item['waveform'] for item in batch]
    speaker_ids = torch.stack([item['speaker_id'] for item in batch])
    audio_paths = [item['audio_path'] for item in batch]
    speaker_id_strs = [item['speaker_id_str'] for item in batch]
    
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w
    
    return {
        'waveform': padded_waveforms,
        'speaker_id': speaker_ids,
        'audio_path': audio_paths,
        'speaker_id_str': speaker_id_strs
    }


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载并解析 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_det_curve(
    target_scores: Sequence[float],
    non_target_scores: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 DET 曲线所需的误报率、漏报率与阈值序列。"""
    scores = np.concatenate([target_scores, non_target_scores])
    labels = np.concatenate([np.ones(len(target_scores)), np.zeros(len(non_target_scores))])
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    
    n_targets = len(target_scores)
    n_non_targets = len(non_target_scores)
    
    false_alarm_rates = []
    miss_rates = []
    thresholds = []
    
    target_count = 0
    non_target_count = 0
    
    for i, label in enumerate(sorted_labels):
        if label == 1:
            target_count += 1
        else:
            non_target_count += 1
        
        far = non_target_count / n_non_targets
        mr = 1 - target_count / n_targets
        
        false_alarm_rates.append(far)
        miss_rates.append(mr)
        thresholds.append(scores[sorted_indices[i]])
    
    return np.array(false_alarm_rates), np.array(miss_rates), np.array(thresholds)


def compute_eer(
    target_scores: Sequence[float],
    non_target_scores: Sequence[float],
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """计算 EER、对应阈值以及 DET 曲线中间结果。"""
    far, mr, thresholds = compute_det_curve(target_scores, non_target_scores)
    
    # 找到 FAR 与 MR 最接近的点，作为 EER 的近似位置。
    eer_idx = np.argmin(np.abs(far - mr))
    eer = (far[eer_idx] + mr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold, far, mr


def compute_min_dcf(
    target_scores: Sequence[float],
    non_target_scores: Sequence[float],
    p_target: float = 0.01,
    c_miss: float = 1,
    c_fa: float = 1,
) -> float:
    """计算最小检测代价函数（minDCF）。"""
    far, mr, _ = compute_det_curve(target_scores, non_target_scores)
    
    # 计算检测代价
    dcf = c_miss * mr * p_target + c_fa * far * (1 - p_target)
    
    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    
    return min_dcf


def compute_roc_curve(
    target_scores: Sequence[float],
    non_target_scores: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """根据目标分数和非目标分数计算 ROC 曲线点。"""
    far, mr, _ = compute_det_curve(target_scores, non_target_scores)
    
    # 真阳性率（TPR）等于 `1 - 漏报率`。
    tpr = 1 - mr
    
    return far, tpr


def compute_score_distribution(
    target_scores: Sequence[float],
    non_target_scores: Sequence[float],
) -> Dict[str, float]:
    """返回目标分数与非目标分数的统计摘要。"""
    return {
        'target_mean': np.mean(target_scores),
        'target_std': np.std(target_scores),
        'non_target_mean': np.mean(non_target_scores),
        'non_target_std': np.std(non_target_scores)
    }


def evaluate_verification_pairs(
    verification: SpeakerRecognition,
    pairs_csv: str,
    audio_dir: str,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """基于预生成的音频对 CSV 评估说话人验证准确率。"""
    df = pd.read_csv(pairs_csv)
    print(f"\n加载验证对: {len(df)} 对")
    print(f"  正样本 (label=1): {(df['label']==1).sum()} 对")
    print(f"  负样本 (label=0): {(df['label']==0).sum()} 对")
    
    embedding_cache: Dict[str, torch.Tensor] = {}
    
    def get_embedding(audio_path: str) -> torch.Tensor:
        """获取单个音频文件的嵌入向量，并使用缓存避免重复计算。"""
        if audio_path in embedding_cache:
            return embedding_cache[audio_path]
        
        full_path = os.path.join(audio_dir, audio_path)
        
        import torchaudio
        waveform, sr = torchaudio.load(full_path)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.to(device)
        wav_lens = torch.ones(1, device=device)
        
        with torch.no_grad():
            embedding = verification.encode_batch(waveform, wav_lens).squeeze().cpu()
        
        embedding_cache[audio_path] = embedding
        return embedding
    
    # 计算相似度
    scores = []
    labels = []
    predictions = []
    pairs_info = []
    
    print("\n计算音频对相似度...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="验证对评估"):
        audio1 = row['audio1']
        audio2 = row['audio2']
        true_label = row['label']
        
        try:
            emb1 = get_embedding(audio1)
            emb2 = get_embedding(audio2)
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()
            
            # 根据阈值预测
            pred_label = 1 if similarity >= threshold else 0
            
            scores.append(similarity)
            labels.append(true_label)
            predictions.append(pred_label)
            pairs_info.append((audio1, audio2, true_label))
            
        except Exception as e:
            print(f"处理 {audio1} - {audio2} 时出错: {e}")
            continue
    
    scores = np.array(scores)
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # 计算正负样本的分数分布
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    # 计算基于当前正负样本分数的 EER。
    eer, eer_threshold, _, _ = compute_eer(pos_scores, neg_scores)
    
    # 混淆矩阵
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'threshold': threshold,
        'pos_score_mean': np.mean(pos_scores) if len(pos_scores) > 0 else 0,
        'pos_score_std': np.std(pos_scores) if len(pos_scores) > 0 else 0,
        'neg_score_mean': np.mean(neg_scores) if len(neg_scores) > 0 else 0,
        'neg_score_std': np.std(neg_scores) if len(neg_scores) > 0 else 0,
        'total_pairs': len(scores),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'scores': scores,
        'labels': labels,
        'predictions': predictions,
        'pairs':pairs_info
    }


def _style_report_axes(ax: Any) -> None:
    """为评估报告中的坐标轴应用更规范的学术图表样式。"""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#B0B0B0')
    ax.spines['bottom'].set_color('#B0B0B0')
    ax.tick_params(colors='#333333')


def plot_verification_pairs_report(pairs_metrics: Dict[str, Any], output_path: str) -> None:
    """生成验证对评估报告图，并保存到指定路径。"""
    scores = pairs_metrics['scores']
    labels_arr = pairs_metrics['labels']
    pos_scores = scores[labels_arr == 1]
    neg_scores = scores[labels_arr == 0]

    cm = np.array([
        [pairs_metrics['tn'], pairs_metrics['fp']],
        [pairs_metrics['fn'], pairs_metrics['tp']]
    ])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums > 0
    )

    far = pairs_metrics['fp'] / max(pairs_metrics['fp'] + pairs_metrics['tn'], 1)
    frr = pairs_metrics['fn'] / max(pairs_metrics['fn'] + pairs_metrics['tp'], 1)

    fig = plt.figure(figsize=(13.6, 8.8), facecolor='white')
    gs = GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[0.12, 1.02, 0.96],
        hspace=0.42
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.text(
        0.5,
        0.78,
        'Speaker Verification Pair Report',
        ha='center',
        va='center',
        fontsize=20,
        fontweight='bold',
        color='#111111'
    )

    ax_dist = fig.add_subplot(gs[1, 0])
    _style_report_axes(ax_dist)

    score_min = min(np.min(scores), pairs_metrics['threshold'], pairs_metrics['eer_threshold'])
    score_max = max(np.max(scores), pairs_metrics['threshold'], pairs_metrics['eer_threshold'])
    margin = max((score_max - score_min) * 0.08, 0.02)
    bins = np.linspace(score_min - margin, score_max + margin, 36)

    ax_dist.hist(
        neg_scores,
        bins=bins,
        density=True,
        alpha=0.38,
        color='#C44E52',
        edgecolor='white',
        linewidth=0.8,
        label=f'Non-target trials (n={len(neg_scores)})'
    )
    ax_dist.hist(
        pos_scores,
        bins=bins,
        density=True,
        alpha=0.45,
        color='#4C72B0',
        edgecolor='white',
        linewidth=0.8,
        label=f'Target trials (n={len(pos_scores)})'
    )
    ax_dist.axvline(
        pairs_metrics['threshold'],
        color='#1F1F1F',
        linestyle='--',
        linewidth=1.3,
        alpha=0.9,
        ymin=0.0,
        ymax=0.88,
        label=f'Operating threshold = {pairs_metrics["threshold"]:.4f}'
    )
    ax_dist.set_title('Verification Score Distribution', fontsize=14, fontweight='bold', pad=34)
    ax_dist.set_xlabel('Cosine similarity score', fontsize=11, labelpad=2)
    ax_dist.set_ylabel('Density', fontsize=11)
    ax_dist.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.25)
    ax_dist.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=10,
        columnspacing=1.8,
        handlelength=2.2,
        borderaxespad=0.0
    )

    bottom_gs = gs[2, 0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)

    ax_cm = fig.add_subplot(bottom_gs[0, 0])
    _style_report_axes(ax_cm)
    im = ax_cm.imshow(cm_normalized, cmap='Blues', vmin=0.0, vmax=1.0)
    ax_cm.set_title('Confusion Matrix (row-normalized)', fontsize=13, fontweight='bold', pad=6)
    ax_cm.set_xlabel('Predicted label', fontsize=11)
    ax_cm.set_ylabel('True label', fontsize=11)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Different speaker', 'Same speaker'])
    ax_cm.set_yticklabels(['Different speaker', 'Same speaker'])
    ax_cm.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax_cm.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax_cm.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax_cm.tick_params(which='minor', bottom=False, left=False)

    for i in range(2):
        for j in range(2):
            rate = cm_normalized[i, j] * 100
            text_color = 'white' if cm_normalized[i, j] >= 0.55 else '#1F1F1F'
            ax_cm.text(
                j,
                i,
                f'{cm[i, j]}\n{rate:.1f}%',
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                color=text_color
            )

    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label('Row proportion', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax_table = fig.add_subplot(bottom_gs[0, 1])
    ax_table.axis('off')
    ax_table.set_title('Evaluation Summary', fontsize=13, fontweight='bold', pad=10)

    summary_rows = [
        ['Operating threshold', f"{pairs_metrics['threshold']:.4f}"],
        ['EER threshold', f"{pairs_metrics['eer_threshold']:.4f}"],
        ['Accuracy', f"{pairs_metrics['accuracy'] * 100:.2f}%"],
        ['Precision', f"{pairs_metrics['precision'] * 100:.2f}%"],
        ['Recall', f"{pairs_metrics['recall'] * 100:.2f}%"],
        ['F1 score', f"{pairs_metrics['f1'] * 100:.2f}%"],
        ['Equal error rate', f"{pairs_metrics['eer'] * 100:.2f}%"],
        ['False acceptance rate', f'{far * 100:.2f}%'],
        ['False rejection rate', f'{frr * 100:.2f}%'],
        ['Target / non-target', f'{len(pos_scores)} / {len(neg_scores)}'],
    ]

    table = ax_table.table(
        cellText=summary_rows,
        colLabels=['Metric', 'Value'],
        cellLoc='center',
        colLoc='center',
        colWidths=[0.64, 0.36],
        bbox=[0.08, 0.12, 0.84, 0.75]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.28)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#D9D9D9')
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor('#EAEAEA')
            cell.set_text_props(fontweight='bold', color='#1F1F1F')
        else:
            cell.set_facecolor('#FAFAFA' if row % 2 else 'white')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def evaluate_all_pairs(
    embeddings_dict: Dict[str, List[torch.Tensor]],
    _speaker_labels: Sequence[str],
    sample_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    评估所有说话人对
    
    Args:
        embeddings_dict: {speaker_id: [embeddings]}
        _speaker_labels: 所有样本的说话人标签，占位参数，当前实现未直接使用
        sample_ratio: 采样比例（1.0表示全部）
    
    Returns:
        target_scores: 正样本对分数数组
        non_target_scores: 负样本对分数数组
    """
    speakers = list(embeddings_dict.keys())
    n_speakers = len(speakers)
    
    target_scores = []
    non_target_scores = []
    
    print(f"\n评估所有说话人对 (共 {n_speakers} 个说话人)...")
    
    # 遍历说话人组合，计算所有嵌入向量配对的余弦相似度。
    for i, spk1 in enumerate(tqdm(speakers, desc="计算说话人对")):
        emb1_list = embeddings_dict[spk1]
        
        for j, spk2 in enumerate(speakers):
            emb2_list = embeddings_dict[spk2]
            
            # 计算所有嵌入向量对的相似度
            for emb1 in emb1_list:
                for emb2 in emb2_list:
                    # 计算余弦相似度
                    similarity = torch.nn.functional.cosine_similarity(
                        emb1.unsqueeze(0), emb2.unsqueeze(0)
                    ).item()
                    
                    if spk1 == spk2:
                        target_scores.append(similarity)
                    else:
                        non_target_scores.append(similarity)
    
    # 采样以平衡数据量
    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)
    
    if sample_ratio < 1.0:
        n_target = int(len(target_scores) * sample_ratio)
        n_non_target = int(len(non_target_scores) * sample_ratio)
        
        target_idx = np.random.choice(len(target_scores), min(n_target, len(target_scores)), replace=False)
        non_target_idx = np.random.choice(len(non_target_scores), min(n_non_target, len(non_target_scores)), replace=False)
        
        target_scores = target_scores[target_idx]
        non_target_scores = non_target_scores[non_target_idx]
    
    print(f"正样本对数: {len(target_scores)}")
    print(f"负样本对数: {len(non_target_scores)}")
    
    return target_scores, non_target_scores


def plot_comprehensive_report(
    target_scores: Sequence[float],
    non_target_scores: Sequence[float],
    eer: float,
    eer_threshold: float,
    min_dcf: float,
    output_dir: str,
    model_name: str = "Model",
) -> str:
    """
    生成综合评估报告图，并返回图片保存路径。

    Args:
        target_scores: 同一说话人配对的相似度分数。
        non_target_scores: 不同说话人配对的相似度分数。
        eer: 等错误率。
        eer_threshold: 对应 EER 的最优阈值。
        min_dcf: 最小检测代价函数。
        output_dir: 图表输出目录。
        model_name: 报告中展示的模型名称。

    Returns:
        综合评估报告图的完整保存路径
    """
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. DET 曲线
    ax1 = fig.add_subplot(gs[0, 0])
    far, mr, _ = compute_det_curve(target_scores, non_target_scores)
    ax1.plot(far * 100, mr * 100, 'b-', linewidth=2, label='DET Curve')
    ax1.scatter([eer * 100], [eer * 100], color='red', s=100, zorder=5, label=f'EER = {eer*100:.2f}%')
    ax1.set_xlabel('False Alarm Rate (%)')
    ax1.set_ylabel('Miss Rate (%)')
    ax1.set_title('DET Curve (Detection Error Trade-off)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])
    
    # 2. ROC 曲线
    ax2 = fig.add_subplot(gs[0, 1])
    roc_far, roc_tpr = compute_roc_curve(target_scores, non_target_scores)
    ax2.plot(roc_far * 100, roc_tpr * 100, 'g-', linewidth=2, label='ROC Curve')
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
    ax2.fill_between(roc_far * 100, 0, roc_tpr * 100, alpha=0.2, color='green')
    ax2.set_xlabel('False Alarm Rate (%)')
    ax2.set_ylabel('True Positive Rate (%)')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 分数分布
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-0.5, 1.0, 100)
    ax3.hist(target_scores, bins=bins, alpha=0.6, label='Same Speaker', color='green', density=True)
    ax3.hist(non_target_scores, bins=bins, alpha=0.6, label='Different Speaker', color='red', density=True)
    ax3.axvline(x=eer_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {eer_threshold:.4f}')
    ax3.set_xlabel('Cosine Similarity Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 阈值分析
    ax4 = fig.add_subplot(gs[1, 0])
    thresholds = np.linspace(-0.5, 1.0, 200)
    fars = []
    frrs = []
    for th in thresholds:
        far = np.mean(non_target_scores >= th)
        frr = np.mean(target_scores < th)
        fars.append(far * 100)
        frrs.append(frr * 100)
    ax4.plot(thresholds, fars, 'r-', label='FAR (False Acceptance Rate)')
    ax4.plot(thresholds, frrs, 'b-', label='FRR (False Rejection Rate)')
    ax4.axvline(x=eer_threshold, color='green', linestyle='--', label=f'EER Threshold')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Error Rate (%)')
    ax4.set_title('FAR/FRR vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 精确率-召回率曲线
    ax5 = fig.add_subplot(gs[1, 1])
    precision_list = []
    recall_list = []
    for th in thresholds:
        tp = np.sum((target_scores >= th))
        fp = np.sum((non_target_scores >= th))
        fn = np.sum((target_scores < th))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    ax5.plot(recall_list, precision_list, 'purple', linewidth=2)
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall Curve')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # 6. 误差指标柱状图
    ax6 = fig.add_subplot(gs[1, 2])
    metrics = ['EER', 'minDCF', 'FAR@EER', 'FRR@EER']
    values = [eer * 100, min_dcf * 100, eer * 100, eer * 100]
    colors = ['red', 'orange', 'blue', 'green']
    bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
    ax6.set_ylabel('Value (%)')
    ax6.set_title('Error Metrics Summary')
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.2f}%', ha='center', va='bottom')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. 性能摘要表格
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    
    # 计算更多汇总指标
    score_dist = compute_score_distribution(target_scores, non_target_scores)
    
    # 计算 AUC
    auc = np.trapz(roc_tpr, roc_far)
    
    table_data = [
        ['Metric', 'Value'],
        ['EER', f'{eer*100:.4f}%'],
        ['EER Threshold', f'{eer_threshold:.4f}'],
        ['minDCF', f'{min_dcf:.4f}'],
        ['AUC', f'{auc:.4f}'],
        ['Target Score Mean', f'{score_dist["target_mean"]:.4f}'],
        ['Target Score Std', f'{score_dist["target_std"]:.4f}'],
        ['Non-Target Score Mean', f'{score_dist["non_target_mean"]:.4f}'],
        ['Non-Target Std', f'{score_dist["non_target_std"]:.4f}'],
        ['Target Pairs', f'{len(target_scores)}'],
        ['Non-Target Pairs', f'{len(non_target_scores)}'],
    ]
    
    table = ax7.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax7.set_title('Performance Metrics Summary', pad=20)
    
    # 8. 阈值选择建议
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    
    # 给出不同应用场景下的阈值建议
    threshold_suggestions = [
        ['Application', 'Threshold', 'FAR', 'FRR'],
        ['High Security', f'{np.percentile(non_target_scores, 99):.4f}', '1.00%', f'{(1 - np.mean(target_scores >= np.percentile(non_target_scores, 99)))*100:.2f}%'],
        ['Balanced', f'{eer_threshold:.4f}', f'{eer*100:.2f}%', f'{eer*100:.2f}%'],
        ['High Convenience', f'{np.percentile(target_scores, 1):.4f}', f'{np.mean(non_target_scores >= np.percentile(target_scores, 1))*100:.2f}%', '1.00%'],
    ]
    
    table2 = ax8.table(cellText=threshold_suggestions, loc='center', cellLoc='center',
                       colWidths=[0.35, 0.25, 0.2, 0.2])
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.5)
    
    for i in range(4):
        table2[(0, i)].set_facecolor('#4472C4')
        table2[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax8.set_title('Threshold Suggestions for Different Applications', pad=20)
    
    # 9. 模型信息与结论摘要
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    info_text = f"""
    Model Evaluation Report
    ========================
    
    Model: {model_name}
    
    Key Findings:
    - Equal Error Rate (EER): {eer*100:.2f}%
    - Optimal Threshold: {eer_threshold:.4f}
    - Minimum DCF: {min_dcf:.4f}
    - ROC AUC: {auc:.4f}
    
    Score Statistics:
    - Same speaker pairs tend to have
      higher similarity scores
    - Clear separation between target
      and non-target distributions
    
    Recommendations:
    - Use threshold around {eer_threshold:.4f}
      for balanced FAR/FRR
    - Adjust based on security needs
    """
    
    ax9.text(0.1, 0.5, info_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Speaker Verification Model Evaluation Report\n{model_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图片
    report_path = os.path.join(output_dir, 'model_evaluation_report.png')
    plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n评估报告已保存: {report_path}")
    
    return report_path


def generate_metrics_table(
    target_scores: Sequence[float],
    non_target_scores: Sequence[float],
    eer: float,
    eer_threshold: float,
    min_dcf: float,
    output_dir: str,
) -> Tuple[str, str]:
    """
    生成详细指标表与汇总表，并返回两个 CSV 路径。

    Args:
        target_scores: 同一说话人配对的相似度分数。
        non_target_scores: 不同说话人配对的相似度分数。
        eer: 等错误率。
        eer_threshold: 对应 EER 的阈值。
        min_dcf: 最小检测代价函数。
        output_dir: 输出目录。

    Returns:
        `(阈值分析表路径, 评估汇总表路径)`
    """
    import csv
    
    # 计算各种阈值下的性能
    thresholds = np.linspace(-0.5, 1.0, 50)
    rows = []
    
    for th in thresholds:
        far = np.mean(non_target_scores >= th)
        frr = np.mean(target_scores < th)
        precision = np.mean(target_scores[target_scores >= th]) if np.sum(target_scores >= th) > 0 else 0
        recall = 1 - frr
        
        rows.append({
            'threshold': th,
            'FAR': far,
            'FRR': frr,
            'precision': precision,
            'recall': recall,
            'F1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        })
    
    # 保存阈值分析表
    csv_path = os.path.join(output_dir, 'threshold_analysis.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['threshold', 'FAR', 'FRR', 'precision', 'recall', 'F1'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"阈值分析表已保存: {csv_path}")
    
    # 生成评估汇总表
    summary_path = os.path.join(output_dir, 'evaluation_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Description'])
        writer.writerow(['EER', f'{eer*100:.4f}%', 'Equal Error Rate'])
        writer.writerow(['EER_Threshold', f'{eer_threshold:.4f}', 'Threshold at EER'])
        writer.writerow(['minDCF', f'{min_dcf:.4f}', 'Minimum Detection Cost Function'])
        writer.writerow(['Target_Pairs', len(target_scores), 'Number of same-speaker pairs'])
        writer.writerow(['NonTarget_Pairs', len(non_target_scores), 'Number of different-speaker pairs'])
        writer.writerow(['Target_Mean', f'{np.mean(target_scores):.4f}', 'Mean similarity of target pairs'])
        writer.writerow(['Target_Std', f'{np.std(target_scores):.4f}', 'Std of target pair scores'])
        writer.writerow(['NonTarget_Mean', f'{np.mean(non_target_scores):.4f}', 'Mean similarity of non-target pairs'])
        writer.writerow(['NonTarget_Std', f'{np.std(non_target_scores):.4f}', 'Std of non-target scores'])
    
    print(f"评估汇总表已保存: {summary_path}")
    
    return csv_path, summary_path


def evaluate_classification_accuracy(
    verification: SpeakerRecognition,
    classifier: nn.Module,
    valid_loader: DataLoader,
    device: torch.device,
    speaker_to_idx: Dict[str, int],
    _idx_to_speaker: Dict[int, str],
    loss_type: str = 'softmax',
) -> Optional[Dict[str, Any]]:
    """
    评估分类准确率
    
    Args:
        verification: SpeakerRecognition 模型
        classifier: 分类器模型
        valid_loader: 验证数据加载器
        device: 设备
        speaker_to_idx: 说话人名称到索引的映射
        _idx_to_speaker: 索引到说话人名称的映射，占位参数，当前实现未直接使用
        loss_type: 损失函数类型
    
    Returns:
        包含准确率、每类准确率等指标的字典；若无有效样本则返回 `None`
    """
    classifier.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    correct_per_speaker = {}
    total_per_speaker = {}
    unknown_speakers = set()
    unknown_count = 0
    
    print("\n评估分类准确率...")
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="分类评估"):
            waveform = batch['waveform'].to(device)
            speaker_id_str = batch['speaker_id_str'][0]
            true_label = speaker_to_idx.get(speaker_id_str, -1)
            
            if true_label == -1:
                unknown_speakers.add(speaker_id_str)
                unknown_count += 1
                continue
            
            # 提取嵌入向量
            wav_lens = torch.ones(waveform.size(0), device=device)
            embedding = verification.encode_batch(waveform, wav_lens).squeeze()
            
            # 分类预测
            if loss_type == 'softmax':
                logits = classifier(embedding)
                probs = torch.softmax(logits, dim=-1)
            else:
                # AAM-Softmax 评估阶段不直接计算损失，这里使用归一化后的余弦相似度做分类打分。
                weight = classifier.weight  # [num_classes, embedding_dim]
                # 对嵌入和权重做归一化，保持与余弦分类假设一致。
                embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                weight_norm = torch.nn.functional.normalize(weight, p=2, dim=0)
                # 计算每个类别原型与当前嵌入之间的余弦相似度。
                similarities = torch.matmul(weight_norm.T, embedding_norm)  # [num_classes]
                probs = similarities
            
            pred_idx = torch.argmax(probs).item()
            pred_score = probs[pred_idx].item()
            
            all_predictions.append(pred_idx)
            all_labels.append(true_label)
            all_scores.append(pred_score)
            
            # 统计每个说话人的准确率
            if speaker_id_str not in total_per_speaker:
                total_per_speaker[speaker_id_str] = 0
                correct_per_speaker[speaker_id_str] = 0
            total_per_speaker[speaker_id_str] += 1
            if pred_idx == true_label:
                correct_per_speaker[speaker_id_str] += 1
    
    # 若一个有效样本都没有，则无法继续计算分类指标。
    if len(all_predictions) == 0:
        print(f"\n错误: 没有有效样本可评估！")
        print(f"  验证集中的说话人不在训练集中: {unknown_speakers}")
        print(f"  被跳过的样本数: {unknown_count}")
        print(f"\n请确保验证集和训练集使用相同的说话人!")
        return None
    
    # 打印未知说话人统计信息，方便排查训练集与验证集标签不一致的问题。
    if unknown_count > 0:
        print(f"\n警告: 跳过了 {unknown_count} 个样本（来自 {len(unknown_speakers)} 个未知说话人）")
        print(f"  未知说话人: {unknown_speakers}")
    
    # 计算指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    accuracy = np.mean(all_predictions == all_labels)
    
    # 计算每个说话人的准确率。
    per_speaker_accuracy = {}
    for spk in total_per_speaker:
        per_speaker_accuracy[spk] = correct_per_speaker[spk] / total_per_speaker[spk]
    
    # 计算宏平均分类指标与混淆矩阵。
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'per_speaker_accuracy': per_speaker_accuracy,
        'confusion_matrix': cm,
        'total_samples': len(all_predictions),
        'correct_predictions': int(np.sum(all_predictions == all_labels)),
        'unknown_count': unknown_count,
        'unknown_speakers': unknown_speakers
    }


def plot_classification_report(
    metrics: Dict[str, Any],
    speaker_to_idx: Dict[str, int],
    output_dir: str,
) -> str:
    """
    生成分类评估报告图，并返回图片保存路径。

    Args:
        metrics: 分类评估得到的指标字典。
        speaker_to_idx: 说话人到类别索引的映射。
        output_dir: 图表输出目录。

    Returns:
        分类评估报告图的完整保存路径
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. 整体指标
    ax1 = axes[0, 0]
    metric_names = ['Accuracy', 'F1', 'Precision', 'Recall']
    metric_values = [metrics['accuracy'], metrics['f1'], metrics['precision'], metrics['recall']]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    bars = ax1.bar(metric_names, [v * 100 for v in metric_values], color=colors, alpha=0.8)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Overall Classification Metrics')
    ax1.set_ylim([0, 105])
    for bar, val in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.2f}%', ha='center', va='bottom', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 每个说话人的准确率
    ax2 = axes[0, 1]
    speakers = list(metrics['per_speaker_accuracy'].keys())
    accuracies = [metrics['per_speaker_accuracy'][s] for s in speakers]
    
    # 按准确率排序
    sorted_idx = np.argsort(accuracies)
    speakers = [speakers[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    
    # 只显示前20和后20（如果说话人太多）
    if len(speakers) > 40:
        speakers = speakers[:20] + speakers[-20:]
        accuracies = accuracies[:20] + accuracies[-20:]
    
    colors = ['#f44336' if acc < 0.8 else '#4CAF50' for acc in accuracies]
    ax2.barh(speakers, [a * 100 for a in accuracies], color=colors, alpha=0.7)
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title('Per-Speaker Accuracy (sorted)')
    ax2.set_xlim([0, 105])
    ax2.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='80% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. 准确率分布直方图
    ax3 = axes[1, 0]
    all_accs = list(metrics['per_speaker_accuracy'].values())
    ax3.hist([a * 100 for a in all_accs], bins=20, color='#2196F3', alpha=0.7, edgecolor='white')
    ax3.axvline(x=np.mean(all_accs) * 100, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_accs)*100:.2f}%')
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_ylabel('Number of Speakers')
    ax3.set_title('Distribution of Per-Speaker Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 统计信息表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    all_accs = list(metrics['per_speaker_accuracy'].values())
    table_data = [
        ['Metric', 'Value'],
        ['Overall Accuracy', f'{metrics["accuracy"]*100:.4f}%'],
        ['F1 Score (Macro)', f'{metrics["f1"]*100:.4f}%'],
        ['Precision (Macro)', f'{metrics["precision"]*100:.4f}%'],
        ['Recall (Macro)', f'{metrics["recall"]*100:.4f}%'],
        ['Total Samples', f'{metrics["total_samples"]}'],
        ['Correct Predictions', f'{metrics["correct_predictions"]}'],
        ['Number of Speakers', f'{len(metrics["per_speaker_accuracy"])}'],
        ['Min Speaker Accuracy', f'{min(all_accs)*100:.2f}%'],
        ['Max Speaker Accuracy', f'{max(all_accs)*100:.2f}%'],
        ['Std Speaker Accuracy', f'{np.std(all_accs)*100:.2f}%'],
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Classification Summary', pad=20)
    
    plt.suptitle('Speaker Classification Evaluation Report', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    report_path = os.path.join(output_dir, 'classification_report.png')
    plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"分类报告已保存: {report_path}")
    
    return report_path


def main() -> None:
    """解析命令行参数并执行对应的评估流程。"""
    import argparse
    parser = argparse.ArgumentParser(description='模型评估脚本')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/classifier_final.pt', help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='logs/metrics', help='输出目录')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='采样比例（用于加速验证评估）')
    parser.add_argument('--mode', type=str, default='pairs', 
                       choices=['classification', 'verification', 'both', 'pairs'],
                       help='评估模式: pairs(验证对准确率), classification(分类准确率), verification(验证EER), both(两者)')
    parser.add_argument('--pairs_csv', type=str, default='data/valid_pairs.csv', help='音频对CSV文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='相似度阈值')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载预训练模型
    print(f"\n加载预训练模型: {config['model']['pretrained_model']}")
    verification = SpeakerRecognition.from_hparams(
        source=config['model']['pretrained_model'],
        savedir="pretrained_models",
        run_opts={"device": device}
    )
    embedding_model = verification.mods["embedding_model"]
    embedding_model.eval()
    
    # 加载训练好的分类器
    print(f"\n加载分类器检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 获取分类器配置
    loss_type = config['training'].get('loss_type', 'softmax')
    embedding_dim = config['model']['embedding_dim']
    speaker_to_idx = checkpoint.get('speaker_to_idx', {})
    num_speakers = checkpoint.get('num_speakers', len(speaker_to_idx))
    
    # 创建分类器
    if loss_type == 'softmax':
        classifier = torch.nn.Linear(embedding_dim, num_speakers).to(device)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        classifier.eval()
    else:
        # AAM-Softmax
        aam_s = config['training'].get('aam_s', 30.0)
        aam_m = config['training'].get('aam_m', 0.2)
        classifier = AAMSoftmax(embedding_dim, num_speakers, s=aam_s, m=aam_m).to(device)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        classifier.eval()
    
    print(f"分类器类型: {loss_type}")
    print(f"说话人数量: {num_speakers}")
    
    # 创建说话人映射
    idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
    
    # 根据模式选择评估方式
    generated_files = []
    
    # ========== 验证对准确率评估 ==========
    if args.mode in ['pairs', 'both']:
        print(f"\n{'='*60}")
        print("说话人验证对准确率评估")
        print(f"{'='*60}")
        
        pairs_metrics = evaluate_verification_pairs(
            verification, 
            config['data']['valid_csv'], 
            config['data']['valid_dir'], 
            device,
            threshold=args.threshold
        )
        
        print(f"\n{'='*50}")
        print(f"验证对评估结果")
        print(f"{'='*50}")
        print(f"阈值: {pairs_metrics['threshold']:.4f}")
        print(f"准确率: {pairs_metrics['accuracy']*100:.2f}%")
        print(f"精确率: {pairs_metrics['precision']*100:.2f}%")
        print(f"召回率: {pairs_metrics['recall']*100:.2f}%")
        print(f"F1 Score: {pairs_metrics['f1']*100:.2f}%")
        print(f"EER: {pairs_metrics['eer']*100:.2f}% (最佳阈值: {pairs_metrics['eer_threshold']:.4f})")
        print(f"\n分数统计:")
        print(f"  正样本相似度: {pairs_metrics['pos_score_mean']:.4f} ± {pairs_metrics['pos_score_std']:.4f}")
        print(f"  负样本相似度: {pairs_metrics['neg_score_mean']:.4f} ± {pairs_metrics['neg_score_std']:.4f}")
        print(f"\n混淆矩阵:")
        print(f"  TP={pairs_metrics['tp']}, TN={pairs_metrics['tn']}")
        print(f"  FP={pairs_metrics['fp']}, FN={pairs_metrics['fn']}")
        print(f"\n总样本对数: {pairs_metrics['total_pairs']}")
        
        # 保存结果到 CSV
        pairs_csv_output = os.path.join(args.output, 'verification_results.csv')
        with open(pairs_csv_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['audio1', 'audio2', 'true_label', 'predicted_label', 'similarity'])
            for i, (audio1, audio2, true_label) in enumerate(pairs_metrics['pairs']):
                if i < len(pairs_metrics['scores']):
                    writer.writerow([
                        audio1,
                        audio2,
                        pairs_metrics['labels'][i],
                        pairs_metrics['predictions'][i],
                        f"{pairs_metrics['scores'][i]:.4f}"
                    ])
        print(f"\n详细结果已保存: {pairs_csv_output}")
        generated_files.append(('验证对结果', pairs_csv_output))
        
        # 绘制分数分布图
        
        # 分数分布直方图
        
        # 混淆矩阵可视化
        fig_path = os.path.join(args.output, 'verification_pairs_report.png')
        plot_verification_pairs_report(pairs_metrics, fig_path)
        print(f"报告图已保存: {fig_path}")
        generated_files.append(('验证对报告图', fig_path))
    
    # ========== 加载验证数据集（仅 classification/verification 模式需要）==========
    if args.mode in ['classification', 'verification', 'both']:
        print(f"\n加载验证数据集...")
        sample_rate = config['audio']['sample_rate']
        valid_dataset = SpeakerDataset(
            csv_path=config['data']['valid_csv'],
            audio_dir=config['data']['valid_dir'],
            sample_rate=sample_rate
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,  # 单个样本处理
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        print(f"验证集样本数: {len(valid_dataset)}")
    
    # ========== 分类准确率评估 ==========
    if args.mode in ['classification', 'both']:
        print(f"\n{'='*60}")
        print("分类准确率评估")
        print(f"{'='*60}")
        
        # 需要重新创建 DataLoader，使用批量处理加速
        valid_loader_batch = DataLoader(
            valid_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        cls_metrics = evaluate_classification_accuracy(
            verification, classifier, valid_loader_batch, device,
            speaker_to_idx, idx_to_speaker, loss_type
        )
        
        if cls_metrics is None:
            print("\n无法进行分类评估，请检查训练集和验证集的说话人是否一致！")
        else:
            print(f"\n{'='*50}")
            print(f"分类评估结果")
            print(f"{'='*50}")
            print(f"准确率: {cls_metrics['accuracy']*100:.4f}%")
            print(f"F1 Score: {cls_metrics['f1']*100:.4f}%")
            print(f"Precision: {cls_metrics['precision']*100:.4f}%")
            print(f"Recall: {cls_metrics['recall']*100:.4f}%")
            print(f"总样本数: {cls_metrics['total_samples']}")
            print(f"正确预测数: {cls_metrics['correct_predictions']}")
            if cls_metrics.get('unknown_count', 0) > 0:
                print(f"跳过的未知样本: {cls_metrics['unknown_count']}")
            
            # 生成分类报告
            cls_report_path = plot_classification_report(cls_metrics, speaker_to_idx, args.output)
            generated_files.append(('分类报告图', cls_report_path))
        
        # 保存详细CSV
        cls_csv_path = os.path.join(args.output, 'classification_per_speaker.csv')
        with open(cls_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['speaker_id', 'accuracy', 'total_samples'])
            for spk, acc in sorted(cls_metrics['per_speaker_accuracy'].items()):
                writer.writerow([spk, f'{acc*100:.4f}%', 
                               sum(1 for s in valid_dataset.df['speaker_id'] if s == spk)])
        generated_files.append(('每说话人准确率表', cls_csv_path))
        print(f"每说话人准确率已保存: {cls_csv_path}")
    
    # ========== 说话人验证评估 (EER) ==========
    if args.mode in ['verification', 'both']:
        print(f"\n{'='*60}")
        print("说话人验证评估 (EER)")
        print(f"{'='*60}")
        
        # 提取所有嵌入向量
        print(f"\n提取嵌入向量...")
        embeddings_dict = {}  # {speaker_id: [embeddings]}
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="提取嵌入向量"):
                waveform = batch['waveform'].to(device)
                speaker_name = batch['speaker_id_str'][0]
                
                # 提取嵌入向量
                wav_lens = torch.ones(waveform.size(0), device=device)
                embedding = verification.encode_batch(waveform, wav_lens).squeeze().cpu()
                
                if speaker_name not in embeddings_dict:
                    embeddings_dict[speaker_name] = []
                embeddings_dict[speaker_name].append(embedding)
        
        print(f"\n共提取 {len(embeddings_dict)} 个说话人的嵌入向量")
        
        # 评估所有说话人对
        target_scores, non_target_scores = evaluate_all_pairs(
            embeddings_dict, 
            list(embeddings_dict.keys()),
            sample_ratio=args.sample_ratio
        )
        
        # 计算评估指标
        print("\n计算评估指标...")
        eer, eer_threshold, far, mr = compute_eer(target_scores, non_target_scores)
        min_dcf = compute_min_dcf(target_scores, non_target_scores)
        
        print(f"\n{'='*50}")
        print(f"验证评估结果")
        print(f"{'='*50}")
        print(f"EER: {eer*100:.4f}%")
        print(f"EER阈值: {eer_threshold:.4f}")
        print(f"minDCF: {min_dcf:.4f}")
        print(f"正样本对数: {len(target_scores)}")
        print(f"负样本对数: {len(non_target_scores)}")
        
        # 生成报告
        model_name = f"Fine-tuned ECAPA-TDNN ({loss_type})"
        report_path = plot_comprehensive_report(
            target_scores, non_target_scores, eer, eer_threshold,
            min_dcf, args.output, model_name
        )
        generated_files.append(('验证报告图', report_path))
        
        # 生成表格
        csv_path, summary_path = generate_metrics_table(
            target_scores, non_target_scores, eer, eer_threshold,
            min_dcf, args.output
        )
        generated_files.extend([
            ('阈值分析表', csv_path),
            ('验证汇总表', summary_path)
        ])
    
    print(f"\n{'='*50}")
    print("评估完成！生成的文件：")
    for name, path in generated_files:
        print(f"  - {name}: {path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
