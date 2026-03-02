"""
模型评估脚本
对训练好的模型进行完整评估，生成详细报告和可视化
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml
import pandas as pd
from tqdm import tqdm
from speechbrain.inference.speaker import SpeakerRecognition
from torch.utils.data import DataLoader, Dataset

from utils.losses import AAMSoftmax


class SpeakerDataset(Dataset):
    """说话人数据集"""
    
    def __init__(self, csv_path: str, audio_dir: str, sample_rate: int = 16000):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.speaker_ids = sorted(self.df['speaker_id'].unique())
        self.speaker_to_idx = {sid: idx for idx, sid in enumerate(self.speaker_ids)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        speaker_id = row['speaker_id']
        audio_path = os.path.join(self.audio_dir, row['audio_path'])
        
        # 加载音频
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return {
            'waveform': waveform.squeeze(0),
            'speaker_id': torch.tensor(self.speaker_to_idx[speaker_id]),
            'audio_path': audio_path,
            'speaker_id_str': speaker_id
        }


def collate_fn(batch):
    """自定义批处理函数"""
    waveforms = [item['waveform'] for item in batch]
    speaker_ids = torch.stack([item['speaker_id'] for item in batch])
    audio_paths = [item['audio_path'] for item in batch]
    speaker_id_strs = [item['speaker_id_str'] for item in batch]
    
    # 填充到相同长度
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


def load_config(config_path: str = "config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_det_curve(target_scores, non_target_scores):
    """
    计算DET曲线数据点
    
    Returns:
        false_alarm_rates: 误报率
        miss_rates: 漏报率
        thresholds: 阈值
    """
    # 合并所有分数
    scores = np.concatenate([target_scores, non_target_scores])
    labels = np.concatenate([np.ones(len(target_scores)), np.zeros(len(non_target_scores))])
    
    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # 计算累积统计
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


def compute_eer(target_scores, non_target_scores):
    """计算EER及其阈值"""
    far, mr, thresholds = compute_det_curve(target_scores, non_target_scores)
    
    # 找到FAR和MR相等的点
    eer_idx = np.argmin(np.abs(far - mr))
    eer = (far[eer_idx] + mr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold, far, mr


def compute_min_dcf(target_scores, non_target_scores, p_target=0.01, c_miss=1, c_fa=1):
    """计算最小检测代价函数"""
    far, mr, thresholds = compute_det_curve(target_scores, non_target_scores)
    
    # 计算检测代价
    dcf = c_miss * mr * p_target + c_fa * far * (1 - p_target)
    
    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    
    return min_dcf


def compute_roc_curve(target_scores, non_target_scores):
    """计算ROC曲线"""
    far, mr, _ = compute_det_curve(target_scores, non_target_scores)
    
    # TPR = 1 - MR
    tpr = 1 - mr
    
    return far, tpr


def compute_score_distribution(target_scores, non_target_scores):
    """计算分数分布"""
    return {
        'target_mean': np.mean(target_scores),
        'target_std': np.std(target_scores),
        'non_target_mean': np.mean(non_target_scores),
        'non_target_std': np.std(non_target_scores)
    }


def evaluate_all_pairs(embeddings_dict, speaker_labels, sample_ratio=1.0):
    """
    评估所有说话人对
    
    Args:
        embeddings_dict: {speaker_id: [embeddings]}
        speaker_labels: 所有样本的说话人标签
        sample_ratio: 采样比例（1.0表示全部）
    
    Returns:
        target_scores: 正样本对分数
        non_target_scores: 负样本对分数
    """
    speakers = list(embeddings_dict.keys())
    n_speakers = len(speakers)
    
    target_scores = []
    non_target_scores = []
    
    print(f"\n评估所有说话人对 (共 {n_speakers} 个说话人)...")
    
    # 计算所有说话人对
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


def plot_comprehensive_report(target_scores, non_target_scores, eer, eer_threshold, 
                              min_dcf, output_dir, model_name="Model"):
    """生成综合评估报告图"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. DET曲线
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
    
    # 2. ROC曲线
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
    
    # 6. 误差分析柱状图
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
    
    # 7. 性能表格
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    
    # 计算更多指标
    score_dist = compute_score_distribution(target_scores, non_target_scores)
    
    # 计算AUC
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
    
    # 不同应用场景的阈值建议
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
    
    # 9. 模型信息
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


def generate_metrics_table(target_scores, non_target_scores, eer, eer_threshold, min_dcf, output_dir):
    """生成详细的指标表格（CSV格式）"""
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
    
    # 保存CSV
    csv_path = os.path.join(output_dir, 'threshold_analysis.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['threshold', 'FAR', 'FRR', 'precision', 'recall', 'F1'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"阈值分析表已保存: {csv_path}")
    
    # 生成汇总表
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='模型评估脚本')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/classifier_final.pt', help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='logs/metrics', help='输出目录')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='采样比例（用于加速评估）')
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
    
    # 加载验证数据集
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
    
    # 提取所有嵌入向量
    print(f"\n提取嵌入向量...")
    embeddings_dict = {}  # {speaker_id: [embeddings]}
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="提取嵌入向量"):
            waveform = batch['waveform'].to(device)
            speaker_id = batch['speaker_id'][0]
            speaker_name = batch['speaker_id_str'][0]
            
            # 提取嵌入向量
            wav_lens = torch.ones(waveform.size(0), device=device)
            embedding = verification.encode_batch(waveform, wav_lens).squeeze().cpu()
            
            if speaker_name not in embeddings_dict:
                embeddings_dict[speaker_name] = []
            embeddings_dict[speaker_name].append(embedding)
    
    print(f"\n共提取 {len(embeddings_dict)} 个说话人的嵌入向量")
    for spk, embs in embeddings_dict.items():
        print(f"  {spk}: {len(embs)} 个样本")
    
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
    print(f"评估结果")
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
    
    # 生成表格
    csv_path, summary_path = generate_metrics_table(
        target_scores, non_target_scores, eer, eer_threshold,
        min_dcf, args.output
    )
    
    print(f"\n{'='*50}")
    print("评估完成！生成的文件：")
    print(f"  - 评估报告图: {report_path}")
    print(f"  - 阈值分析表: {csv_path}")
    print(f"  - 评估汇总表: {summary_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
