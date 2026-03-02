"""
模型微调训练脚本
基于SpeechBrain框架微调ECAPA-TDNN模型
支持：多指标评估、训练可视化、预训练模型对比
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
from tqdm import tqdm
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.data_utils import undo_padding

from utils.audio_processor import AudioChecker, load_config
from utils.visualization import (
    plot_epoch_metrics, 
    plot_training_trends, 
    plot_pretrain_comparison,
    compute_classification_metrics
)
from utils.scorer import compute_eer, compute_min_dcf, cosine_similarity


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


def evaluate_model(verification, classifier, dataloader, device, idx_to_speaker):
    """
    评估模型性能（说话人验证方式）
    
    使用嵌入向量相似度进行说话人验证评估，而不是分类任务。
    主要指标：EER（等错误率）、minDCF（最小检测代价函数）
    
    Returns:
        dict: 包含eer, min_dcf等指标
    """
    classifier.eval()
    
    all_embeddings = []
    all_speaker_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取嵌入向量"):
            waveforms = batch['waveform'].to(device)
            
            # 提取嵌入向量
            wav_lens = torch.ones(waveforms.size(0), device=device)
            embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
            
            all_embeddings.append(embeddings.cpu())
            all_speaker_ids.extend(batch['speaker_id_str'])
    
    # 合并所有嵌入向量
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"\n验证集评估:")
    print(f"  总样本数: {len(all_speaker_ids)}")
    print(f"  说话人数: {len(set(all_speaker_ids))}")
    
    # 计算说话人验证指标
    eer, eer_threshold, scores, labels = compute_verification_eer(
        all_embeddings, 
        all_speaker_ids
    )
    
    # 计算minDCF
    min_dcf = None
    if len(set(labels)) >= 2:
        min_dcf, _ = compute_min_dcf(scores, labels)
    
    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'min_dcf': min_dcf if min_dcf else 0.0,
        'num_samples': len(all_speaker_ids),
        'num_speakers': len(set(all_speaker_ids))
    }


def compute_verification_eer(embeddings, speaker_ids, max_pairs=5000):
    """
    计算验证任务上的EER（使用全部样本随机组合）
    
    Args:
        embeddings: 所有音频的嵌入向量 (N, D)
        speaker_ids: 对应的说话人ID列表 (N,)
        max_pairs: 最大采样对数，避免计算量过大
        
    Returns:
        (eer, threshold, scores, labels)
    """
    import numpy as np
    
    embeddings = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
    n = len(speaker_ids)
    
    # 计算所有可能的组合数
    total_pairs = n * (n - 1) // 2
    
    print(f"  验证集样本数: {n}, 总组合对数: {total_pairs}")
    
    # 如果组合数太大，随机采样
    if total_pairs > max_pairs:
        print(f"  随机采样 {max_pairs} 对进行评估...")
        np.random.seed(42)
        # 随机生成不重复的索引对
        all_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sampled_indices = np.random.choice(len(all_indices), max_pairs, replace=False)
        pairs = [all_indices[idx] for idx in sampled_indices]
    else:
        # 使用全部组合
        print(f"  使用全部组合对进行评估...")
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    scores = []
    labels = []
    
    # 批量计算余弦相似度
    for i, j in pairs:
        # 计算余弦相似度
        e1 = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
        e2 = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
        sim = np.dot(e1, e2)
        
        scores.append(sim)
        labels.append(1 if speaker_ids[i] == speaker_ids[j] else 0)
    
    # 统计正负样本数
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"  正样本对(同一说话人): {pos_count}, 负样本对(不同说话人): {neg_count}")
    
    if len(set(labels)) < 2:
        print("  警告: 只有一个类别，无法计算EER")
        return 0.0, 0.5, scores, labels
    
    eer, threshold, _ = compute_eer(scores, labels)
    return eer, threshold, scores, labels


def evaluate_pretrained_model(verification, dataloader, device):
    """
    评估预训练模型性能（不使用分类器）
    
    使用嵌入向量相似度进行说话人验证评估。
    
    Returns:
        dict: 预训练模型在验证集上的指标
    """
    all_embeddings = []
    all_speaker_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估预训练模型"):
            waveforms = batch['waveform'].to(device)
            
            wav_lens = torch.ones(waveforms.size(0), device=device)
            embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
            
            all_embeddings.append(embeddings.cpu())
            all_speaker_ids.extend(batch['speaker_id_str'])
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"\n预训练模型评估:")
    print(f"  总样本数: {len(all_speaker_ids)}")
    print(f"  说话人数: {len(set(all_speaker_ids))}")
    
    # 计算EER
    eer, threshold, scores, labels = compute_verification_eer(
        all_embeddings,
        all_speaker_ids
    )
    
    # 计算minDCF
    min_dcf, _ = compute_min_dcf(scores, labels) if len(set(labels)) >= 2 else (0.0, None)
    
    return {
        'eer': eer,
        'eer_threshold': threshold,
        'min_dcf': min_dcf
    }


def train(config_path: str = "config/config.yaml"):
    """训练主函数"""
    
    # 加载配置
    config = load_config(config_path)
    model_cfg = config['model']
    training_cfg = config['training']
    data_cfg = config['data']
    audio_cfg = config['audio']
    output_cfg = config['output']
    
    device = torch.device(training_cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_cfg['checkpoint_dir'], exist_ok=True)
    metrics_dir = os.path.join(output_cfg.get('log_dir', 'logs'), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 加载预训练模型
    pretrained_path = model_cfg['pretrained_model']
    print(f"加载预训练模型: {pretrained_path}")
    
    local_hparams = os.path.join(pretrained_path, "hyperparams.yaml")
    
    if os.path.exists(local_hparams):
        # 使用本地模型 - 直接使用 SpeakerRecognition 加载本地路径
        # SpeechBrain 支持本地目录作为 source
        verification = SpeakerRecognition.from_hparams(
            source=pretrained_path,
            run_opts={"device": device}
        )
        print("本地预训练模型加载完成")
    else:
        # 使用HuggingFace模型
        verification = SpeakerRecognition.from_hparams(
            source=pretrained_path,
            savedir="pretrained_models",
            run_opts={"device": device}
        )
    
    # 创建训练数据集
    train_dataset = SpeakerDataset(
        csv_path=data_cfg['train_csv'],
        audio_dir=data_cfg['train_dir'],
        sample_rate=audio_cfg['sample_rate']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg['num_workers'],
        collate_fn=collate_fn
    )
    
    # 创建验证数据集（如果存在）
    valid_loader = None
    if os.path.exists(data_cfg.get('valid_csv', '')):
        valid_dataset = SpeakerDataset(
            csv_path=data_cfg['valid_csv'],
            audio_dir=data_cfg['valid_dir'],
            sample_rate=audio_cfg['sample_rate']
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=training_cfg['batch_size'],
            shuffle=False,
            num_workers=training_cfg['num_workers'],
            collate_fn=collate_fn
        )
        print(f"验证集大小: {len(valid_dataset)}")
    
    # 获取说话人数量
    num_speakers = len(train_dataset.speaker_ids)
    idx_to_speaker = {idx: sid for sid, idx in train_dataset.speaker_to_idx.items()}
    print(f"说话人数量: {num_speakers}")
    
    # 创建分类器头
    classifier = nn.Linear(model_cfg['embedding_dim'], num_speakers).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 只优化分类器参数
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=float(training_cfg['learning_rate']),
        weight_decay=float(training_cfg['weight_decay'])
    )
    
    # 评估预训练模型性能（作为基线）
    print("\n=== 评估预训练模型基线性能 ===")
    pretrain_metrics = {}
    if valid_loader:
        pretrain_metrics = evaluate_pretrained_model(verification, valid_loader, device)
        print(f"预训练模型 EER: {pretrain_metrics['eer']:.4f}")
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'val_eer': [],
        'val_min_dcf': []
    }
    
    # 训练循环
    print("\n=== 开始训练 ===")
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(training_cfg['epochs']):
        verification.eval()
        classifier.train()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_cfg['epochs']}")
        
        for batch in pbar:
            waveforms = batch['waveform'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            
            # 提取嵌入向量
            with torch.no_grad():
                wav_lens = torch.ones(waveforms.size(0), device=device)
                embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
            
            # 分类
            logits = classifier(embeddings)
            loss = criterion(logits, speaker_ids)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(speaker_ids.cpu().numpy())
            
            current_acc = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        # 计算训练指标
        train_metrics = compute_classification_metrics(all_predictions, all_labels)
        train_metrics['loss'] = total_loss / len(train_loader)
        
        # 记录训练历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        
        print(f"\nEpoch {epoch + 1} 训练结果:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1: {train_metrics['f1']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        
        # 验证集评估
        val_metrics = {}
        if valid_loader:
            val_metrics = evaluate_model(verification, classifier, valid_loader, device, idx_to_speaker)
            
            history['val_eer'].append(val_metrics['eer'])
            history['val_min_dcf'].append(val_metrics['min_dcf'])
            
            print(f"\nEpoch {epoch + 1} 验证结果 (说话人验证):")
            print(f"  EER: {val_metrics['eer']:.4f}")
            print(f"  EER阈值: {val_metrics['eer_threshold']:.4f}")
            print(f"  minDCF: {val_metrics['min_dcf']:.4f}")
            
            # 保存最佳模型（EER越低越好）
            if not hasattr(train, 'best_val_eer') or val_metrics['eer'] < train.best_val_eer:
                train.best_val_eer = val_metrics['eer']
                best_epoch = epoch + 1
        
        # 每轮保存指标图片
        epoch_metrics = {k: v for k, v in train_metrics.items()}
        if val_metrics:
            epoch_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        plot_epoch_metrics(
            epoch_metrics,
            epoch=epoch + 1,
            save_path=os.path.join(metrics_dir, f'epoch_{epoch + 1}_metrics.png')
        )
        
        # 每5轮保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                output_cfg['checkpoint_dir'],
                f"classifier_epoch_{epoch + 1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'speaker_to_idx': train_dataset.speaker_to_idx
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(output_cfg['checkpoint_dir'], "classifier_final.pt")
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'speaker_to_idx': train_dataset.speaker_to_idx,
        'num_speakers': num_speakers,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1
    }, final_path)
    print(f"\n最终模型已保存: {final_path}")
    
    # 保存训练趋势图
    plot_training_trends(
        history,
        save_path=os.path.join(metrics_dir, 'training_trends.png')
    )
    
    # 保存预训练vs微调对比图
    if pretrain_metrics and val_metrics:
        finetuned_metrics = {
            'eer': val_metrics['eer'],
            'min_dcf': val_metrics['min_dcf']
        }
        plot_pretrain_comparison(
            pretrain_metrics,
            finetuned_metrics,
            save_path=os.path.join(metrics_dir, 'pretrain_vs_finetuned.png')
        )
        
        # 打印性能提升总结
        print("\n" + "="*50)
        print("=== 训练完成 - 性能总结 ===")
        print("="*50)
        print(f"最佳轮次: Epoch {best_epoch}")
        
        if 'eer' in pretrain_metrics:
            # EER越低越好
            eer_improvement = (pretrain_metrics['eer'] - finetuned_metrics['eer']) / pretrain_metrics['eer'] * 100
            print(f"\n预训练模型 EER: {pretrain_metrics['eer']:.4f}")
            print(f"微调后模型 EER: {finetuned_metrics['eer']:.4f}")
            if eer_improvement > 0:
                print(f"EER 改善: {eer_improvement:.2f}% (越低越好)")
            else:
                print(f"EER 变化: {eer_improvement:.2f}% (越低越好)")
            
            if 'min_dcf' in pretrain_metrics and pretrain_metrics['min_dcf'] > 0:
                dcf_improvement = (pretrain_metrics['min_dcf'] - finetuned_metrics['min_dcf']) / pretrain_metrics['min_dcf'] * 100
                print(f"\n预训练模型 minDCF: {pretrain_metrics['min_dcf']:.4f}")
                print(f"微调后模型 minDCF: {finetuned_metrics['min_dcf']:.4f}")
                if dcf_improvement > 0:
                    print(f"minDCF 改善: {dcf_improvement:.2f}% (越低越好)")
                else:
                    print(f"minDCF 变化: {dcf_improvement:.2f}% (越低越好)")
        
        print(f"\n指标图片已保存至: {metrics_dir}")
    
    return classifier, train_dataset.speaker_to_idx, history


if __name__ == "__main__":
    train()