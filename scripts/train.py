"""
模型微调训练脚本
基于SpeechBrain框架微调ECAPA-TDNN模型
支持：多指标评估、训练可视化、预训练模型对比
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
from utils.losses import get_loss_function, AAMSoftmax


class EarlyStopping:
    """早停机制：支持混合指标监控，准确率优先策略"""
    
    def __init__(self, patience=5, min_delta=0.001, monitor='val_eer', mode='min',
                 use_hybrid=False, accuracy_weight=0.7, eer_weight=0.3,
                 accuracy_threshold=0.90):
        """
        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            monitor: 监控指标名称（单指标模式）
            mode: 'min' 指标越小越好，'max' 指标越大越好
            use_hybrid: 是否使用混合指标模式
            accuracy_weight: 准确率权重（混合模式）
            eer_weight: EER权重（混合模式）
            accuracy_threshold: 准确率阈值，低于此值时优先提升准确率
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.use_hybrid = use_hybrid
        self.accuracy_weight = accuracy_weight
        self.eer_weight = eer_weight
        self.accuracy_threshold = accuracy_threshold
        
        self.counter = 0
        self.best_score = None
        self.best_accuracy = 0.0
        self.best_eer = float('inf')
        self.early_stop = False
        self.best_epoch = 0
        
    def _compute_hybrid_score(self, accuracy, eer):
        """计算混合得分：准确率优先策略"""
        # 准确率越高越好，EER越低越好
        # 综合得分 = accuracy_weight * accuracy - eer_weight * eer
        return self.accuracy_weight * accuracy - self.eer_weight * eer
    
    def __call__(self, metrics, current_epoch):
        """
        检查是否应该早停
        
        Args:
            metrics: 字典，包含 'accuracy' 和 'eer'
                     - accuracy: 训练或验证准确率
                     - eer: 验证EER
            current_epoch: 当前轮数
            
        Returns:
            bool: True 表示应该停止训练
        """
        accuracy = metrics.get('accuracy', 0.0)
        eer = metrics.get('eer', 1.0)
        
        if self.use_hybrid:
            # 混合指标模式：准确率优先
            current_score = self._compute_hybrid_score(accuracy, eer)
            
            if self.best_score is None:
                self.best_score = current_score
                self.best_accuracy = accuracy
                self.best_eer = eer
                self.best_epoch = current_epoch
                return False
            
            # 准确率优先策略：
            # 1. 如果准确率低于阈值，优先提升准确率
            # 2. 如果准确率高于阈值，综合考虑准确率和EER
            
            accuracy_improved = accuracy > self.best_accuracy + self.min_delta
            eer_improved = eer < self.best_eer - self.min_delta
            
            # 准确率低于阈值时，只要准确率有提升就不计数
            if accuracy < self.accuracy_threshold:
                if accuracy_improved:
                    self.counter = 0
                    self.best_score = current_score
                    self.best_accuracy = accuracy
                    self.best_eer = min(self.best_eer, eer)
                    self.best_epoch = current_epoch
                    print(f"  混合指标更新: accuracy={accuracy:.4f}↑, eer={eer:.4f}, score={current_score:.4f}")
                    return False
            else:
                # 准确率达到阈值后，使用综合评分
                if current_score > self.best_score + self.min_delta * 0.1:
                    self.counter = 0
                    self.best_score = current_score
                    self.best_accuracy = accuracy
                    self.best_eer = eer
                    self.best_epoch = current_epoch
                    print(f"  混合指标更新: accuracy={accuracy:.4f}, eer={eer:.4f}, score={current_score:.4f}↑")
                    return False
            
            # 没有改善
            self.counter += 1
            print(f"  早停计数: {self.counter}/{self.patience} "
                  f"(最佳 accuracy={self.best_accuracy:.4f}, eer={self.best_eer:.4f}, "
                  f"score={self.best_score:.4f} @ Epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n早停触发！最佳模型在 Epoch {self.best_epoch}，"
                      f"accuracy={self.best_accuracy:.4f}, eer={self.best_eer:.4f}")
                return True
            
            return False
        else:
            # 单指标模式（兼容原有逻辑）
            current_value = metrics.get(self.monitor.replace('val_', ''), 0.0)
            
            if self.best_score is None:
                self.best_score = current_value
                self.best_epoch = current_epoch
                return False
            
            if self.mode == 'min':
                improved = current_value < self.best_score - self.min_delta
            else:
                improved = current_value > self.best_score + self.min_delta
            
            if improved:
                self.best_score = current_value
                self.best_epoch = current_epoch
                self.counter = 0
            else:
                self.counter += 1
                print(f"  早停计数: {self.counter}/{self.patience} (最佳 {self.monitor}: {self.best_score:.4f} @ Epoch {self.best_epoch})")
                
                if self.counter >= self.patience:
                    self.early_stop = True
                    print(f"\n早停触发！最佳模型在 Epoch {self.best_epoch}，{self.monitor} = {self.best_score:.4f}")
                    return True
            
            return False


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
    verification.eval()  # 确保评估模式
    
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
        'min_dcf': min_dcf if min_dcf else 0.0
    }


def compute_verification_eer(embeddings, speaker_ids, max_pairs=5000):
    """
    计算验证任务上的EER（使用分层平衡采样）
    
    改进：确保正负样本平衡，每个说话人都有代表性
    
    Args:
        embeddings: 所有音频的嵌入向量 (N, D)
        speaker_ids: 对应的说话人ID列表 (N,)
        max_pairs: 最大采样对数，避免计算量过大
        
    Returns:
        (eer, threshold, scores, labels)
    """
    import numpy as np
    from collections import defaultdict
    
    embeddings = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
    n = len(speaker_ids)
    
    # 按说话人分组索引
    speaker_to_indices = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(idx)
    
    unique_speakers = list(speaker_to_indices.keys())
    num_speakers = len(unique_speakers)
    
    print(f"  验证集样本数: {n}, 说话人数: {num_speakers}")
    
    # 生成正样本对（同一说话人）和负样本对（不同说话人）
    positive_pairs = []
    negative_pairs = []
    
    # 正样本对：每个说话人内部的组合
    for spk, indices in speaker_to_indices.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                positive_pairs.append((indices[i], indices[j]))
    
    # 负样本对：不同说话人之间的组合
    for i in range(num_speakers):
        for j in range(i + 1, num_speakers):
            spk_i = unique_speakers[i]
            spk_j = unique_speakers[j]
            # 每对说话人之间采样若干对
            indices_i = speaker_to_indices[spk_i]
            indices_j = speaker_to_indices[spk_j]
            # 每对说话人最多采样 min(3, len_i * len_j) 对
            max_per_pair = min(3, len(indices_i) * len(indices_j))
            for _ in range(max_per_pair):
                idx_i = np.random.choice(indices_i)
                idx_j = np.random.choice(indices_j)
                negative_pairs.append((idx_i, idx_j))
    
    total_positive = len(positive_pairs)
    total_negative = len(negative_pairs)
    
    print(f"  正样本对总数: {total_positive}, 负样本对总数: {total_negative}")
    
    # 平衡采样：正负样本各采样 max_pairs // 2
    np.random.seed(42)  # 保证可复现
    half_pairs = max_pairs // 2
    
    if total_positive > half_pairs:
        sampled_positive = [positive_pairs[i] for i in np.random.choice(total_positive, half_pairs, replace=False)]
    else:
        sampled_positive = positive_pairs
    
    if total_negative > half_pairs:
        sampled_negative = [negative_pairs[i] for i in np.random.choice(total_negative, half_pairs, replace=False)]
    else:
        sampled_negative = negative_pairs[:half_pairs]
    
    pairs = sampled_positive + sampled_negative
    labels = [1] * len(sampled_positive) + [0] * len(sampled_negative)
    
    print(f"  采样后 - 正样本对: {len(sampled_positive)}, 负样本对: {len(sampled_negative)}")
    
    # 批量计算余弦相似度
    scores = []
    for i, j in pairs:
        e1 = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
        e2 = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
        sim = np.dot(e1, e2)
        scores.append(sim)
    
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
    verification.eval()  # 确保评估模式
    
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


def get_state_dict(model):
    """获取模型的state_dict，兼容DataParallel包装"""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def train(config_path: str = "config/config.yaml"):
    """训练主函数"""
    
    # 加载配置
    config = load_config(config_path)
    model_cfg = config['model']
    training_cfg = config['training']
    data_cfg = config['data']
    audio_cfg = config['audio']
    output_cfg = config['output']
    
    # 多卡训练检测与初始化
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"检测到 {num_gpus} 张GPU，启用多卡训练")
            device = torch.device("cuda")
            use_multi_gpu = True
            # 打印GPU信息
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print(f"检测到 1 张GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
            use_multi_gpu = False
    else:
        print("未检测到GPU，使用CPU训练")
        device = torch.device("cpu")
        use_multi_gpu = False
    
    print(f"训练设备: {device}, 多卡模式: {use_multi_gpu}")
    
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
    
    # 多卡训练：用DataParallel包装embedding_model
    embedding_model = verification.mods["embedding_model"]
    if use_multi_gpu:
        embedding_model = nn.DataParallel(embedding_model)
        verification.mods["embedding_model"] = embedding_model
        print(f"Embedding模型已用DataParallel包装，使用 {num_gpus} 张GPU")
    
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
    
    # 验证数据集已禁用（使用单独的验证脚本评估）
    valid_loader = None
    print("验证集评估已禁用，请使用 evaluate_model.py 进行验证")
    
    # 获取说话人数量
    num_speakers = len(train_dataset.speaker_ids)
    idx_to_speaker = {idx: sid for sid, idx in train_dataset.speaker_to_idx.items()}
    print(f"说话人数量: {num_speakers}")
    
    # 获取损失函数配置
    loss_type = training_cfg.get('loss_type', 'softmax')
    finetune_backbone = training_cfg.get('finetune_backbone', False)
    backbone_lr_scale = training_cfg.get('backbone_lr_scale', 0.1)  # backbone 学习率缩放因子
    aam_s = training_cfg.get('aam_s', 30.0)  # AAM-Softmax 缩放因子
    aam_m = training_cfg.get('aam_m', 0.2)   # AAM-Softmax 角度边界
    dropout = training_cfg.get('dropout', 0.0)  # Dropout 概率
    
    print(f"\n训练配置:")
    print(f"  损失函数类型: {loss_type}")
    print(f"  微调 backbone: {finetune_backbone}")
    print(f"  Dropout: {dropout}")
    print(f"  Weight Decay: {training_cfg.get('weight_decay', 0)}")
    if finetune_backbone:
        print(f"  Backbone 学习率缩放: {backbone_lr_scale}")
    if loss_type in ['aam_softmax', 'aam', 'subcenter_aam', 'combined']:
        print(f"  AAM-Softmax 参数: s={aam_s}, m={aam_m}")
    
    # 创建Dropout层（防止过拟合）
    dropout_layer = nn.Dropout(p=dropout).to(device) if dropout > 0 else None
    
    # 创建损失函数/分类器
    if loss_type == 'softmax':
        # 普通的线性分类器 + CrossEntropy
        classifier = nn.Linear(model_cfg['embedding_dim'], num_speakers).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        # AAM-Softmax 或其他损失函数
        classifier = get_loss_function(
            loss_type=loss_type,
            in_features=model_cfg['embedding_dim'],
            num_classes=num_speakers,
            s=aam_s,
            m=aam_m
        ).to(device)
        criterion = None  # AAM-Softmax 内部包含损失计算
    
    # 注意：分类器不使用 DataParallel 包装
    # 原因：AAMSoftmax 返回标量 loss，DataParallel 会将其收集为 tensor 导致 backward 失败
    # 分类器计算量小，在单 GPU 上计算即可
    
    # 收集需要优化的参数
    param_groups = []
    
    if finetune_backbone:
        # 解冻 backbone 参数
        print("\n解冻 ECAPA-TDNN backbone 参数...")
        backbone_params = []
        total_backbone_params = 0
        
        for name, param in verification.mods.named_parameters():
            param.requires_grad = True
            backbone_params.append(param)
            total_backbone_params += param.numel()
        
        print(f"  Backbone 总参数量: {total_backbone_params:,}")
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': float(training_cfg['learning_rate']) * backbone_lr_scale,
                'name': 'backbone'
            })
    
    # 添加分类器/损失函数参数组
    if loss_type == 'softmax':
        classifier_params = classifier.parameters()
    else:
        classifier_params = classifier.parameters()
    
    param_groups.append({
        'params': classifier_params,
        'lr': float(training_cfg['learning_rate']),
        'name': 'classifier'
    })
    
    # 创建优化器
    optimizer = optim.Adam(
        param_groups,
        weight_decay=float(training_cfg['weight_decay'])
    )
    
    # 打印参数统计
    total_params = sum(p.numel() for group in param_groups for p in group['params'])
    trainable_params = sum(p.numel() for group in param_groups for p in group['params'] if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': []
    }
    
    # 初始化早停机制
    early_stopping_cfg = training_cfg.get('early_stopping', {})
    early_stopping = None
    if early_stopping_cfg.get('enabled', False):
        early_stopping = EarlyStopping(
            patience=int(early_stopping_cfg.get('patience', 5)),
            min_delta=float(early_stopping_cfg.get('min_delta', 0.001)),
            monitor=early_stopping_cfg.get('monitor', 'val_eer'),
            mode='min',
            use_hybrid=early_stopping_cfg.get('use_hybrid', False),
            accuracy_weight=float(early_stopping_cfg.get('accuracy_weight', 0.7)),
            eer_weight=float(early_stopping_cfg.get('eer_weight', 0.3)),
            accuracy_threshold=float(early_stopping_cfg.get('accuracy_threshold', 0.92))
        )
        if early_stopping_cfg.get('use_hybrid', False):
            print(f"早停已启用(混合指标模式): patience={early_stopping_cfg.get('patience', 5)}, "
                  f"accuracy_weight={early_stopping_cfg.get('accuracy_weight', 0.7)}, "
                  f"eer_weight={early_stopping_cfg.get('eer_weight', 0.3)}, "
                  f"accuracy_threshold={early_stopping_cfg.get('accuracy_threshold', 0.92)}")
        else:
            print(f"早停已启用: patience={early_stopping_cfg.get('patience', 5)}, "
                  f"monitor={early_stopping_cfg.get('monitor', 'val_eer')}")
    
    # 训练循环
    print("\n=== 开始训练 ===")
    best_val_f1 = 0
    best_epoch = 0
    best_val_eer = float('inf')  # 跟踪最佳 EER
    best_hybrid_score = -float('inf')  # 混合指标得分（准确率优先）
    
    # 获取混合指标权重
    early_stopping_cfg = training_cfg.get('early_stopping', {})
    accuracy_weight = float(early_stopping_cfg.get('accuracy_weight', 0.7))
    eer_weight = float(early_stopping_cfg.get('eer_weight', 0.3))
    
    for epoch in range(training_cfg['epochs']):
        # 设置模型模式
        if finetune_backbone:
            verification.train()  # 微调 backbone 时需要 train 模式
        else:
            verification.eval()   # 冻结 backbone 时使用 eval 模式
        classifier.train()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_cfg['epochs']}")
        
        for batch in pbar:
            waveforms = batch['waveform'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            
            # 提取嵌入向量
            if finetune_backbone:
                # 微调模式下需要梯度，但 BatchNorm 使用固定统计量（避免多卡单样本问题）
                # 临时将 embedding_model 设为 eval 模式（BatchNorm 使用固定统计量）
                # 同时用 enable_grad 确保梯度可以回传
                was_training = embedding_model.training
                embedding_model.eval()  # BatchNorm 使用固定统计量
                with torch.enable_grad():
                    wav_lens = torch.ones(waveforms.size(0), device=device)
                    embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
                if was_training:
                    embedding_model.train()  # 恢复 train 模式（Dropout 等）
            else:
                # 冻结模式下不需要梯度
                with torch.no_grad():
                    wav_lens = torch.ones(waveforms.size(0), device=device)
                    embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
            
            # 应用 Dropout（防止过拟合）
            if dropout_layer is not None:
                embeddings = dropout_layer(embeddings)
            
            # 计算损失
            if loss_type == 'softmax':
                # 普通分类器
                logits = classifier(embeddings)
                loss = criterion(logits, speaker_ids)
            else:
                # AAM-Softmax 等
                loss = classifier(embeddings, speaker_ids)
                # 获取 logits 用于计算准确率
                if hasattr(classifier, 'get_logits'):
                    logits = classifier.get_logits(embeddings)
                else:
                    logits = classifier.get_logits(embeddings) if hasattr(classifier, 'get_logits') else None
            
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
        
        # 保存最佳模型（基于训练准确率）
        if train_metrics['accuracy'] > best_hybrid_score:
            best_hybrid_score = train_metrics['accuracy']
            best_epoch = epoch + 1
            # 保存最佳模型检查点
            best_checkpoint_path = os.path.join(
                output_cfg['checkpoint_dir'],
                "classifier_best.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'classifier_state_dict': get_state_dict(classifier),
                'embedding_model': get_state_dict(verification.mods["embedding_model"]),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'speaker_to_idx': train_dataset.speaker_to_idx,
                'best_accuracy': best_hybrid_score,
                'finetune_backbone': finetune_backbone
            }, best_checkpoint_path)
            print(f"最佳模型已保存: {best_checkpoint_path}")
            print(f"  准确率: {best_hybrid_score:.4f}")
        
        # 每轮保存指标图片
        epoch_metrics = {k: v for k, v in train_metrics.items()}
        
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
                'classifier_state_dict': get_state_dict(classifier),
                'embedding_model': get_state_dict(verification.mods["embedding_model"]),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'speaker_to_idx': train_dataset.speaker_to_idx,
                'finetune_backbone': finetune_backbone
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
        
        # 早停检查（基于训练准确率）
        if early_stopping:
            hybrid_metrics = {
                'accuracy': train_metrics['accuracy'],
                'eer': 0.0  # 无验证EER
            }
            if early_stopping(hybrid_metrics, epoch + 1):
                print(f"\n早停触发于 Epoch {epoch + 1}")
                break
    
    # 保存最终模型
    final_path = os.path.join(output_cfg['checkpoint_dir'], "classifier_final.pt")
    torch.save({
        'classifier_state_dict': get_state_dict(classifier),
        'embedding_model': get_state_dict(verification.mods["embedding_model"]),
        'speaker_to_idx': train_dataset.speaker_to_idx,
        'num_speakers': num_speakers,
        'best_epoch': best_epoch,
        'best_accuracy': best_hybrid_score,
        'finetune_backbone': finetune_backbone
    }, final_path)
    print(f"\n最终模型已保存: {final_path}")
    print(f"最佳模型: Epoch {best_epoch}, 准确率: {best_hybrid_score:.4f}")
    
    # 保存训练趋势图
    plot_training_trends(
        history,
        save_path=os.path.join(metrics_dir, 'training_trends.png')
    )
    
    # 打印性能总结
    print("\n" + "="*50)
    print("=== 训练完成 - 性能总结 ===")
    print("="*50)
    print(f"最佳轮次: Epoch {best_epoch}")
    print(f"最佳准确率: {best_hybrid_score:.4f}")
    print(f"\n指标图片已保存至: {metrics_dir}")
    
    return classifier, train_dataset.speaker_to_idx, history


if __name__ == "__main__":
    train()