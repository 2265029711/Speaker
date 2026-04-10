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

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from speechbrain.inference.speaker import SpeakerRecognition

from utils.audio_processor import load_config
from utils.visualization import (
    plot_epoch_metrics, 
    plot_training_trends, 
    compute_classification_metrics
)
from utils.scorer import compute_eer, compute_min_dcf
from utils.losses import get_loss_function


SpeakerSample = Dict[str, Any]
SpeakerBatch = Dict[str, Any]


class EarlyStopping:
    """早停控制器，支持单指标模式和准确率优先的混合评分模式。"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        monitor: str = 'val_eer',
        mode: str = 'min',
        use_hybrid: bool = False,
        accuracy_weight: float = 0.7,
        eer_weight: float = 0.3,
        accuracy_threshold: float = 0.90,
    ) -> None:
        """
        Args:
            patience: 连续多少个轮次无改进后触发早停。
            min_delta: 认定为“有改进”所需的最小提升幅度。
            monitor: 单指标模式下监控的指标名称。
            mode: 单指标模式的优化方向，`min` 表示越小越好，`max` 表示越大越好。
            use_hybrid: 是否启用准确率优先的混合评分策略。
            accuracy_weight: 混合模式中准确率的权重。
            eer_weight: 混合模式中 EER 的权重。
            accuracy_threshold: 准确率达到该阈值前，优先只看准确率是否提升。
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
        
    def _compute_hybrid_score(self, accuracy: float, eer: float) -> float:
        """计算混合模式下的评分。"""
        return self.accuracy_weight * accuracy - self.eer_weight * eer
    
    def __call__(self, metrics: Mapping[str, float], current_epoch: int) -> bool:
        """
        根据当前轮次指标判断是否应触发早停。
        
        Args:
            metrics: 当前轮次的指标字典。混合模式下至少包含 `accuracy` 和 `eer`。
            current_epoch: 当前轮次，从 1 开始计数。
        """
        accuracy = metrics.get('accuracy', 0.0)
        eer = metrics.get('eer', 1.0)
        
        if self.use_hybrid:
            current_score = self._compute_hybrid_score(accuracy, eer)
            
            if self.best_score is None:
                self.best_score = current_score
                self.best_accuracy = accuracy
                self.best_eer = eer
                self.best_epoch = current_epoch
                return False
            
            accuracy_improved = accuracy > self.best_accuracy + self.min_delta
            
            # 在准确率未达到阈值前，只要准确率继续提升就重置计数器。
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
                # 达到阈值后，再使用混合评分决定是否视为改进。
                if current_score > self.best_score + self.min_delta * 0.1:
                    self.counter = 0
                    self.best_score = current_score
                    self.best_accuracy = accuracy
                    self.best_eer = eer
                    self.best_epoch = current_epoch
                    print(f"  混合指标更新: accuracy={accuracy:.4f}, eer={eer:.4f}, score={current_score:.4f}↑")
                    return False
            
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
    """从 CSV 加载音频路径与说话人标签，并返回训练所需张量。"""
    
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


def evaluate_model(
    verification: SpeakerRecognition,
    classifier: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    _idx_to_speaker: Mapping[int, str],
) -> Dict[str, float]:
    """
    基于嵌入向量相似度评估说话人验证性能。

    该函数不依赖分类器输出，而是通过嵌入向量之间的相似度计算验证任务指标，
    返回 EER、EER 阈值和 minDCF 等面向验证场景的结果。
    """
    classifier.eval()
    verification.eval()
    
    all_embeddings = []
    all_speaker_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取嵌入向量"):
            waveforms = batch['waveform'].to(device)
            
            wav_lens = torch.ones(waveforms.size(0), device=device)
            embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
            
            all_embeddings.append(embeddings.cpu())
            all_speaker_ids.extend(batch['speaker_id_str'])
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"\n验证集评估:")
    print(f"  总样本数: {len(all_speaker_ids)}")
    print(f"  说话人数: {len(set(all_speaker_ids))}")
    
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


def compute_verification_eer(
    embeddings: torch.Tensor | np.ndarray,
    speaker_ids: Sequence[str],
    max_pairs: int = 5000,
) -> Tuple[float, float, List[float], List[int]]:
    """通过分层平衡采样的验证对估计 EER。"""
    embeddings = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
    n = len(speaker_ids)
    
    speaker_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(idx)
    
    unique_speakers = list(speaker_to_indices.keys())
    num_speakers = len(unique_speakers)
    
    print(f"  验证集样本数: {n}, 说话人数: {num_speakers}")
    
    positive_pairs: List[Tuple[int, int]] = []
    negative_pairs: List[Tuple[int, int]] = []
    
    for spk, indices in speaker_to_indices.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                positive_pairs.append((indices[i], indices[j]))
    
    for i in range(num_speakers):
        for j in range(i + 1, num_speakers):
            spk_i = unique_speakers[i]
            spk_j = unique_speakers[j]
            indices_i = speaker_to_indices[spk_i]
            indices_j = speaker_to_indices[spk_j]
            max_per_pair = min(3, len(indices_i) * len(indices_j))
            for _ in range(max_per_pair):
                idx_i = np.random.choice(indices_i)
                idx_j = np.random.choice(indices_j)
                negative_pairs.append((idx_i, idx_j))
    
    total_positive = len(positive_pairs)
    total_negative = len(negative_pairs)
    
    print(f"  正样本对总数: {total_positive}, 负样本对总数: {total_negative}")
    
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
    
    scores: List[float] = []
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


def evaluate_pretrained_model(
    verification: SpeakerRecognition,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """在不使用分类器头的情况下评估预训练骨干网络。"""
    verification.eval()
    
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


def get_state_dict(model: nn.Module) -> Mapping[str, torch.Tensor]:
    """获取兼容 `DataParallel` 包装的 `state_dict`。"""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def train(config_path: str = "config/config.yaml") -> Tuple[nn.Module, Mapping[str, int], Dict[str, List[float]]]:
    """执行训练流程，并保存检查点、指标及可视化结果。"""
    
    # 加载配置
    config = load_config(config_path)
    model_cfg = config['model']
    training_cfg = config['training']
    data_cfg = config['data']
    audio_cfg = config['audio']
    output_cfg = config['output']
    
    # 检测运行设备，并根据 GPU 数量决定是否启用多卡训练。
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"检测到 {num_gpus} 张GPU，启用多卡训练")
            device = torch.device("cuda")
            use_multi_gpu = True
            # 打印每张 GPU 的名称与显存，便于确认训练环境。
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
        # 使用本地预训练模型目录直接初始化 SpeakerRecognition。
        verification = SpeakerRecognition.from_hparams(
            source=pretrained_path,
            run_opts={"device": device}
        )
        print("本地预训练模型加载完成")
    else:
        # 否则按远程模型源方式加载预训练模型。
        verification = SpeakerRecognition.from_hparams(
            source=pretrained_path,
            savedir="pretrained_models",
            run_opts={"device": device}
        )
    
    # 多卡训练时仅对嵌入模型做 DataParallel 包装。
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
    
    # 验证逻辑统一放在独立评估脚本中，训练脚本仅负责优化与保存检查点。
    print("验证集评估已禁用，请使用 evaluate_model.py 进行验证")
    
    # 获取说话人数量
    num_speakers = len(train_dataset.speaker_ids)
    print(f"说话人数量: {num_speakers}")
    
    # 读取损失函数与微调相关配置。
    loss_type = training_cfg.get('loss_type', 'softmax')
    finetune_backbone = training_cfg.get('finetune_backbone', False)
    backbone_lr_scale = training_cfg.get('backbone_lr_scale', 0.1)  # 骨干网络学习率缩放因子
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
    
    # 仅在配置中显式启用时，才在嵌入向量后追加 Dropout。
    dropout_layer = nn.Dropout(p=dropout).to(device) if dropout > 0 else None
    
    # 创建损失函数/分类器
    if loss_type == 'softmax':
        # 使用线性分类器与交叉熵损失。
        classifier = nn.Linear(model_cfg['embedding_dim'], num_speakers).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        # 使用 AAM-Softmax 或其他带分类头的损失函数。
        classifier = get_loss_function(
            loss_type=loss_type,
            in_features=model_cfg['embedding_dim'],
            num_classes=num_speakers,
            s=aam_s,
            m=aam_m
        ).to(device)
        criterion = None  # AAM-Softmax 内部已封装损失计算。
    
    # 分类器不使用 DataParallel。
    # 对于 AAM 一类损失，分类器会直接返回标量 loss，多卡包装后会让反向传播变复杂，
    # 但收益很小，因此保持单卡更稳妥。
    
    # 组织需要优化的参数组。
    param_groups = []
    
    if finetune_backbone:
        # 解冻骨干网络参数。
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
    
    # 分类器参数统一使用基础学习率。
    classifier_params = classifier.parameters()

    param_groups.append({
        'params': classifier_params,
        'lr': float(training_cfg['learning_rate']),
        'name': 'classifier'
    })
    
    # 创建优化器。
    optimizer = optim.Adam(
        param_groups,
        weight_decay=float(training_cfg['weight_decay'])
    )
    
    # 打印参数规模统计信息。
    total_params = sum(p.numel() for group in param_groups for p in group['params'])
    trainable_params = sum(p.numel() for group in param_groups for p in group['params'] if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    
    # 初始化训练历史记录。
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'train_precision': [],
        'train_recall': []
    }
    
    # 根据配置初始化早停机制。
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
    
    # 进入正式训练循环。
    print("\n=== 开始训练 ===")
    best_epoch = 0
    best_hybrid_score = -float('inf')  # 混合指标得分（准确率优先）
    
    for epoch in range(training_cfg['epochs']):
        # 根据是否微调骨干网络设置模型模式。
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
            
            if finetune_backbone:
                # 微调模式下需要梯度，但 BatchNorm 仍固定统计量，以避免小批量和多卡场景下的不稳定。
                # 临时切换为 eval 模式只影响 BatchNorm/Dropout 行为，不阻断梯度回传。
                was_training = embedding_model.training
                embedding_model.eval()  # 固定 BatchNorm 统计量。
                with torch.enable_grad():
                    wav_lens = torch.ones(waveforms.size(0), device=device)
                    embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
                if was_training:
                    embedding_model.train()  # 恢复训练模式。
            else:
                # 骨干网络冻结时无需计算梯度。
                with torch.no_grad():
                    wav_lens = torch.ones(waveforms.size(0), device=device)
                    embeddings = verification.encode_batch(waveforms, wav_lens).squeeze(1)
            
            # 在嵌入层之后可选地施加 Dropout 以抑制过拟合。
            if dropout_layer is not None:
                embeddings = dropout_layer(embeddings)
            
            # 前向计算损失与预测结果。
            if loss_type == 'softmax':
                logits = classifier(embeddings)
                loss = criterion(logits, speaker_ids)
            else:
                loss = classifier(embeddings, speaker_ids)
                # 对于带角度间隔的分类头，额外取回 logits 用于统计训练准确率。
                if hasattr(classifier, 'get_logits'):
                    logits = classifier.get_logits(embeddings)
                else:
                    logits = classifier.get_logits(embeddings) if hasattr(classifier, 'get_logits') else None
            
            # 执行一次标准的反向传播与参数更新。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新本轮训练统计量。
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(speaker_ids.cpu().numpy())
            
            current_acc = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        # 汇总本轮训练指标。
        train_metrics = compute_classification_metrics(all_predictions, all_labels)
        train_metrics['loss'] = total_loss / len(train_loader)
        
        # 将本轮结果写入历史序列，供后续绘图使用。
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        
        print(f"\nEpoch {epoch + 1} 训练结果:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1: {train_metrics['f1']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        
        # 若训练准确率刷新历史最佳，则覆盖保存最佳检查点。
        if train_metrics['accuracy'] > best_hybrid_score:
            best_hybrid_score = train_metrics['accuracy']
            best_epoch = epoch + 1
            # 保存最佳模型检查点。
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
        
        # 每轮训练结束后保存一次指标快照图。
        epoch_metrics = {k: v for k, v in train_metrics.items()}
        
        plot_epoch_metrics(
            epoch_metrics,
            epoch=epoch + 1,
            save_path=os.path.join(metrics_dir, f'epoch_{epoch + 1}_metrics.png')
        )
        
        # 每 5 轮额外落盘一个阶段性检查点。
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
        
        # 基于当前训练指标执行早停检查。
        if early_stopping:
            hybrid_metrics = {
                'accuracy': train_metrics['accuracy'],
                'eer': 0.0  # 当前训练流程未接入验证 EER，先以 0 占位。
            }
            if early_stopping(hybrid_metrics, epoch + 1):
                print(f"\n早停触发于 Epoch {epoch + 1}")
                break
    
    # 训练结束后保存最终模型。
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
    
    # 训练结束后统一生成训练趋势图。
    plot_training_trends(
        history,
        save_path=os.path.join(metrics_dir, 'training_trends.png')
    )
    
    # 打印训练完成后的摘要信息。
    print("\n" + "="*50)
    print("=== 训练完成 - 性能总结 ===")
    print("="*50)
    print(f"最佳轮次: Epoch {best_epoch}")
    print(f"最佳准确率: {best_hybrid_score:.4f}")
    print(f"\n指标图片已保存至: {metrics_dir}")
    
    return classifier, train_dataset.speaker_to_idx, history


if __name__ == "__main__":
    train()
