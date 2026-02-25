"""
模型微调训练脚本
基于SpeechBrain框架微调ECAPA-TDNN模型
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
            'audio_path': audio_path
        }


def collate_fn(batch):
    """自定义批处理函数"""
    waveforms = [item['waveform'] for item in batch]
    speaker_ids = torch.stack([item['speaker_id'] for item in batch])
    audio_paths = [item['audio_path'] for item in batch]
    
    # 填充到相同长度
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = torch.zeros(len(batch), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w
    
    return {
        'waveform': padded_waveforms,
        'speaker_id': speaker_ids,
        'audio_path': audio_paths
    }


def train(config_path: str = "config/config.yaml"):
    """训练主函数"""
    
    # 加载配置
    config = load_config(config_path)
    model_cfg = config['model']
    training_cfg = config['training']
    data_cfg = config['data']
    audio_cfg = config['audio']
    
    device = torch.device(training_cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    print(f"加载预训练模型: {model_cfg['pretrained_model']}")
    verification = SpeakerRecognition.from_hparams(
        source=model_cfg['pretrained_model'],
        savedir="pretrained_models",
        run_opts={"device": device}
    )
    
    # 获取模型和分类器
    embedding_model = verification.mods["embedding_model"]
    
    # 创建数据集
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
    
    # 获取说话人数量
    num_speakers = len(train_dataset.speaker_ids)
    print(f"说话人数量: {num_speakers}")
    
    # 创建分类器头
    classifier = nn.Linear(model_cfg['embedding_dim'], num_speakers).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 只优化分类器参数
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay']
    )
    
    # 创建检查点目录
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    
    # 训练循环
    for epoch in range(training_cfg['epochs']):
        embedding_model.eval()
        classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_cfg['epochs']}")
        
        for batch in pbar:
            waveforms = batch['waveform'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            
            # 提取嵌入向量
            with torch.no_grad():
                embeddings = embedding_model(waveforms)
            
            # 分类
            logits = classifier(embeddings)
            
            # 计算损失
            loss = criterion(logits, speaker_ids)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == speaker_ids).sum().item()
            total += speaker_ids.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config['output']['checkpoint_dir'],
                f"classifier_epoch_{epoch + 1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'speaker_to_idx': train_dataset.speaker_to_idx
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(config['output']['checkpoint_dir'], "classifier_final.pt")
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'speaker_to_idx': train_dataset.speaker_to_idx,
        'num_speakers': num_speakers
    }, final_path)
    print(f"最终模型已保存: {final_path}")
    
    return classifier, train_dataset.speaker_to_idx


if __name__ == "__main__":
    train()
