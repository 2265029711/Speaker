"""
音频处理工具模块
包含：音频检查、VAD、预加重、特征提取等高可复用功能
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import yaml


class AudioChecker:
    """音频质量检查器"""
    
    def __init__(self, sample_rate: int = 16000, min_duration: float = 3.0):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
    
    def check(self, audio_path: str) -> Tuple[bool, str, Optional[torch.Tensor]]:
        """
        检查音频是否满足要求
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (是否通过, 错误信息, 音频张量)
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # 检查采样率
            if sr != self.sample_rate:
                # 重采样
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 检查时长
            duration = waveform.shape[1] / self.sample_rate
            if duration < self.min_duration:
                return False, f"音频时长 {duration:.2f}s 小于最小要求 {self.min_duration}s", None
            
            return True, "", waveform
            
        except Exception as e:
            return False, f"音频加载失败: {str(e)}", None


class VAD:
    """基于能量的端点检测"""
    
    def __init__(self, 
                 frame_length: int = 400,  # 25ms @ 16kHz
                 frame_shift: int = 160,   # 10ms @ 16kHz
                 energy_threshold: float = 0.01):
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.energy_threshold = energy_threshold
    
    def detect(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        检测并提取有效语音片段
        
        Args:
            waveform: 音频张量 [1, samples]
            
        Returns:
            处理后的音频张量
        """
        waveform = waveform.squeeze(0)  # [samples]
        
        # 计算每帧能量
        num_frames = (waveform.shape[0] - self.frame_length) // self.frame_shift + 1
        energies = []
        
        for i in range(num_frames):
            start = i * self.frame_shift
            end = start + self.frame_length
            frame = waveform[start:end]
            energy = torch.mean(frame ** 2).item()
            energies.append(energy)
        
        energies = torch.tensor(energies)
        
        # 找出有效帧
        valid_frames = energies > self.energy_threshold
        
        if not valid_frames.any():
            return waveform.unsqueeze(0)
        
        # 找到有效区域的边界
        valid_indices = torch.where(valid_frames)[0]
        start_frame = valid_indices[0]
        end_frame = valid_indices[-1]
        
        # 转换为样本索引
        start_sample = start_frame * self.frame_shift
        end_sample = (end_frame + 1) * self.frame_shift
        
        return waveform[start_sample:end_sample].unsqueeze(0)


class Preprocessor:
    """音频预处理器：预加重、分帧加窗"""
    
    def __init__(self, preemphasis: float = 0.97):
        self._preemphasis = preemphasis
    
    def preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        预加重滤波
        
        Args:
            waveform: 音频张量
            
        Returns:
            预加重后的音频
        """
        waveform = waveform.squeeze(0)
        emphasized = torch.zeros_like(waveform)
        emphasized[1:] = waveform[1:] - self._preemphasis * waveform[:-1]
        emphasized[0] = waveform[0]
        return emphasized.unsqueeze(0)


class FeatureExtractor:
    """Fbank特征提取器（集成SpeechBrain）"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 frame_length: int = 25,
                 frame_shift: int = 10,
                 n_fft: int = 512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_fft = n_fft
        
        # 创建Mel声谱图变换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(frame_length * sample_rate / 1000),
            hop_length=int(frame_shift * sample_rate / 1000),
            n_mels=n_mels
        )
    
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        提取Log Mel声谱图特征
        
        Args:
            waveform: 音频张量 [1, samples]
            
        Returns:
            Fbank特征 [n_mels, time]
        """
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-8)
        return log_mel.squeeze(0)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def process_audio(audio_path: str, config: dict = None) -> Tuple[bool, str, Optional[torch.Tensor]]:
    """
    完整的音频处理流水线
    
    Args:
        audio_path: 音频文件路径
        config: 配置字典
        
    Returns:
        (是否成功, 错误信息, 特征张量)
    """
    if config is None:
        config = load_config()
    
    audio_cfg = config['audio']
    
    # 1. 音频检查
    checker = AudioChecker(
        sample_rate=audio_cfg['sample_rate'],
        min_duration=audio_cfg['min_duration']
    )
    success, msg, waveform = checker.check(audio_path)
    if not success:
        return False, msg, None
    
    # 2. VAD
    vad = VAD()
    waveform = vad.detect(waveform)
    
    # 3. 预加重
    preprocessor = Preprocessor(preemphasis=audio_cfg['preemphasis'])
    waveform = preprocessor.preemphasis(waveform)
    
    # 4. 特征提取
    extractor = FeatureExtractor(
        sample_rate=audio_cfg['sample_rate'],
        n_mels=audio_cfg['n_mels'],
        frame_length=audio_cfg['frame_length'],
        frame_shift=audio_cfg['frame_shift'],
        n_fft=audio_cfg['n_fft']
    )
    features = extractor.extract(waveform)
    
    return True, "", features
