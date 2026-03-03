"""
声纹注册脚本
将用户语音转换为声纹嵌入向量并存储
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import json
import yaml
import argparse
from speechbrain.inference.speaker import SpeakerRecognition

from utils.audio_processor import process_audio, load_config


class VoiceprintEnroller:
    """声纹注册器"""
    
    def __init__(self, config_path: str = "config/config.yaml", checkpoint_path: str = None):
        self.config = load_config(config_path)
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        
        # 加载模型
        if checkpoint_path and os.path.exists(checkpoint_path):
            # 加载训练好的模型
            print(f"加载训练模型: {checkpoint_path}")
            self.verification = SpeakerRecognition.from_hparams(
                source=self.config['model']['pretrained_model'],
                savedir="pretrained_models",
                run_opts={"device": self.device}
            )
            self.embedding_model = self.verification.mods["embedding_model"]
            # 加载 checkpoint 权重
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'embedding_model' in checkpoint:
                self.embedding_model.load_state_dict(checkpoint['embedding_model'])
            elif 'model_state_dict' in checkpoint:
                # 兼容不同的保存格式
                state_dict = checkpoint['model_state_dict']
                # 过滤出 embedding_model 的参数
                embedding_state = {k.replace('embedding_model.', ''): v 
                                   for k, v in state_dict.items() 
                                   if k.startswith('embedding_model.')}
                if embedding_state:
                    self.embedding_model.load_state_dict(embedding_state)
            print(f"模型权重已从 {checkpoint_path} 加载")
        else:
            # 加载预训练模型
            print(f"加载预训练模型: {self.config['model']['pretrained_model']}")
            self.verification = SpeakerRecognition.from_hparams(
                source=self.config['model']['pretrained_model'],
                savedir="pretrained_models",
                run_opts={"device": self.device}
            )
            self.embedding_model = self.verification.mods["embedding_model"]
        
        self.embedding_model.eval()
        
        # 声纹数据库
        self.voiceprint_db = {}
        self.db_path = "voiceprint_db.json"
        self._load_db()
    
    def _load_db(self):
        """加载已有声纹数据库"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                # 将列表转回张量
                for user_id, embedding_list in data.items():
                    self.voiceprint_db[user_id] = torch.tensor(embedding_list)
            print(f"已加载 {len(self.voiceprint_db)} 个注册用户")
    
    def _save_db(self):
        """保存声纹数据库"""
        data = {uid: emb.tolist() for uid, emb in self.voiceprint_db.items()}
        with open(self.db_path, 'w') as f:
            json.dump(data, f)
    
    def enroll(self, user_id: str, audio_path: str) -> dict:
        """
        注册声纹
        
        Args:
            user_id: 用户ID
            audio_path: 音频文件路径
            
        Returns:
            注册结果字典
        """
        # 1. 加载音频
        waveform, sr = self._load_audio(audio_path)
        if waveform is None:
            return {
                'success': False,
                'message': '音频加载失败',
                'user_id': user_id
            }
        
        # 2. 检查时长
        min_duration = self.config['audio'].get('min_duration', 3.0)
        duration = waveform.shape[0] / sr
        if duration < min_duration:
            return {
                'success': False,
                'message': f"音频时长 {duration:.2f}s 小于最小要求 {min_duration}s",
                'user_id': user_id
            }
        
        # 3. 提取嵌入向量（使用 encode_batch 方法）
        with torch.no_grad():
            waveform = waveform.to(self.device)
            # encode_batch 输入格式: [batch, time]
            embedding = self.verification.encode_batch(waveform.unsqueeze(0))
            embedding = embedding.squeeze().cpu()
        
        # 4. 存储到数据库
        self.voiceprint_db[user_id] = embedding
        self._save_db()
        
        return {
            'success': True,
            'message': f"用户 {user_id} 注册成功",
            'user_id': user_id,
            'embedding_dim': embedding.shape[0]
        }
    
    def _load_audio(self, audio_path: str):
        """加载音频文件"""
        import torchaudio
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样
            target_sr = self.config['audio']['sample_rate']
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            
            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform.squeeze(0), target_sr
        except Exception as e:
            print(f"音频加载失败: {e}")
            return None, 0
    
    def get_registered_users(self):
        """获取已注册用户列表"""
        return list(self.voiceprint_db.keys())
    
    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        if user_id in self.voiceprint_db:
            del self.voiceprint_db[user_id]
            self._save_db()
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description='声纹注册')
    parser.add_argument('--user_id', type=str, required=True, help='用户ID')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='训练模型路径（如 checkpoints/best_model.pt）')
    
    args = parser.parse_args()
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在: {args.audio}")
        return
    
    # 注册
    enroller = VoiceprintEnroller(config_path=args.config, checkpoint_path=args.checkpoint)
    result = enroller.enroll(args.user_id, args.audio)
    
    if result['success']:
        print(f"注册成功!")
        print(f"用户ID: {result['user_id']}")
        print(f"嵌入维度: {result['embedding_dim']}")
    else:
        print(f"注册失败: {result['message']}")


if __name__ == "__main__":
    main()
