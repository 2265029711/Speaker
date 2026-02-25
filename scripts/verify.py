"""
身份认证脚本
验证待测语音与注册声纹的相似度
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import yaml
import argparse
from speechbrain.inference.speaker import SpeakerRecognition

from utils.audio_processor import process_audio, load_config
from utils.scorer import cosine_similarity


class VoiceprintVerifier:
    """声纹认证器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        
        # 加载预训练模型
        print(f"加载预训练模型: {self.config['model']['pretrained_model']}")
        self.verification = SpeakerRecognition.from_hparams(
            source=self.config['model']['pretrained_model'],
            savedir="pretrained_models",
            run_opts={"device": self.device}
        )
        self.embedding_model = self.verification.mods["embedding_model"]
        self.embedding_model.eval()
        
        # 认证阈值
        self.threshold = self.config['verification']['threshold']
        
        # 声纹数据库
        self.voiceprint_db = {}
        self.db_path = "voiceprint_db.json"
        self._load_db()
    
    def _load_db(self):
        """加载声纹数据库"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                for user_id, embedding_list in data.items():
                    self.voiceprint_db[user_id] = torch.tensor(embedding_list)
            print(f"已加载 {len(self.voiceprint_db)} 个注册用户")
        else:
            print("警告: 声纹数据库不存在，请先注册用户")
    
    def _load_audio(self, audio_path: str):
        """加载音频文件"""
        import torchaudio
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
    
    def verify(self, user_id: str, audio_path: str) -> dict:
        """
        验证身份
        
        Args:
            user_id: 声称的用户ID
            audio_path: 待测音频路径
            
        Returns:
            认证结果字典
        """
        # 1. 检查用户是否存在
        if user_id not in self.voiceprint_db:
            return {
                'success': False,
                'message': f"用户 {user_id} 未注册",
                'user_id': user_id,
                'decision': 'reject'
            }
        
        # 2. 音频处理
        success, msg, features = process_audio(audio_path, self.config)
        if not success:
            return {
                'success': False,
                'message': msg,
                'user_id': user_id,
                'decision': 'reject'
            }
        
        # 3. 提取待测嵌入向量
        with torch.no_grad():
            waveform, sr = self._load_audio(audio_path)
            waveform = waveform.to(self.device)
            test_embedding = self.embedding_model(waveform.unsqueeze(0))
            test_embedding = test_embedding.squeeze().cpu()
        
        # 4. 获取注册嵌入向量
        registered_embedding = self.voiceprint_db[user_id]
        
        # 5. 计算余弦相似度
        similarity = cosine_similarity(test_embedding, registered_embedding)
        
        # 6. 阈值判决
        decision = 'accept' if similarity >= self.threshold else 'reject'
        
        return {
            'success': True,
            'message': "认证完成",
            'user_id': user_id,
            'similarity': similarity,
            'threshold': self.threshold,
            'decision': decision,
            'confidence': self._compute_confidence(similarity)
        }
    
    def _compute_confidence(self, similarity: float) -> str:
        """计算置信度"""
        if similarity >= 0.8:
            return 'very_high'
        elif similarity >= 0.6:
            return 'high'
        elif similarity >= self.threshold:
            return 'medium'
        else:
            return 'low'
    
    def identify(self, audio_path: str) -> dict:
        """
        说话人辨认（1:N识别）
        
        Args:
            audio_path: 待测音频路径
            
        Returns:
            识别结果字典
        """
        if not self.voiceprint_db:
            return {
                'success': False,
                'message': "声纹数据库为空",
                'decision': 'reject'
            }
        
        # 提取待测嵌入向量
        with torch.no_grad():
            waveform, sr = self._load_audio(audio_path)
            waveform = waveform.to(self.device)
            test_embedding = self.embedding_model(waveform.unsqueeze(0))
            test_embedding = test_embedding.squeeze().cpu()
        
        # 与所有注册用户比较
        scores = {}
        for user_id, registered_embedding in self.voiceprint_db.items():
            scores[user_id] = cosine_similarity(test_embedding, registered_embedding)
        
        # 找到最佳匹配
        best_user = max(scores, key=scores.get)
        best_score = scores[best_user]
        
        decision = 'accept' if best_score >= self.threshold else 'reject'
        
        return {
            'success': True,
            'message': "识别完成",
            'best_match': best_user,
            'similarity': best_score,
            'threshold': self.threshold,
            'decision': decision,
            'all_scores': scores
        }


def main():
    parser = argparse.ArgumentParser(description='声纹认证')
    parser.add_argument('--user_id', type=str, help='声称的用户ID（认证模式）')
    parser.add_argument('--audio', type=str, required=True, help='待测音频路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--identify', action='store_true', help='使用辨认模式（1:N）')
    
    args = parser.parse_args()
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在: {args.audio}")
        return
    
    # 创建认证器
    verifier = VoiceprintVerifier(config_path=args.config)
    
    # 执行认证或辨认
    if args.identify:
        # 辨认模式（1:N）
        result = verifier.identify(args.audio)
        if result['success']:
            print(f"识别结果: {result['best_match']}")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"判决: {result['decision']}")
        else:
            print(f"识别失败: {result['message']}")
    else:
        # 认证模式（1:1）
        if not args.user_id:
            print("错误: 认证模式需要提供 --user_id 参数")
            return
        
        result = verifier.verify(args.user_id, args.audio)
        print(f"用户ID: {result['user_id']}")
        print(f"相似度: {result.get('similarity', 'N/A')}")
        print(f"阈值: {result.get('threshold', 'N/A')}")
        print(f"判决: {result['decision']}")
        print(f"置信度: {result.get('confidence', 'N/A')}")


if __name__ == "__main__":
    main()
