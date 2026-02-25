"""
评分工具模块
包含：余弦相似度、EER计算、MinDCF计算
"""

import torch
import numpy as np
from typing import Tuple, List
from sklearn.metrics import roc_curve


def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    计算两个声纹嵌入向量的余弦相似度
    
    Args:
        embedding1: 第一个嵌入向量 [dim]
        embedding2: 第二个嵌入向量 [dim]
        
    Returns:
        余弦相似度得分 [-1, 1]
    """
    # 归一化
    e1 = embedding1 / (torch.norm(embedding1) + 1e-8)
    e2 = embedding2 / (torch.norm(embedding2) + 1e-8)
    
    # 余弦相似度
    similarity = torch.dot(e1, e2)
    return similarity.item()


def compute_eer(scores: List[float], labels: List[int]) -> Tuple[float, float, float]:
    """
    计算等错误率 (Equal Error Rate)
    
    Args:
        scores: 相似度得分列表
        labels: 真实标签列表 (1=同一说话人, 0=不同说话人)
        
    Returns:
        (EER, 阈值, FAR在EER处)
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 计算FAR和FRR
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr  # FRR = 1 - TPR
    
    # 找到EER点 (FAR = FRR)
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    
    return float(eer), float(threshold), float(fpr[eer_idx])


def compute_min_dcf(scores: List[float], 
                    labels: List[int],
                    c_miss: float = 1.0,
                    c_fa: float = 1.0,
                    p_target: float = 0.01) -> Tuple[float, float]:
    """
    计算最小检测代价函数 (Minimum Detection Cost Function)
    
    Args:
        scores: 相似度得分列表
        labels: 真实标签列表
        c_miss: 漏检代价
        c_fa: 误检代价
        p_target: 目标说话人先验概率
        
    Returns:
        (MinDCF, 最优阈值)
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # 计算DCF
    dcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    
    min_idx = np.argmin(dcf)
    min_dcf = dcf[min_idx]
    best_threshold = thresholds[min_idx]
    
    return float(min_dcf), float(best_threshold)


def compute_all_metrics(scores: List[float], labels: List[int]) -> dict:
    """
    计算所有评估指标
    
    Args:
        scores: 相似度得分列表
        labels: 真实标签列表
        
    Returns:
        包含EER、MinDCF等指标的字典
    """
    eer, eer_threshold, far_at_eer = compute_eer(scores, labels)
    min_dcf, dcf_threshold = compute_min_dcf(scores, labels)
    
    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'far_at_eer': far_at_eer,
        'min_dcf': min_dcf,
        'dcf_threshold': dcf_threshold
    }
