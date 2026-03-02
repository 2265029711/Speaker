"""
损失函数模块
包含 AAM-Softmax 等说话人识别专用损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (AAM-Softmax / ArcFace)
    
    论文: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698
    
    公式: L = -log(exp(s * cos(θ_yi + m)) / (exp(s * cos(θ_yi + m)) + Σ_j≠yi exp(s * cos(θ_j))))
    
    Args:
        in_features: 输入特征维度（嵌入向量维度）
        num_classes: 类别数量（说话人数量）
        s: 缩放因子，通常为 30
        m: 角度边界，通常为 0.2（弧度）
        easy_margin: 是否使用简单边界模式
    """
    
    def __init__(
        self, 
        in_features: int, 
        num_classes: int, 
        s: float = 30.0, 
        m: float = 0.2,
        easy_margin: bool = False
    ):
        super(AAMSoftmax, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        # 权重参数
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算 cos(m) 和 sin(m) 用于角度计算
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # 阈值，用于限制角度范围
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, in_features)
            labels: 标签 (batch_size,)
            
        Returns:
            loss: 标量损失值
        """
        # 归一化权重和特征
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # (num_classes, in_features)
        x_norm = F.normalize(x, p=2, dim=1)  # (batch_size, in_features)
        
        # 计算余弦相似度 cos(θ)
        cosine = F.linear(x_norm, weight_norm)  # (batch_size, num_classes)
        
        # 计算 sin(θ) = sqrt(1 - cos²(θ))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m  # (batch_size, num_classes)
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 当 θ + m > π 时，使用 cos(θ) - sin(π - m)*m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 创建 one-hot 标签
        one_hot = torch.zeros_like(cosine)  # (batch_size, num_classes)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 对目标类别应用角度边界
        # 对于目标类别 y: output = cos(θ_y + m)
        # 对于其他类别 j: output = cos(θ_j)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 缩放
        output = output * self.s
        
        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        
        return loss
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取分类 logits（用于推理）
        
        Args:
            x: 输入特征 (batch_size, in_features)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        cosine = F.linear(x_norm, weight_norm)
        logits = cosine * self.s
        return logits


class SubCenterAAMSoftmax(nn.Module):
    """
    Sub-center AAM-Softmax
    
    论文: Large-Scale Contrastive Learning of Phrase Representations
    每个 类有 K 个子中心，可以提高对噪声标签的鲁棒性
    
    Args:
        in_features: 输入特征维度
        num_classes: 类别数量
        K: 每个类别的子中心数量
        s: 缩放因子
        m: 角度边界
    """
    
    def __init__(
        self, 
        in_features: int, 
        num_classes: int, 
        K: int = 3,
        s: float = 30.0, 
        m: float = 0.2
    ):
        super(SubCenterAAMSoftmax, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.K = K
        self.s = s
        self.m = m
        
        # 权重参数: (num_classes * K, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 归一化
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        
        # 计算所有子中心的余弦相似度
        cosine_all = F.linear(x_norm, weight_norm)  # (batch_size, num_classes * K)
        
        # 重塑为 (batch_size, num_classes, K)
        cosine_all = cosine_all.view(-1, self.num_classes, self.K)
        
        # 对每个类别的 K 个子中心取最大值
        cosine, _ = torch.max(cosine_all, dim=2)  # (batch_size, num_classes)
        
        # 应用角度边界
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # one-hot 编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = F.cross_entropy(output, labels)
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数：AAM-Softmax + Softmax
    
    用于同时优化角度边界和普通分类
    
    Args:
        in_features: 输入特征维度
        num_classes: 类别数量
        aam_weight: AAM-Softmax 的权重
        s: AAM-Softmax 缩放因子
        m: AAM-Softmax 角度边界
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        aam_weight: float = 0.9,
        s: float = 30.0,
        m: float = 0.2
    ):
        super(CombinedLoss, self).__init__()
        self.aam_weight = aam_weight
        self.softmax_weight = 1.0 - aam_weight
        
        self.aam_softmax = AAMSoftmax(in_features, num_classes, s, m)
        self.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        aam_loss = self.aam_softmax(x, labels)
        softmax_loss = F.cross_entropy(self.linear(x), labels)
        
        return self.aam_weight * aam_loss + self.softmax_weight * softmax_loss
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 AAM-Softmax 的 logits
        return self.aam_softmax.get_logits(x)


def get_loss_function(
    loss_type: str,
    in_features: int,
    num_classes: int,
    s: float = 30.0,
    m: float = 0.2,
    K: int = 3,
    aam_weight: float = 0.9
) -> nn.Module:
    """
    工厂函数：根据类型创建损失函数
    
    Args:
        loss_type: 损失函数类型，可选 'softmax', 'aam_softmax', 'subcenter_aam', 'combined'
        in_features: 输入特征维度
        num_classes: 类别数量
        s: 缩放因子
        m: 角度边界
        K: 子中心数量（仅用于 subcenter_aam）
        aam_weight: AAM 权重（仅用于 combined）
        
    Returns:
        损失函数模块
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'softmax':
        return nn.Linear(in_features, num_classes)
    
    elif loss_type == 'aam_softmax' or loss_type == 'aam':
        return AAMSoftmax(in_features, num_classes, s, m)
    
    elif loss_type == 'subcenter_aam':
        return SubCenterAAMSoftmax(in_features, num_classes, K, s, m)
    
    elif loss_type == 'combined':
        return CombinedLoss(in_features, num_classes, aam_weight, s, m)
    
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}，可选: softmax, aam_softmax, subcenter_aam, combined")
