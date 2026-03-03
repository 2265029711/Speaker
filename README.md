# 声纹识别系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-orange.svg)](https://pytorch.org/)
[![SpeechBrain 1.0](https://img.shields.io/badge/SpeechBrain-1.0-green.svg)](https://speechbrain.github.io/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)

基于 ECAPA-TDNN 的声纹识别系统，支持说话人验证、辨认及模型微调。

## 功能特性

- **声纹注册** - 将用户语音转换为 192 维嵌入向量并存储
- **1:1 认证** - 验证待测语音是否属于指定用户
- **1:N 辨认** - 从注册库中识别说话人身份
- **模型微调** - 支持 AAM-Softmax 损失函数微调
- **早停机制** - 自动监控验证集 EER，防止过拟合
- **分层采样** - 训练期间正负样本均衡采样

## 环境依赖

- Python 3.11+
- PyTorch 2.10.0+ (CUDA 13.0)
- SpeechBrain 1.0.3

## 安装方法

```bash
# PyTorch (CUDA 13.0)
pip install torch==2.10.0+cu130 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# SpeechBrain
pip install speechbrain==1.0.3

# 其他依赖
pip install numpy pyyaml scipy soundfile matplotlib seaborn scikit-learn pandas tqdm
```

## 快速开始

```bash
# 1. 训练模型（可选，可直接使用预训练模型）
python scripts/train.py

# 2. 注册声纹（使用训练后的模型）
python scripts/enroll.py --user_id user_001 --audio path/to/voice.wav --checkpoint checkpoints/classifier_final.pt

# 3. 身份认证
python scripts/verify.py --user_id user_001 --audio path/to/test.wav --checkpoint checkpoints/classifier_final.pt

# 4. 说话人辨认
python scripts/verify.py --audio path/to/test.wav --identify --checkpoint checkpoints/classifier_final.pt
```

## 项目结构

```
Speaker/
├── config/
│   └── config.yaml          # 配置文件
├── data/
│   ├── train/               # 训练集音频
│   ├── valid/               # 验证集音频
│   ├── train.csv            # 训练集标签
│   └── valid.csv            # 验证集标签
├── pretrained_models/       # 预训练模型
├── checkpoints/             # 训练检查点
├── logs/                    # 训练日志
├── scripts/
│   ├── train.py             # 训练脚本
│   ├── enroll.py            # 声纹注册
│   ├── verify.py            # 身份认证
│   └── evaluate_model.py    # 模型评估
└── utils/
    ├── audio_processor.py   # 音频处理
    ├── losses.py            # 损失函数
    ├── scorer.py            # 评分函数
    └── visualization.py     # 可视化工具
```

## 数据格式

**目录结构：**
```
data/
├── train/
│   ├── speaker_001/
│   │   ├── audio_01.wav
│   │   └── audio_02.wav
│   └── speaker_002/
│       └── audio_01.wav
└── train.csv
```

**CSV 格式：**
```csv
speaker_id,audio_path
speaker_001,speaker_001/audio_01.wav
speaker_001,speaker_001/audio_02.wav
speaker_002,speaker_002/audio_01.wav
```

## 配置说明

主要参数说明（`config/config.yaml`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `audio.sample_rate` | 采样率 | 16000 |
| `model.embedding_dim` | 嵌入维度 | 192 |
| `training.batch_size` | 批大小 | 8 |
| `training.learning_rate` | 学习率 | 0.001 |
| `training.epochs` | 最大训练轮数 | 30 |
| `training.loss_type` | 损失函数 | aam_softmax |
| `training.aam_s` | AAM 缩放因子 | 30.0 |
| `training.aam_m` | AAM 角度边界 | 0.2 |
| `training.dropout` | Dropout 概率 | 0.1 |
| `training.early_stopping.patience` | 早停容忍轮数 | 5 |
| `verification.threshold` | 认证阈值 | 0.5 |

## 模型说明

- **Backbone**: ECAPA-TDNN (Embedded Attentive Channel and Pattern Aggregation)
- **Pretrained**: [LanceaKing/spkrec-ecapa-cnceleb](https://huggingface.co/LanceaKing/spkrec-ecapa-cnceleb)
- **Loss Function**: AAM-Softmax (Additive Angular Margin)
- **Embedding Dimension**: 192

## 性能指标

| 指标 | 数值 |
|------|------|
| EER | 0.12% |
| AUC | 1.0000 |
| Min DCF | 0.0012 |

## 文档

详细使用说明请参考 [USAGE.md](USAGE.md)

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。