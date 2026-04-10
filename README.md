# 声纹识别系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)
[![SpeechBrain 1.0](https://img.shields.io/badge/SpeechBrain-1.0.3-green.svg)](https://speechbrain.github.io/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

基于 ECAPA-TDNN 的声纹识别系统，支持说话人验证、辨认及模型微调。

## 功能特性

- **声纹注册** - 将用户语音转换为 192 维嵌入向量并存储
- **1:1 认证** - 验证待测语音是否属于指定用户
- **1:N 辨认** - 从注册库中识别说话人身份
- **模型微调** - 支持 AAM-Softmax 损失函数微调
- **早停机制** - 自动监控验证集 EER，防止过拟合
- **分层采样** - 训练期间正负样本均衡采样

## 环境依赖

### 系统要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10+ | 编程语言环境 |
| CUDA | 12.4 | GPU 加速 |
| FFmpeg | 7.1.2 | 音频解码、格式转换 |

#### FFmpeg 安装

**Windows:**
```bash
# 方法1: 使用 conda 安装
conda install ffmpeg -c conda-forge

# 方法2: 使用 pip 安装 imageio-ffmpeg（轻量级）
pip install imageio-ffmpeg

# 方法3: 手动安装
# 1. 从 https://www.gyan.dev/ffmpeg/builds/ 下载 ffmpeg-release-essentials.zip
# 2. 解压到任意目录（如 E:\ffmpeg）
# 3. 将 bin 目录添加到系统 PATH 环境变量
```

**Linux:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

**macOS:**
```bash
brew install ffmpeg
```

### 核心依赖

以下为 `jj1` 虚拟环境测试通过的版本：

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.6.0+cu124 | PyTorch 深度学习框架 |
| torchaudio | 2.6.0+cu124 | 音频处理工具 |
| torchcodec | 0.10.0 | 视频音频编解码（调用 FFmpeg） |
| speechbrain | 1.0.3 | 语音处理框架 |
| numpy | 2.3.5 | 数值计算 |
| pandas | 3.0.1 | 数据处理 |
| PyYAML | 6.0.3 | 配置文件解析 |
| scipy | 1.17.1 | 科学计算 |
| matplotlib | 3.10.8 | 可视化绑图 |
| scikit-learn | 1.8.0 | 机器学习工具 |
| tqdm | 4.67.3 | 进度条显示 |
| soundfile | 0.13.1 | 音频文件读写 |
| ffmpeg | 1.4 | FFmpeg Python 绑定 |

## 安装方法

### 1. 创建虚拟环境

默认使用 conda 虚拟环境 `jj1`：

```bash
# 创建虚拟环境
conda create -n jj1 python=3.10

# 激活环境
conda activate jj1
```

### 2. 安装 PyTorch 和 torchaudio

根据你的 CUDA 版本选择对应的安装命令：

```bash
# CUDA 12.4 (推荐)
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# CPU 版本
pip install torch==2.6.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装 torchcodec（FFmpeg 后端）

```bash
pip install torchcodec==0.10.0
```

### 4. 安装 SpeechBrain

```bash
pip install speechbrain==1.0.3
```

### 5. 安装其他依赖

```bash
# 数据处理
pip install numpy==2.3.5 pandas==3.0.1 PyYAML==6.0.3 scipy==1.17.1

# 可视化
pip install matplotlib==3.10.8

# 机器学习工具
pip install scikit-learn==1.8.0

# 进度条
pip install tqdm==4.67.3

# 音频读写
pip install soundfile==0.13.1

# FFmpeg Python 绑定
pip install ffmpeg==1.4
```

### 6. 一键安装（推荐）

使用以下命令一次性安装所有依赖（指定版本）：

```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchcodec==0.10.0 speechbrain==1.0.3 numpy==2.3.5 pandas==3.0.1 PyYAML==6.0.3 scipy==1.17.1 matplotlib==3.10.8 scikit-learn==1.8.0 tqdm==4.67.3 soundfile==0.13.1 ffmpeg==1.4
```

> **注意**: `torch` 和 `torchaudio` 需要从 PyTorch 官方源安装，其他包可从 PyPI 安装。

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