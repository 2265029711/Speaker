# 声纹识别系统使用说明

## 1. 环境准备

### 1.1 安装 FFmpeg

FFmpeg 用于音频解码和格式转换，是必需的系统依赖。

**Windows:**
```bash
# 方法1: 使用 conda 安装（推荐）
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

**验证安装:**
```bash
ffmpeg -version
```

### 1.2 创建虚拟环境

默认使用 conda 虚拟环境 `jj1`：

```bash
# 创建虚拟环境（首次使用）
conda create -n jj1 python=3.10

# 激活环境
conda activate jj1
```

### 1.3 安装 Python 依赖

以下为 `jj1` 虚拟环境测试通过的版本：

**步骤 1: 安装 PyTorch 和 torchaudio**

```bash
# CUDA 12.4 (推荐)
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# CPU 版本
pip install torch==2.6.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

**步骤 2: 安装 torchcodec（FFmpeg 后端）**

```bash
pip install torchcodec==0.10.0
```

**步骤 3: 安装 SpeechBrain**

```bash
pip install speechbrain==1.0.3
```

**步骤 4: 安装数据处理依赖**

```bash
pip install numpy==2.3.5 pandas==3.0.1 PyYAML==6.0.3 scipy==1.17.1 soundfile==0.13.1
```

**步骤 5: 安装可视化依赖**

```bash
pip install matplotlib==3.10.8
```

**步骤 6: 安装机器学习工具**

```bash
pip install scikit-learn==1.8.0
```

**步骤 7: 安装进度条工具**

```bash
pip install tqdm==4.67.3
```

**步骤 8: 安装 FFmpeg Python 绑定**

```bash
pip install ffmpeg==1.4
```

### 1.4 一键安装（推荐）

```bash
# 先安装 PyTorch（需从官方源安装）
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 再安装其他依赖
pip install torchcodec==0.10.0 speechbrain==1.0.3 numpy==2.3.5 pandas==3.0.1 PyYAML==6.0.3 scipy==1.17.1 soundfile==0.13.1 matplotlib==3.10.8 scikit-learn==1.8.0 tqdm==4.67.3 ffmpeg==1.4
```

> **注意**: `torch` 和 `torchaudio` 需要从 PyTorch 官方源安装，其他包可从 PyPI 安装。

### 1.5 完整依赖列表（jj1 环境版本）

**系统依赖：**

| 软件 | 版本 | 用途 |
|------|------|------|
| FFmpeg | 7.1.2 | 音频解码、格式转换 |

**Python 依赖：**

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.6.0+cu124 | PyTorch 深度学习框架 |
| torchaudio | 2.6.0+cu124 | 音频加载、重采样、特征提取 |
| torchcodec | 0.10.0 | 视频音频编解码（调用 FFmpeg） |
| speechbrain | 1.0.3 | ECAPA-TDNN 预训练模型、说话人嵌入 |
| numpy | 2.3.5 | 数值计算、数组操作 |
| pandas | 3.0.1 | CSV 数据读取、数据处理 |
| PyYAML | 6.0.3 | YAML 配置文件解析 |
| scipy | 1.17.1 | 科学计算 |
| soundfile | 0.13.1 | 音频文件读写 |
| matplotlib | 3.10.8 | 训练曲线、评估图表绑制 |
| scikit-learn | 1.8.0 | ROC 曲线、F1 分数等评估指标 |
| tqdm | 4.67.3 | 训练进度条显示 |
| ffmpeg | 1.4 | FFmpeg Python 绑定 |

---

## 2. 数据准备

**目录结构示例：**
```
data/
├── train/
│   ├── speaker_001/
│   │   ├── audio_01.wav
│   │   └── audio_02.wav
│   ├── speaker_002/
│   │   └── audio_01.wav
│   └── ...
├── valid/
│   └── ...
├── train.csv
└── valid.csv
```

**CSV格式：**
```csv
speaker_id,audio_path
speaker_001,speaker_001/audio_01.wav
speaker_001,speaker_001/audio_02.wav
speaker_002,speaker_002/audio_01.wav
```

---

## 3. 模型训练（微调）

### 3.1 基本训练命令

```bash
# 使用默认配置训练
python scripts/train.py

# 指定配置文件
python scripts/train.py --config config/config.yaml
```

### 3.2 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | config/config.yaml | 配置文件路径 |

### 3.3 训练配置项（config/config.yaml）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `training.batch_size` | 64 | 批大小 |
| `training.learning_rate` | 0.0003 | 学习率 |
| `training.weight_decay` | 5e-4 | L2 正则化系数 |
| `training.epochs` | 15 | 最大训练轮数 |
| `training.num_workers` | 8 | 数据加载线程数 |
| `training.device` | cuda | 训练设备（cuda/cpu） |
| `training.loss_type` | aam_softmax | 损失函数类型 |
| `training.aam_s` | 30.0 | AAM-Softmax 缩放因子 |
| `training.aam_m` | 0.2 | AAM-Softmax 角度边界 |
| `training.finetune_backbone` | true | 是否微调 backbone |
| `training.backbone_lr_scale` | 0.1 | backbone 学习率缩放 |
| `training.dropout` | 0.3 | Dropout 概率 |
| `training.early_stopping.enabled` | true | 是否启用早停 |
| `training.early_stopping.patience` | 5 | 早停容忍轮数 |
| `training.early_stopping.use_hybrid` | true | 使用混合指标模式 |
| `training.early_stopping.accuracy_threshold` | 0.92 | 准确率阈值 |

### 3.4 多卡训练

系统自动检测 GPU 数量，多卡时自动启用 `DataParallel` 并行训练：

```
检测到 10 张GPU，启用多卡训练
  GPU 0: NVIDIA GeForce RTX 2080 Ti (11.0 GB)
  GPU 1: NVIDIA GeForce RTX 2080 Ti (11.0 GB)
  ...
训练设备: cuda, 多卡模式: True
```

### 3.5 训练输出

训练完成后，模型保存在 `checkpoints/` 目录：

| 文件 | 说明 |
|------|------|
| `classifier_best.pt` | 最佳模型（训练准确率最高） |
| `classifier_final.pt` | 最终模型 |
| `classifier_epoch_N.pt` | 每 5 轮保存的检查点 |

训练日志和可视化保存在 `logs/metrics/` 目录：

| 文件 | 说明 |
|------|------|
| `epoch_N_metrics.png` | 每轮训练指标图 |
| `training_trends.png` | 训练趋势图（Loss、Accuracy、F1） |

### 3.6 训练示例输出

```
=== 开始训练 ===
Epoch 1/15: 100%|██████████| 1250/1250 [05:23<00:00,  3.87it/s, loss=2.3456, acc=0.8521]

Epoch 1 训练结果:
  Loss: 2.3456
  Accuracy: 0.8521
  F1: 0.8498
  Precision: 0.8532
  Recall: 0.8521
最佳模型已保存: checkpoints/classifier_best.pt
  准确率: 0.8521
...
早停触发于 Epoch 8

=== 训练完成 - 性能总结 ===
最佳轮次: Epoch 6
最佳准确率: 0.9234
```

---

## 4. 模型验证

### 4.1 验证模式说明

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `pairs` | 验证对准确率评估 | 快速评估模型在音频对上的表现 |
| `classification` | 分类准确率评估 | 评估说话人分类性能 |
| `verification` | 说话人验证 EER 评估 | 评估验证系统性能（EER、minDCF） |
| `both` | 综合评估 | 全面评估所有指标 |

### 4.2 验证对评估（推荐）

```bash
# 基本验证对评估
python scripts/evaluate_model.py --mode pairs

# 指定验证对文件和阈值
python scripts/evaluate_model.py --mode pairs \
    --pairs_csv data/valid_pairs.csv \
    --threshold 0.5 \
    --checkpoint checkpoints/classifier_final.pt

# 指定输出目录
python scripts/evaluate_model.py --mode pairs \
    --output logs/eval_results
```

**生成混淆矩阵图与分数分布图：**

下面这条命令会直接生成 `verification_pairs_report.png`，其中包含：
- 左侧余弦相似度分布图
- 右侧混淆矩阵图

```bash
python scripts/evaluate_model.py --mode pairs \
    --checkpoint checkpoints/classifier_final.pt \
    --config config/config.yaml \
    --pairs_csv data/valid_pairs.csv \
    --output logs/metrics
```

生成后的图片路径为：

```text
logs/metrics/verification_pairs_report.png
```

**验证对 CSV 格式：**
```csv
audio1,audio2,label
speaker_001/audio_01.wav,speaker_001/audio_02.wav,1
speaker_001/audio_01.wav,speaker_002/audio_01.wav,0
```

**输出示例：**
```
验证对评估结果
==================================================
阈值: 0.5000
准确率: 95.23%
精确率: 94.87%
召回率: 95.61%
F1 Score: 95.24%
EER: 4.12% (最佳阈值: 0.5123)

分数统计:
  正样本相似度: 0.7823 ± 0.0891
  负样本相似度: 0.3421 ± 0.1234

混淆矩阵:
  TP=485, TN=478
  FP=22, FN=15
```

### 4.3 分类准确率评估

```bash
# 分类准确率评估
python scripts/evaluate_model.py --mode classification \
    --checkpoint checkpoints/classifier_final.pt

# 指定配置文件
python scripts/evaluate_model.py --mode classification \
    --config config/config.yaml \
    --checkpoint checkpoints/classifier_best.pt
```

**输出示例：**
```
分类评估结果
==================================================
准确率: 92.34%
F1 Score: 91.87%
Precision: 92.12%
Recall: 91.87%
总样本数: 500
正确预测数: 462
```

### 4.4 说话人验证 EER 评估

```bash
# EER 评估（使用采样加速）
python scripts/evaluate_model.py --mode verification \
    --checkpoint checkpoints/classifier_final.pt \
    --sample_ratio 0.1

# 全量评估（较慢）
python scripts/evaluate_model.py --mode verification \
    --sample_ratio 1.0
```

**输出示例：**
```
验证评估结果
==================================================
EER: 0.12%
EER阈值: 0.6234
minDCF: 0.0012
正样本对数: 12500
负样本对数: 12500
```

### 4.5 综合评估

```bash
# 执行所有评估模式
python scripts/evaluate_model.py --mode both \
    --checkpoint checkpoints/classifier_final.pt \
    --pairs_csv data/valid_pairs.csv \
    --output logs/metrics
```

### 4.6 验证参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | checkpoints/classifier_final.pt | 模型检查点路径 |
| `--config` | config/config.yaml | 配置文件路径 |
| `--output` | logs/metrics | 输出目录 |
| `--mode` | pairs | 评估模式 |
| `--pairs_csv` | data/valid_pairs.csv | 验证对 CSV 文件 |
| `--threshold` | 0.5 | 相似度阈值 |
| `--sample_ratio` | 0.1 | 采样比例（加速验证评估） |

### 4.7 验证输出文件

| 文件 | 说明 |
|------|------|
| `verification_results.csv` | 验证对结果详情 |
| `verification_pairs_report.png` | 验证对报告图（分数分布、混淆矩阵） |
| `classification_report.png` | 分类报告图 |
| `model_evaluation_report.png` | 综合评估报告（DET、ROC、PR曲线等） |
| `evaluation_summary.csv` | 评估汇总表 |
| `threshold_analysis.csv` | 阈值分析表 |
| `classification_per_speaker.csv` | 每说话人准确率 |

---

## 4. 声纹注册

### 使用预训练模型

```bash
python scripts/enroll.py --user_id user_001 --audio path/to/voice.wav
```

### 使用训练后的模型

```bash
python scripts/enroll.py --user_id user_001 --audio path/to/voice.wav --checkpoint checkpoints/classifier_final.pt
```

**输出示例：**
```
加载训练模型: checkpoints/classifier_final.pt
模型权重已从 checkpoints/classifier_final.pt 加载
注册成功!
用户ID: user_001
嵌入维度: 192
```

注册后，声纹向量存储在 `voiceprint_db.json`。

---

## 5. 身份认证

### 1:1 认证模式（验证是否是某个用户）

```bash
# 使用预训练模型
python scripts/verify.py --user_id user_001 --audio path/to/test.wav

# 使用训练后的模型
python scripts/verify.py --user_id user_001 --audio path/to/test.wav --checkpoint checkpoints/classifier_final.pt
```

**输出示例：**
```
加载训练模型: checkpoints/classifier_final.pt
模型权重已从 checkpoints/classifier_final.pt 加载
已加载 2 个注册用户
用户ID: user_001
相似度: 0.7714
阈值: 0.5
判决: accept
置信度: high
```

### 1:N 辨认模式（找出是谁）

```bash
# 使用预训练模型
python scripts/verify.py --audio path/to/test.wav --identify

# 使用训练后的模型
python scripts/verify.py --audio path/to/test.wav --identify --checkpoint checkpoints/classifier_final.pt
```

**输出示例：**
```
加载训练模型: checkpoints/classifier_final.pt
模型权重已从 checkpoints/classifier_final.pt 加载
已加载 2 个注册用户
识别结果: user_001
相似度: 0.7881
判决: accept
```

---

## 6. 完整示例流程

```bash
# Step 1: 注册用户 user_001（使用说话人 D11 的音频）
python scripts/enroll.py --user_id user_001 --audio data/valid/D11_750.wav --checkpoint checkpoints/classifier_final.pt

# Step 2: 注册用户 user_002（使用另一个说话人的音频）
python scripts/enroll.py --user_id user_002 --audio data/train/A11_0.wav --checkpoint checkpoints/classifier_final.pt

# Step 3: 身份认证（用同一说话人的另一段音频验证 user_001）
python scripts/verify.py --user_id user_001 --audio data/valid/D11_751.wav --checkpoint checkpoints/classifier_final.pt
# 输出: 相似度 0.77, 判决 accept

# Step 4: 说话人辨认（不指定用户，系统自动识别）
python scripts/verify.py --audio data/valid/D11_752.wav --identify --checkpoint checkpoints/classifier_final.pt
# 输出: 识别结果 user_001, 相似度 0.79

# Step 5: 冒充攻击测试（用 user_002 的音频冒充 user_001）
python scripts/verify.py --user_id user_001 --audio data/train/A11_1.wav --checkpoint checkpoints/classifier_final.pt
# 输出: 相似度 0.54, 判决 accept（边界情况，可调高阈值）
```

---

## 7. 参数说明

### 命令行参数

#### enroll.py

| 参数 | 说明 | 必需 |
|------|------|------|
| `--user_id` | 用户ID | 是 |
| `--audio` | 音频文件路径 | 是 |
| `--checkpoint` | 训练模型路径 | 否 |
| `--config` | 配置文件路径 | 否 |

#### verify.py

| 参数 | 说明 | 必需 |
|------|------|------|
| `--user_id` | 声称的用户ID（认证模式） | 认证模式必需 |
| `--audio` | 待测音频路径 | 是 |
| `--checkpoint` | 训练模型路径 | 否 |
| `--identify` | 使用辨认模式（1:N） | 否 |
| `--config` | 配置文件路径 | 否 |

### config/config.yaml

| 参数 | 说明 | 默认值 |
|------|------|--------|
| audio.sample_rate | 采样率 | 16000 |
| audio.min_duration | 最小时长(秒) | 3.0 |
| model.pretrained_model | 预训练模型路径 | pretrained_models |
| training.batch_size | 批大小 | 8 |
| training.learning_rate | 学习率 | 0.001 |
| training.epochs | 训练轮数 | 30 |
| verification.threshold | 认证阈值 | 0.5 |

---

## 8. 置信度说明

| 置信度 | 相似度范围 | 说明 |
|--------|------------|------|
| very_high | >= 0.8 | 非常确信是同一人 |
| high | 0.6 ~ 0.8 | 较为确信是同一人 |
| medium | 阈值 ~ 0.6 | 刚好通过验证 |
| low | < 阈值 | 未通过验证 |
