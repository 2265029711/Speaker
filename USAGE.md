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

```bash
python scripts/train.py
```

训练完成后，模型保存在 `checkpoints/` 目录：
- `classifier_best.pt` - 最佳模型（验证集 EER 最低）
- `classifier_final.pt` - 最终模型
- `classifier_epoch_N.pt` - 每 5 轮保存的检查点

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