# 声纹识别系统使用说明

## 1. 环境准备

首先确保安装了正确版本的依赖：

```bash
# 安装 PyTorch（CUDA 13.0版本）
pip install torch==2.10.0+cu130 torchaudio==2.10.0 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# 安装 SpeechBrain
pip install speechbrain==1.0.3

# 安装其他依赖
pip install numpy==2.3.5 tqdm==4.67.3 pyyaml scipy soundfile matplotlib seaborn scikit-learn pandas
```

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
cd D:\github\Speaker
python scripts/train.py
```

训练完成后，模型保存在 `checkpoints/` 目录。

---

## 4. 声纹注册

```bash
# 注册用户
python scripts/enroll.py --user_id user_001 --audio data/train/speaker_001/audio_01.wav

# 注册多个用户
python scripts/enroll.py --user_id user_002 --audio path/to/voice.wav
```

注册后，声纹向量存储在 `voiceprint_db.json`。

---

## 5. 身份认证

**1:1认证模式**（验证是否是某个用户）：
```bash
python scripts/verify.py --user_id user_001 --audio path/to/test_voice.wav
```

**1:N辨认模式**（找出是谁）：
```bash
python scripts/verify.py --audio path/to/test_voice.wav --identify
```

---

## 6. 输出示例

**认证结果：**
```
用户ID: user_001
相似度: 0.7823
阈值: 0.5
判决: accept
置信度: high
```

---

## 快速测试流程

```bash
# 1. 准备一条测试音频到 data/train/ 目录

# 2. 编辑 data/train.csv 添加标签

# 3. 训练（可选，直接用预训练模型也行）
python scripts/train.py

# 4. 注册声纹
python scripts/enroll.py --user_id test_user --audio your_audio.wav

# 5. 验证
python scripts/verify.py --user_id test_user --audio another_audio.wav
```

---

## 参数说明

### config/config.yaml

| 参数 | 说明 | 默认值 |
|------|------|--------|
| audio.sample_rate | 采样率 | 16000 |
| audio.min_duration | 最小时长(秒) | 3.0 |
| audio.n_mels | Mel滤波器数量 | 80 |
| model.pretrained_model | 预训练模型ID | LanceaKing/spkrec-ecapa-cnceleb |
| training.batch_size | 批大小 | 32 |
| training.learning_rate | 学习率 | 0.001 |
| training.epochs | 训练轮数 | 50 |
| verification.threshold | 认证阈值 | 0.5 |
