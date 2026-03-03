# 声纹识别系统使用说明

## 1. 环境准备

首先确保安装了正确版本的依赖：

```bash
# 安装 PyTorch（CUDA 13.0版本）
pip install torch==2.10.0+cu130 torchaudio==2.10.0 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# 安装 SpeechBrain
pip install speechbrain==1.0.3

# 安装其他依赖
pip install numpy tqdm pyyaml scipy soundfile matplotlib seaborn scikit-learn pandas
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