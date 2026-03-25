"""
生成说话人验证音频对
正样本：同一说话人的两个音频 (label=1)
负样本：不同说话人的两个音频 (label=0)
"""

import pandas as pd
import random
import os
from collections import defaultdict

def generate_pairs(csv_path, output_path, n_pairs_per_speaker=100, seed=42):
    """
    生成验证音频对
    
    Args:
        csv_path: valid.csv 路径
        output_path: 输出文件路径
        n_pairs_per_speaker: 每个说话人生成的正样本对数量
        seed: 随机种子
    """
    random.seed(seed)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 按说话人分组
    speaker_files = defaultdict(list)
    for _, row in df.iterrows():
        speaker_files[row['speaker_id']].append(row['audio_path'])
    
    speakers = list(speaker_files.keys())
    print(f"说话人数量: {len(speakers)}")
    for spk in speakers:
        print(f"  {spk}: {len(speaker_files[spk])} 个音频")
    
    pairs = []
    
    # 生成正样本对 (同一说话人)
    for spk in speakers:
        files = speaker_files[spk]
        if len(files) < 2:
            continue
        
        # 随机选择 n_pairs_per_speaker 对
        n_generate = min(n_pairs_per_speaker, len(files) * (len(files) - 1) // 2)
        
        generated = set()
        attempts = 0
        while len([p for p in pairs if p[2] == 1 and p[0].startswith(spk)]) < n_generate and attempts < n_generate * 10:
            f1, f2 = random.sample(files, 2)
            key = tuple(sorted([f1, f2]))
            if key not in generated:
                generated.add(key)
                pairs.append((f1, f2, 1))  # label=1 表示同一说话人
            attempts += 1
    
    n_positive = sum(1 for p in pairs if p[2] == 1)
    print(f"\n正样本对数量: {n_positive}")
    
    # 生成负样本对 (不同说话人)，数量与正样本持平
    n_negative_target = n_positive
    generated = set()
    attempts = 0
    
    while len([p for p in pairs if p[2] == 0]) < n_negative_target and attempts < n_negative_target * 20:
        spk1, spk2 = random.sample(speakers, 2)
        f1 = random.choice(speaker_files[spk1])
        f2 = random.choice(speaker_files[spk2])
        key = tuple(sorted([f1, f2]))
        if key not in generated:
            generated.add(key)
            pairs.append((f1, f2, 0))  # label=0 表示不同说话人
        attempts += 1
    
    n_negative = sum(1 for p in pairs if p[2] == 0)
    print(f"负样本对数量: {n_negative}")
    print(f"总样本对数量: {len(pairs)}")
    
    # 打乱顺序
    random.shuffle(pairs)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('audio1,audio2,label\n')
        for audio1, audio2, label in pairs:
            f.write(f'{audio1},{audio2},{label}\n')
    
    print(f"\n验证对文件已保存: {output_path}")
    
    # 统计
    print("\n统计信息:")
    print(f"  正样本 (同一说话人): {n_positive} ({n_positive/len(pairs)*100:.1f}%)")
    print(f"  负样本 (不同说话人): {n_negative} ({n_negative/len(pairs)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='生成说话人验证音频对')
    parser.add_argument('--input', type=str, default='data/valid.csv', help='输入CSV文件')
    parser.add_argument('--output', type=str, default='data/valid.csv', help='输出CSV文件')
    parser.add_argument('--n_pairs', type=int, default=100, help='每个说话人的正样本对数量')
    args = parser.parse_args()
    
    generate_pairs(args.input, args.output, args.n_pairs)
