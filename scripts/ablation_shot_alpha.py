import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from transformers import CLIPProcessor, CLIPModel

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from utils.model_loader import load_clip_model

# --- Configuration ---
MODEL_NAME = "large"
BATCH_SIZE = 64  # 推理可以大一点
DATASET_ROOT = os.path.join(project_root, "data")

# Ablation Settings
SHOTS_LIST = [1, 2, 4, 8, 16, 32]
ALPHA_LIST = [1.0, 2.0, 3.0, 4.0, 5.0] 
BETA = 5.5  # Tip-Adapter 的锐化参数，通常固定

def log_msg(stage, msg):
    print(f"[{stage}] {msg}")

def build_cache_model(model, processor, dataset, n_shots, device):
    """build Tip-Adapter  Key-Value Cache"""
    cache_keys = []
    cache_values = []
    
    classes = dataset.classes
    class_indices = {c: [] for c in classes}
    
    # 快速建立索引
    for idx, (_, label_idx) in enumerate(dataset):
        label_name = classes[label_idx]
        if len(class_indices[label_name]) < n_shots:
            class_indices[label_name].append(idx)
            
    print(f"Building Cache for {n_shots} shots...") 
    with torch.no_grad():
        for i, c in enumerate(classes):
            indices = class_indices[c]
            if len(indices) < n_shots:
                indices = indices * (n_shots // len(indices)) + indices[:n_shots % len(indices)]
            indices = indices[:n_shots]
            
            # 1. 获取图像 Tensor
            image_tensors = torch.stack([dataset[idx][0] for idx in indices]).to(device)
            
            # 2. 提取特征 (模型输出可能是 FP16)
            image_features = model.get_image_features(pixel_values=image_tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            cache_keys.append(image_features) 
            
            label_onehot = torch.zeros(n_shots, len(classes)).to(device)
            label_onehot = label_onehot.to(image_features.dtype) # <--- 加上这一句！
            
            label_onehot[:, i] = 1
            cache_values.append(label_onehot)

    return torch.cat(cache_keys, dim=0), torch.cat(cache_values, dim=0)

def run_ablation():
    print("\n" + "="*60)
    print(f"🚀 STARTING ABLATION STUDY: Shots vs. Alpha")
    print(f"   Model: {MODEL_NAME} | Dataset: Oxford-IIIT Pet")
    print("="*60 + "\n")

    # --- 1. Load Model ---
    log_msg("Setup", f"Loading CLIP Model ({MODEL_NAME})...")
    model, processor, device = load_clip_model(MODEL_NAME)
    
    # --- 2. Load Data ---
    log_msg("Setup", "Loading Datasets...")
    def transform(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

    # Trainval 用于构建 Cache (Few-Shot Source)
    train_dataset = OxfordIIITPet(root=DATASET_ROOT, split="trainval", target_types="category", download=True, transform=transform)
    # Test 用于评估
    test_dataset = OxfordIIITPet(root=DATASET_ROOT, split="test", target_types="category", download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    class_names = [c.replace("_", " ") for c in test_dataset.classes]
    log_msg("Setup", f"Classes: {len(class_names)} | Test Samples: {len(test_dataset)}")

    # --- 3. Pre-compute Test Features (Optimization) ---
    # 只需要跑一次测试集推理，后面调参全是矩阵运算，极快。
    log_msg("Pre-compute", "Extracting Features for ALL Test Images...")
    all_test_feats = []
    all_test_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            feats = model.get_image_features(images)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_test_feats.append(feats)
            all_test_labels.append(labels.to(device))
            
    test_feats = torch.cat(all_test_feats, dim=0)   # [N_test, dim]
    test_labels = torch.cat(all_test_labels, dim=0) # [N_test]
    log_msg("Pre-compute", "Done. Feature Matrix Ready.")

    # --- 4. Zero-Shot Base Weights ---
    log_msg("TextEncoder", "Computing Oracle Prompt Embeddings...")
    tpl_oracle = "a photo of a {}, a type of pet."
    texts = [tpl_oracle.format(c) for c in class_names]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        w_zero = model.get_text_features(**inputs)
        w_zero /= w_zero.norm(dim=-1, keepdim=True)
        
    # Pre-calculate Zero-Shot Logits [N_test, n_classes]
    logits_zero = 100. * test_feats @ w_zero.T
    acc_zero = (logits_zero.argmax(dim=1) == test_labels).float().mean().item() * 100
    log_msg("Baseline", f"Zero-Shot (Oracle) Accuracy: {acc_zero:.2f}%")

    # --- 5. Main Ablation Loop ---
    print("\n" + "-"*60)
    log_msg("Ablation", "Running Grid Search...")
    print("-"*60)
    
    results_matrix = {} # {shots: {alpha: acc}}

    # 外层循环：Shots
    for shots in SHOTS_LIST:
        print(f"\n>> Configuration: {shots}-Shot Cache")
        
        # 构建 Cache (耗时极短)
        cache_keys, cache_values = build_cache_model(model, processor, train_dataset, shots, device)
        
        # 计算 Cache Logits (Tip-Adapter Formula)
        # Affinity = exp(-beta * (1 - cosine))
        affinity = test_feats @ cache_keys.T
        cache_logits = ((-BETA * (1 - affinity)).exp()) @ cache_values
        
        acc_per_alpha = {}
        
        # 内层循环：Alpha (瞬间完成)
        pbar = tqdm(ALPHA_LIST, desc=f"   Sweeping Alpha ({shots}-shot)", leave=False)
        for alpha in pbar:
            # Tip-Adapter: Logits = Zero + alpha * Cache
            logits_final = logits_zero + alpha * cache_logits
            pred = logits_final.argmax(dim=1)
            acc = (pred == test_labels).float().mean().item() * 100
            acc_per_alpha[alpha] = acc
            
        results_matrix[shots] = acc_per_alpha
        
        # 打印当前 Shot 的最佳结果
        best_alpha = max(acc_per_alpha, key=acc_per_alpha.get)
        print(f"   [Result] Best: {acc_per_alpha[best_alpha]:.2f}% (at alpha={best_alpha})")

    # --- 6. Final Report Table ---
    print("\n\n")
    print("="*65)
    print("FINAL ABLATION RESULTS (Accuracy %)")
    print("="*65)
    
    # Header
    # 动态生成表头
    header = f"{'Shots':<6} |"
    for alpha in ALPHA_LIST:
        header += f" a={alpha:<3} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for shots in SHOTS_LIST:
        row = f" {shots:<2}-Shot |"
        for alpha in ALPHA_LIST:
            acc = results_matrix[shots][alpha]
            # 简单的加粗逻辑（用星号标记最高分）
            is_best = (acc == max(results_matrix[shots].values()))
            mark = "*" if is_best else " "
            row += f" {acc:.2f}{mark} |"
        print(row)
    print("="*65)
    print("(* indicates best alpha for that shot count)")

if __name__ == "__main__":
    run_ablation()