import os
import sys
import torch
import gc
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

# --- Path Setup (为了能引用 utils) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

# ✅ 引入你写好的本地加载器
from utils.model_loader import load_clip_model

# --- Configuration ---
# 这里对应你 utils/model_loader.py 里的 keys
# B-32 -> L-14 -> H-14
MODELS_TO_TEST = ["base", "large", "huge"] 

DATASET_ROOT = os.path.join(project_root, "data")
FIXED_SHOTS = 16 
FIXED_ALPHA = 2.0 
BETA = 5.5

def build_cache_model(model, dataset, n_shots, device):
    """ (保持不变：这是修复了精度问题的版本) """
    cache_keys = []
    cache_values = []
    classes = dataset.classes
    class_indices = {c: [] for c in classes}
    
    for idx, (_, label_idx) in enumerate(dataset):
        label_name = classes[label_idx]
        if len(class_indices[label_name]) < n_shots:
            class_indices[label_name].append(idx)
            
    with torch.no_grad():
        for i, c in enumerate(classes):
            indices = class_indices[c]
            if len(indices) < n_shots:
                indices = indices * (n_shots // len(indices)) + indices[:n_shots % len(indices)]
            indices = indices[:n_shots]
            
            # Get Tensors
            image_tensors = torch.stack([dataset[idx][0] for idx in indices]).to(device)
            # Extract
            feat = model.get_image_features(pixel_values=image_tensors)
            feat /= feat.norm(dim=-1, keepdim=True)
            cache_keys.append(feat)
            
            # Label & Dtype Fix (关键)
            label_onehot = torch.zeros(n_shots, len(classes)).to(device)
            label_onehot = label_onehot.to(feat.dtype) 
            label_onehot[:, i] = 1
            cache_values.append(label_onehot)

    return torch.cat(cache_keys, dim=0), torch.cat(cache_values, dim=0)

def run_backbone_ablation():
    print("="*60)
    print(f"🚀 Ablation Study: Laion-2B Scaling (Shots={FIXED_SHOTS})")
    print("="*60)

    results = {}

    for model_key in MODELS_TO_TEST:
        print(f"\n>> Testing Backbone: Laion-2B [{model_key.upper()}]")
        
        # 1. 使用你的加载器加载本地模型
        try:
            # use_fp16=True 跟你的默认设置保持一致
            model, processor, device = load_clip_model(model_name=model_key, use_fp16=True)
        except Exception as e:
            print(f"   ❌ Error loading {model_key}: {e}")
            continue

        # 2. Prepare Data (Transform 随模型变化)
        def transform(image):
            # 注意：Laion模型可能有不同的预处理要求，processor 会自动处理
            inputs = processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)

        train_dataset = OxfordIIITPet(root=DATASET_ROOT, split="trainval", target_types="category", transform=transform)
        test_dataset = OxfordIIITPet(root=DATASET_ROOT, split="test", target_types="category", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0) # ViT-H 显存占用大，batch_size 调小点保险
        class_names = [c.replace("_", " ") for c in test_dataset.classes]

        # 3. Oracle Zero-Shot Weights
        tpl = "a photo of a {}, a type of pet."
        texts = [tpl.format(c) for c in class_names]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            w_zero = model.get_text_features(**inputs)
            w_zero /= w_zero.norm(dim=-1, keepdim=True)

        # 4. Build Cache
        print("   Building Cache...")
        cache_keys, cache_values = build_cache_model(model, train_dataset, FIXED_SHOTS, device)

        # 5. Inference
        print("   Running Inference...")
        total, correct_zero, correct_few = 0, 0, 0
        
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                img_feats = model.get_image_features(images)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)
                
                # Zero-Shot
                logits_zero = 100. * img_feats @ w_zero.T
                
                # Few-Shot
                affinity = img_feats @ cache_keys.T
                cache_logits = ((-BETA * (1 - affinity)).exp()) @ cache_values
                logits_final = logits_zero + FIXED_ALPHA * cache_logits
                
                correct_zero += (logits_zero.argmax(1) == labels).sum().item()
                correct_few += (logits_final.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        acc_zero = 100 * correct_zero / total
        acc_few = 100 * correct_few / total
        results[model_key] = (acc_zero, acc_few)
        
        print(f"   [Result] Zero-Shot: {acc_zero:.2f}% | Few-Shot: {acc_few:.2f}%")

        # Cleanup GPU memory (ViT-Huge 很大，必须清理)
        del model, processor, w_zero, cache_keys, cache_values
        torch.cuda.empty_cache()
        gc.collect()

    # --- Final Table ---
    print("\n" + "="*60)
    print(f"SCALING LAW ABLATION (Laion-2B, {FIXED_SHOTS}-Shot)")
    print("="*60)
    print(f"{'Model Size':<12} | {'Zero-Shot':<10} | {'Few-Shot':<10} | {'Gain':<6}")
    print("-" * 60)
    
    # 按 Base -> Large -> Huge 顺序打印
    for key in MODELS_TO_TEST:
        if key in results:
            z, f = results[key]
            # 格式化一下名字
            display_name = f"ViT-{key[0].upper()}" # ViT-B, ViT-L, ViT-H
            print(f"{display_name:<12} | {z:.2f}%     | {f:.2f}%     | +{f-z:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_backbone_ablation()