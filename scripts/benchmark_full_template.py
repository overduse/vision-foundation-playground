import os
import sys
import torch
import torch.nn.functional as F
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
MODEL_NAME = "large"  # 使用 ViT-L/14 测试
BATCH_SIZE = 32
DATASET_ROOT = os.path.join(project_root, "data")

# --- 🏆 The Golden List: OpenAI's 80 Templates ---
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def run_benchmark():
    print(f"Loading Model: {MODEL_NAME}...")
    model, processor, device = load_clip_model(MODEL_NAME)
    
    print("Loading Oxford Pets...")
    # 这里简单起见只用 BenchmarkTransform 逻辑
    def transform(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

    test_dataset = OxfordIIITPet(root=DATASET_ROOT, split="test", target_types="category", download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    class_names = [c.replace("_", " ") for c in test_dataset.classes]
    
    # --- 1. Compute Weights for 3 Methods ---
    print("Computing Text Embeddings...")
    
    # A. Standard Single
    tpl_std = "a photo of a {}."
    texts_std = [tpl_std.format(c) for c in class_names]
    inputs_std = processor(text=texts_std, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        w_std = model.get_text_features(**inputs_std)
        w_std /= w_std.norm(dim=-1, keepdim=True)
        
    # B. Oracle (Best) Single
    tpl_oracle = "a photo of a {}, a type of pet."
    texts_oracle = [tpl_oracle.format(c) for c in class_names]
    inputs_oracle = processor(text=texts_oracle, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        w_oracle = model.get_text_features(**inputs_oracle)
        w_oracle /= w_oracle.norm(dim=-1, keepdim=True)

    # C. Full Ensemble (80 Templates)
    w_ensemble_list = []
    # Batch process templates to avoid OOM or slow tokenization
    print(f"Processing {len(IMAGENET_TEMPLATES)} templates for Ensemble...")
    for tpl in tqdm(IMAGENET_TEMPLATES):
        texts = [tpl.format(c) for c in class_names]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = model.get_text_features(**inputs)
            feat /= feat.norm(dim=-1, keepdim=True)
            w_ensemble_list.append(feat)
    
    # Mean pooling
    w_ensemble = torch.stack(w_ensemble_list).mean(dim=0)
    w_ensemble /= w_ensemble.norm(dim=-1, keepdim=True) # Re-normalize after mean

    # --- 2. Inference ---
    print("Running Inference...")
    correct_std = 0
    correct_oracle = 0
    correct_ens = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            img_feats = model.get_image_features(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            
            # Prediction
            acc_std = (100 * img_feats @ w_std.T).argmax(dim=-1)
            acc_oracle = (100 * img_feats @ w_oracle.T).argmax(dim=-1)
            acc_ens = (100 * img_feats @ w_ensemble.T).argmax(dim=-1)
            
            correct_std += (acc_std == labels).sum().item()
            correct_oracle += (acc_oracle == labels).sum().item()
            correct_ens += (acc_ens == labels).sum().item()
            total += labels.size(0)

    # --- 3. Report ---
    acc_std = 100 * correct_std / total
    acc_oracle = 100 * correct_oracle / total
    acc_ens = 100 * correct_ens / total
    
    print("\n" + "="*50)
    print(f"RESULTS (Model: {MODEL_NAME} | Dataset: Pets)")
    print("="*50)
    print(f"1. Standard Single ('a photo of a {{}}')       : {acc_std:.2f}%")
    print(f"2. Oracle Single   ('... a type of pet')      : {acc_oracle:.2f}%")
    print(f"3. Full Ensemble   (80 Templates)             : {acc_ens:.2f}%")
    print("="*50)
    print(f"Gain (Ensemble vs Standard): {acc_ens - acc_std:+.2f}%")
    print(f"Gain (Ensemble vs Oracle):   {acc_ens - acc_oracle:+.2f}%")

if __name__ == "__main__":
    run_benchmark()