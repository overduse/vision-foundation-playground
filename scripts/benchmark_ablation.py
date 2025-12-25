import os
import sys
import torch
import random
import numpy as np
import logging
import datetime
import gc
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, OxfordIIITPet, Food101
from transformers import CLIPProcessor, CLIPModel

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.model_loader import load_clip_model

# --- Global Configuration ---
BATCH_SIZE = 32  # Conservative batch size to accommodate Food-101 + Huge Model
SEED = 42
SHOTS = 16       # For Tip-Adapter
ALPHA = 2.0      # Tip-Adapter Strength
BETA = 5.5       # Tip-Adapter Sharpness

DATASET_ROOT = os.path.join(project_root, "data")
LOG_DIR = os.path.join(project_root, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# --- Templates Library ---
TEMPLATES_LIB = {
    "cifar10": [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a low resolution photo of a {}.",
        "a bad photo of a {}.",
        "a photo of the {}.",
    ],
    "pets": [
        "a photo of a {}, a type of pet.",
        "a close-up photo of a {}.",
        "a photo of a {}, a domesticated animal.",
        "artistic photo of a {}.",
        "a photo of the {}, looking at the camera.",
    ],
    "food101": [
        "a photo of {}, a type of food.",
        "a photo of the delicious {}.",
        "artistic food photography of {}.",
        "a close-up photo of the {}.",
        "a top-down photo of {}.",
    ]
}

# The single "Best" template to use for the Baseline comparison
BEST_TEMPLATE_MAP = {
    "cifar10": "a photo of a {}.",
    "pets": "a photo of a {}, a type of pet.",
    "food101": "a photo of the delicious {}.",
}

# --- Helper Functions ---

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"benchmark_ultimate_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger("UltimateBenchmark")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger, log_path

class BenchmarkTransform:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

# --- Dataset Manager ---

def load_dataset_manager(name, processor):
    transform = BenchmarkTransform(processor)
    
    if name == "cifar10":
        train_set = CIFAR10(root=DATASET_ROOT, train=True, download=True, transform=transform)
        test_set = CIFAR10(root=DATASET_ROOT, train=False, download=True, transform=transform)
        class_names = train_set.classes
        
    elif name == "pets":
        train_set = OxfordIIITPet(root=DATASET_ROOT, split="trainval", target_types="category", download=True, transform=transform)
        test_set = OxfordIIITPet(root=DATASET_ROOT, split="test", target_types="category", download=True, transform=transform)
        class_names = [c.replace("_", " ") for c in train_set.classes]
        
    elif name == "food101":
        train_set = Food101(root=DATASET_ROOT, split="train", download=True, transform=transform)
        test_set = Food101(root=DATASET_ROOT, split="test", download=True, transform=transform)
        class_names = [c.replace("_", " ") for c in train_set.classes]
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    return train_set, test_set, class_names

# --- Method Implementations ---

def get_few_shot_loader(dataset, shots, num_classes, seed):
    """Samples N-shots for Tip-Adapter cache."""
    # Handle label access for different datasets
    if hasattr(dataset, 'targets'): labels = dataset.targets # CIFAR
    elif hasattr(dataset, '_labels'): labels = dataset._labels # Pets/Food
    else: labels = [y for _, y in dataset] # Fallback
    
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    selected_indices = []
    rng = random.Random(seed)
    
    for label_idx in range(num_classes):
        indices = class_indices[label_idx]
        if len(indices) >= shots:
            selected = rng.sample(indices, shots)
        else:
            selected = indices
        selected_indices.extend(selected)
    
    subset = Subset(dataset, selected_indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def build_tip_adapter_cache(model, loader, device, num_classes):
    """Builds cache with FP16 fix."""
    cache_keys = []
    cache_values = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="   [Cache] Building"):
            images = images.to(device)
            labels = labels.to(device)
            
            features = model.get_image_features(images)
            features /= features.norm(dim=-1, keepdim=True)
            cache_keys.append(features)
            
            # Fix: Ensure one_hot matches features dtype (FP16/FP32)
            one_hot = F.one_hot(labels, num_classes=num_classes).to(features.dtype)
            cache_values.append(one_hot)
            
    return torch.cat(cache_keys, dim=0), torch.cat(cache_values, dim=0)

def compute_text_embeddings(model, processor, class_names, templates, device):
    """Computes text embeddings. Returns Single Best and Ensemble Mean."""
    # 1. Single Best
    single_template = templates[0] # Assume first one is best for baseline
    texts_single = [single_template.format(c) for c in class_names]
    inputs_s = processor(text=texts_single, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        w_single = model.get_text_features(**inputs_s)
        w_single /= w_single.norm(dim=-1, keepdim=True)
    
    # 2. Ensemble
    all_features = []
    for tpl in templates:
        texts = [tpl.format(c) for c in class_names]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = model.get_text_features(**inputs)
            feat /= feat.norm(dim=-1, keepdim=True)
            all_features.append(feat)
    
    w_ensemble = torch.stack(all_features).mean(dim=0)
    w_ensemble /= w_ensemble.norm(dim=-1, keepdim=True)
    
    return w_single.T, w_ensemble.T

# --- Core Logic ---

def process_dataset(logger, model, processor, device, dataset_name):
    logger.info(f"   [Data] Loading {dataset_name}...")
    
    # 1. Load Data
    try:
        train_set, test_set, class_names = load_dataset_manager(dataset_name, processor)
    except Exception as e:
        logger.error(f"   [Error] Failed to load {dataset_name}: {e}")
        return None

    num_classes = len(class_names)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Prepare Text Classifiers (Weights)
    # Map dataset to specific templates
    templates_str = TEMPLATES_LIB[dataset_name] 
    # Ensure the "Best" one is used for baseline (override first element logic if needed)
    best_tpl_str = BEST_TEMPLATE_MAP[dataset_name]
    # Construct list where first item is always the Best/Baseline template
    final_templates = [best_tpl_str] + [t for t in templates_str if t != best_tpl_str]
    
    logger.info("   [Text] Computing Zero-Shot & Ensemble weights...")
    w_single, w_ensemble = compute_text_embeddings(model, processor, class_names, final_templates, device)
    
    # 3. Build Tip-Adapter Cache
    logger.info(f"   [Tip] Sampling {SHOTS}-shot cache...")
    cache_loader = get_few_shot_loader(train_set, SHOTS, num_classes, SEED)
    cache_keys, cache_values = build_tip_adapter_cache(model, cache_loader, device, num_classes)
    
    # 4. Inference Loop (One-Pass)
    logger.info("   [Eval] Running Inference (Baseline vs Ensemble vs Adapter)...")
    
    correct_baseline = 0
    correct_ensemble = 0
    correct_adapter = 0 # Adapter uses Baseline weights + Cache
    total = 0
    
    model.eval()
    for images, labels in tqdm(test_loader, desc=f"   Eval {dataset_name}"):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            # Image Features
            img_feats = model.get_image_features(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            
            # --- Method 1: Baseline Zero-Shot ---
            logits_base = 100.0 * img_feats @ w_single
            
            # --- Method 2: Prompt Ensemble ---
            logits_ens = 100.0 * img_feats @ w_ensemble
            
            # --- Method 3: Tip-Adapter (Base + Cache) ---
            # Using w_single (Base) as the zero-shot foundation for Tip-Adapter
            affinity = img_feats @ cache_keys.T
            cache_logits = ((-BETA * (1 - affinity)).exp()) @ cache_values
            logits_adapter = logits_base + (ALPHA * cache_logits)
            
            # Counting
            correct_baseline += (logits_base.argmax(dim=-1) == labels).sum().item()
            correct_ensemble += (logits_ens.argmax(dim=-1) == labels).sum().item()
            correct_adapter += (logits_adapter.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            
    # Calculate Metrics
    results = {
        "Baseline": 100 * correct_baseline / total,
        "Ensemble": 100 * correct_ensemble / total,
        "Adapter": 100 * correct_adapter / total
    }
    
    logger.info(f"   [Result] {dataset_name} | Base: {results['Baseline']:.2f}% | Ens: {results['Ensemble']:.2f}% | Tip: {results['Adapter']:.2f}%")
    
    # Cleanup dataset specific tensors
    del cache_keys, cache_values, w_single, w_ensemble, train_set, test_set, cache_loader
    torch.cuda.empty_cache()
    
    return results

# --- Main Entry Point ---

if __name__ == "__main__":
    setup_seed(SEED)
    logger, log_path = setup_logger()
    
    MODELS = ["base", "large", "huge"]
    DATASETS = ["cifar10", "pets", "food101"]
    
    logger.info("="*60)
    logger.info(f"ULTIMATE BENCHMARK STARTING")
    logger.info(f"Models: {MODELS}")
    logger.info(f"Datasets: {DATASETS}")
    logger.info(f"Config: Batch={BATCH_SIZE}, Shots={SHOTS}")
    logger.info("="*60)
    
    # Store all results: final_report[model][dataset] = {scores}
    final_report = {} 
    
    for model_name in MODELS:
        logger.info(f"\n>>> PROCESSING MODEL: {model_name.upper()} <<<")
        
        try:
            model, processor, device = load_clip_model(model_name)
        except Exception as e:
            logger.error(f"Critical error loading {model_name}: {e}")
            continue
            
        final_report[model_name] = {}
        
        for dataset_name in DATASETS:
            logger.info(f"\n--- Dataset: {dataset_name} ---")
            scores = process_dataset(logger, model, processor, device, dataset_name)
            
            if scores:
                final_report[model_name][dataset_name] = scores
            else:
                final_report[model_name][dataset_name] = {"Baseline": 0, "Ensemble": 0, "Adapter": 0}
                
        # Cleanup Model
        del model, processor
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- Generate Summary Table ---
    print("\n\n")
    logger.info("="*80)
    logger.info("GRAND SUMMARY REPORT")
    logger.info("="*80)
    
    # Header
    # Model | Dataset | Baseline | Ensemble | Adapter | Best Method
    header = f"{'Model':<8} | {'Dataset':<10} | {'Baseline':<8} | {'Ensemble':<8} | {'Tip-Adapt':<9} | {'Best':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    
    for model_name in MODELS:
        if model_name not in final_report: continue
        
        for dataset_name in DATASETS:
            res = final_report[model_name].get(dataset_name, {})
            base = res.get("Baseline", 0)
            ens = res.get("Ensemble", 0)
            tip = res.get("Adapter", 0)
            
            # Determine best
            best_val = max(base, ens, tip)
            if best_val == tip: best_str = "Tip-Adapt"
            elif best_val == ens: best_str = "Ensemble"
            else: best_str = "Baseline"
            
            row = f"{model_name:<8} | {dataset_name:<10} | {base:<8.2f} | {ens:<8.2f} | {tip:<9.2f} | {best_str:<10}"
            logger.info(row)
        
        logger.info("-" * len(header)) # Separator between models

    logger.info(f"[DONE] Full logs saved to: {log_path}")