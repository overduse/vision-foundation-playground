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
from torchvision.datasets import OxfordIIITPet
from transformers import CLIPProcessor, CLIPModel

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.model_loader import load_clip_model

# --- Global Configuration ---
BATCH_SIZE = 64
SEED = 42
SHOTS = 32       # 16-shot learning (standard setting)
ALPHA = 0.5      # Residual weight (Strength of the Adapter)
BETA = 5.5       # Sharpness (Focus on most similar examples)

DATASET_ROOT = os.path.join(project_root, "data")
LOG_DIR = os.path.join(project_root, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# We use the best performing prompt from previous benchmarks
BEST_TEMPLATE = lambda c: f"a photo of a {c}, a type of pet"

# --- Helper Functions ---

def setup_seed(seed):
    """
    Sets the global random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[INFO] Random seed set to: {seed}")

def setup_logger(model_name):
    """
    Configures logging to capture execution details.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"tip_adapter_pets_{model_name}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger(f"TipAdapter-{model_name}")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"[INFO] Logging initialized. Log file: {log_path}")
    return logger

class BenchmarkTransform:
    """
    Wrapper to handle image processing within DataLoader.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

# --- Tip-Adapter Components ---

def get_few_shot_subset(dataset, shots, num_classes, seed):
    """
    Randomly samples 'shots' images per class for the cache model.
    """
    class_indices = [[] for _ in range(num_classes)]
    # OxfordPets labels are integers 0-36
    for idx, label in enumerate(dataset._labels):
        class_indices[label].append(idx)
    
    selected_indices = []
    rng = random.Random(seed)
    
    for label_idx in range(num_classes):
        indices = class_indices[label_idx]
        if len(indices) >= shots:
            selected = rng.sample(indices, shots)
        else:
            selected = indices # Fallback if not enough images
        selected_indices.extend(selected)
    
    return Subset(dataset, selected_indices)

def build_cache_model(model, train_loader, device, num_classes):
    """
    Extracts features from the few-shot set to build the Key-Value cache.
    """
    cache_keys = []
    cache_values = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Building Cache"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Keys: Image Features
            features = model.get_image_features(images)
            features /= features.norm(dim=-1, keepdim=True)
            cache_keys.append(features)
            
            # Values: One-hot Labels
            # one_hot = F.one_hot(labels, num_classes=num_classes).float()
            one_hot = F.one_hot(labels, num_classes=num_classes).to(features.dtype)
            cache_values.append(one_hot)
            
    cache_keys = torch.cat(cache_keys, dim=0)
    cache_values = torch.cat(cache_values, dim=0)
    
    return cache_keys, cache_values

def run_tip_adapter_benchmark(model_name):
    """
    Runs the full Tip-Adapter pipeline and returns both Zero-Shot and Adapter accuracy.
    """
    logger = setup_logger(model_name)
    logger.info("-" * 40)
    logger.info(f"[START] TIP-ADAPTER BENCHMARK: {model_name}")
    logger.info(f"[CONFIG] Shots={SHOTS}, Alpha={ALPHA}, Beta={BETA}")
    logger.info("-" * 40)

    try:
        model, processor, device = load_clip_model(model_name)
        logger.info(f"[INFO] Model loaded on {device}")
    except Exception as e:
        logger.error(f"[ERROR] Model load failed: {e}")
        return None

    transform = BenchmarkTransform(processor)

    # 1. Load Datasets
    logger.info("[DATA] Loading Oxford-IIIT Pet dataset...")
    # 'trainval' for building the cache (Knowledge Base)
    train_dataset = OxfordIIITPet(root=DATASET_ROOT, split="trainval", target_types="category", download=True, transform=transform)
    # 'test' for evaluation
    test_dataset = OxfordIIITPet(root=DATASET_ROOT, split="test", target_types="category", download=True, transform=transform)
    
    # Clean class names
    class_names = [c.replace("_", " ") for c in train_dataset.classes]
    num_classes = len(class_names)
    
    # 2. Sample Few-Shot Data
    logger.info(f"[DATA] Sampling {SHOTS}-shot subset from training data...")
    few_shot_dataset = get_few_shot_subset(train_dataset, SHOTS, num_classes, SEED)
    
    train_loader = DataLoader(few_shot_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Build Cache
    logger.info(f"[CACHE] Constructing adapter cache ({SHOTS * num_classes} samples)...")
    cache_keys, cache_values = build_cache_model(model, train_loader, device, num_classes)
    logger.info(f"[CACHE] Cache built. Keys Shape: {list(cache_keys.shape)}")

    # 4. Precompute Text Classifier (Zero-Shot Weights)
    logger.info(f"[TEXT] Encoding text prompts using template: '{BEST_TEMPLATE('{}')}'")
    texts = [BEST_TEMPLATE(c) for c in class_names]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        clip_weights = model.get_text_features(**text_inputs)
        clip_weights /= clip_weights.norm(dim=-1, keepdim=True)
        clip_weights = clip_weights.T 

    # 5. Inference
    logger.info("[EVAL] Starting Inference Loop...")
    
    correct_zs = 0
    correct_adapter = 0
    total = 0
    
    model.eval()
    for images, labels in tqdm(test_loader, desc="Eval"):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            img_feats = model.get_image_features(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            
            # --- Path 1: Zero-Shot ---
            logits_zs = 100.0 * img_feats @ clip_weights
            
            # --- Path 2: Adapter ---
            affinity = img_feats @ cache_keys.T
            cache_logits = ((-BETA * (1 - affinity)).exp()) @ cache_values
            
            # --- Fusion ---
            logits_adapter = logits_zs + (ALPHA * cache_logits)
            
            # Metrics
            pred_zs = logits_zs.argmax(dim=-1)
            pred_adapter = logits_adapter.argmax(dim=-1)
            
            correct_zs += (pred_zs == labels).sum().item()
            correct_adapter += (pred_adapter == labels).sum().item()
            total += labels.size(0)

    acc_zs = 100 * correct_zs / total
    acc_adapter = 100 * correct_adapter / total
    
    logger.info(f"[RESULT] Zero-Shot Accuracy: {acc_zs:.2f}%")
    logger.info(f"[RESULT] Tip-Adapter Accuracy: {acc_adapter:.2f}%")
    logger.info(f"[RESULT] Improvement: +{acc_adapter - acc_zs:.2f}%")
    
    # Cleanup
    del model, cache_keys, cache_values, clip_weights, img_feats
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"[CLEANUP] Memory cleared for {model_name}")

    return {"Zero-Shot": acc_zs, "Adapter": acc_adapter}

# --- Main Entry Point ---

if __name__ == "__main__":
    setup_seed(SEED)

    # Define models to run
    MODELS_TO_TEST = ["base", "large", "huge"]
    
    print(f"\n[GLOBAL] STARTING TIP-ADAPTER BENCHMARK ON: {MODELS_TO_TEST}\n")

    final_report = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n{'#'*60}")
        print(f"Processing: {model_name} ...")
        print(f"{'#'*60}\n")
        
        res = run_tip_adapter_benchmark(model_name)
        if res:
            final_report[model_name] = res

    # Save Summary Report (Matches previous script style)
    session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_filename = f"summary_tip_adapter_{session_timestamp}.txt"
    summary_path = os.path.join(LOG_DIR, summary_filename)

    def log_both(f_obj, text):
        print(text)
        f_obj.write(text + "\n")

    print("\n\n")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        log_both(f, "="*80)
        log_both(f, f"GRAND SUMMARY REPORT - TIP-ADAPTER (16-shot) - {session_timestamp}")
        log_both(f, "="*80)
        
        # Header: Model | Zero-Shot | Adapter | Gain
        header_row = f"{'Model':<10} | {'Zero-Shot':<15} | {'Tip-Adapter':<15} | {'Gain':<10}"
        
        log_both(f, header_row)
        log_both(f, "-" * len(header_row))

        for model_name, res in final_report.items():
            zs = res['Zero-Shot']
            ad = res['Adapter']
            gain = ad - zs
            
            row = f"{model_name:<10} | {zs:<15.2f} | {ad:<15.2f} | +{gain:<9.2f}"
            log_both(f, row)
        
        log_both(f, "="*80)
        log_both(f, f"[DONE] Report saved to: {summary_path}")
