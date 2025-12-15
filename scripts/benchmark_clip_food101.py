import os
import sys
import torch
import random
import numpy as np
import logging
import datetime
import gc

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from transformers import CLIPProcessor, CLIPModel

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.model_loader import load_clip_model

# --- Global Configuration ---
# WARNING: Food101 is a large dataset (~5GB download, ~10GB uncompressed).
# Ensure you have enough disk space in your 'data' folder.
# Images are high resolution. If OOM occurs, reduce BATCH_SIZE.
BATCH_SIZE = 64
SEED = 42
DATASET_ROOT = os.path.join(project_root, "data")
LOG_DIR = os.path.join(project_root, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

# --- Prompt Templates (Tailored for Food) ---
# Food images rely heavily on texture and lighting.
PROMPT_TEMPLATES = {
    "Raw Label": lambda c: f"{c}",
    "Standard": lambda c: f"a photo of {c}",
    "Delicious": lambda c: f"a photo of the delicious {c}",
    "Food Photography": lambda c: f"artistic food photography of {c}",
}

# --- Helper Functions ---

def setup_seed(seed):
    """
    Sets the global random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[INFO] Random seed set to: {seed}")

def setup_logger(model_name):
    """
    Configures a logger to capture execution details for a specific model.
    Saves logs to both file and console with a clean format.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"run_food101_{model_name}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger(f"Benchmark-Food101-{model_name}")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    logger.info(f"[INFO] Logging initialized. Log file: {log_path}")
    return logger

class BenchmarkTransform:
    """
    A wrapper class to apply CLIPProcessor to images within a PyTorch DataLoader.
    This avoids pickle errors on Windows during multiprocessing.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

def precompute_text_features(model, processor, classes, template_func, device):
    """
    Precomputes text embeddings for all class labels using a specific template.
    """
    texts = [template_func(c) for c in classes]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features, texts

# --- Core Benchmark Logic ---

def run_single_model_benchmark(model_name):
    """
    Runs the benchmark for a single CLIP model on the Food-101 dataset.
    """
    logger = setup_logger(model_name)
    logger.info("-" * 40)
    logger.info(f"[START] BENCHMARK FOR MODEL: {model_name}")
    logger.info(f"[CONFIG] Batch={BATCH_SIZE}, Seed={SEED}")
    logger.info("-" * 40)

    try:
        model, processor, device = load_clip_model(model_name)
        logger.info(f"[INFO] Model loaded on {device}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model {model_name}: {e}")
        return None

    # Prepare Dataset
    logger.info("[INFO] Preparing Food-101 dataset. This may take a while if downloading...")
    
    transform_pipeline = BenchmarkTransform(processor)
    
    try:
        # Food101 split is 'test' or 'train', different from CIFAR's train=False
        test_dataset = Food101(
            root=DATASET_ROOT, 
            split="test", 
            download=True, 
            transform=transform_pipeline
        )
    except Exception as e:
        logger.error(f"[ERROR] Failed to load dataset: {e}")
        logger.error("[HINT] Ensure you have ~10GB free space for Food-101.")
        return None

    # Clean class names (replace underscores with spaces)
    # Example: 'baby_back_ribs' -> 'baby back ribs'
    class_names = [c.replace("_", " ") for c in test_dataset.classes]
    
    logger.info(f"[DATA] Dataset size: {len(test_dataset)} images")
    logger.info(f"[DATA] Number of classes: {len(class_names)}")

    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    results = {}
    
    for template_name, template_func in PROMPT_TEMPLATES.items():
        logger.info(f"\n[TEST] Template: [{template_name}]")
        
        text_features, sample_texts = precompute_text_features(model, processor, class_names, template_func, device)
        logger.info(f"       Sample prompt: '{sample_texts[0]}'")

        correct_count = 0
        total_count = 0

        model.eval()
        for images, labels in tqdm(dataloader, desc=f"Eval {template_name}"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                image_features = model.get_image_features(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T)
                probs = similarity.softmax(dim=-1)
                pred_indices = probs.argmax(dim=-1)
                
                correct_count += (pred_indices == labels).sum().item()
                total_count += labels.size(0)

        accuracy = 100 * correct_count / total_count
        results[template_name] = accuracy
        logger.info(f"[RESULT] Accuracy ({template_name}): {accuracy:.2f}%")

    # Cleanup Memory
    del model
    del processor
    del text_features
    del image_features
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"[CLEANUP] Memory cleared for {model_name}")

    return results

# --- Main Entry Point ---

if __name__ == "__main__":
    setup_seed(SEED)

    # List of models to evaluate
    MODELS_TO_TEST = ["base", "large", "huge"] 
    
    print(f"\n[GLOBAL] STARTING FOOD-101 BENCHMARK ON: {MODELS_TO_TEST}\n")

    final_report = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n{'#'*60}")
        print(f"Processing: {model_name} ...")
        print(f"{'#'*60}\n")
        
        scores = run_single_model_benchmark(model_name)
        if scores:
            final_report[model_name] = scores

    # Save Summary Report
    session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_filename = f"summary_food101_{session_timestamp}.txt"
    summary_path = os.path.join(LOG_DIR, summary_filename)

    def log_both(f_obj, text):
        print(text)
        f_obj.write(text + "\n")

    print("\n\n")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        log_both(f, "="*80)
        log_both(f, f"GRAND SUMMARY REPORT - FOOD-101 - {session_timestamp}")
        log_both(f, "="*80)
        
        headers = list(PROMPT_TEMPLATES.keys())
        header_row = f"{'Model':<10} | " + " | ".join([f"{h:<15}" for h in headers])
        
        log_both(f, header_row)
        log_both(f, "-" * len(header_row))

        for model_name, scores in final_report.items():
            row = f"{model_name:<10} | "
            for h in headers:
                acc = scores.get(h, 0.0)
                row += f"{acc:<15.2f} | "
            log_both(f, row)
        
        log_both(f, "="*80)
        log_both(f, f"[DONE] Report saved to: {summary_path}")
