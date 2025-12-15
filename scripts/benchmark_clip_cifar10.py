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
from torchvision.datasets import CIFAR10
from transformers import CLIPProcessor, CLIPModel

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.model_loader import load_clip_model

BATCH_SIZE = 128
SEED = 42
DATASET_ROOT = os.path.join(project_root, "data")
LOG_DIR = os.path.join(project_root, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

PROMPT_TEMPLATES = {
    "Raw Label": lambda c: f"{c}",
    "Standard": lambda c: f"a photo of a {c}",
    "Descriptive": lambda c: f"a photo of a {c}, a type of object",
    "Bad Prompt": lambda c: f"a microscopic image of a {c}",
}

def setup_seed(seed):
    """
    sets the random seed to ensure experiment reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set to: {seed}")

def setup_logger(model_name):
    """
    sets up a logger to track the detailed runtime progress of a single model.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"run_cifar10_{model_name}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger(f"Benchmark-{model_name}-cifar10")
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

    logger.info(f"Logging initialized. Saving detailed run logs to: {log_path}")
    return logger

def get_cifar10_classes():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

class CifarTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

def precompute_text_features(model, processor, classes, template_func, device):
    texts = [template_func(c) for c in classes]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features, texts


def run_single_model_benchmark(model_name):
    logger = setup_logger(model_name)
    logger.info("="*40)
    logger.info(f"STARTING BENCHMARK FOR MODEL: {model_name}")
    logger.info("="*40)

    try:
        model, processor, device = load_clip_model(model_name)
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None

    transform_pipeline = CifarTransform(processor)
    test_dataset = CIFAR10(root=DATASET_ROOT, train=False, download=True, transform=transform_pipeline)
    
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    class_names = get_cifar10_classes()

    results = {}
    
    for template_name, template_func in PROMPT_TEMPLATES.items():
        logger.info(f"\nTesting Template: [{template_name}]")
        
        text_features, sample_texts = precompute_text_features(model, processor, class_names, template_func, device)
        logger.info(f"Sample prompt: '{sample_texts[0]}'")

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
        logger.info(f"Accuracy ({template_name}): {accuracy:.2f}%")

    del model
    del processor
    del text_features
    del image_features
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"Memory cleared for {model_name}")

    return results

if __name__ == "__main__":
    setup_seed(SEED)

    MODELS_TO_TEST = ["base", "large", "huge"] 
    
    print(f"\nGLOBAL BENCHMARK STARTING: {MODELS_TO_TEST} - cifar10\n")

    final_report = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n{'#'*60}")
        print(f"Processing: {model_name} ...")
        print(f"{'#'*60}\n")
        
        scores = run_single_model_benchmark(model_name)
        if scores:
            final_report[model_name] = scores

    session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_filename = f"summary_cifar10_{session_timestamp}.txt"
    summary_path = os.path.join(LOG_DIR, summary_filename)

    def log_both(f_obj, text):
        print(text)
        f_obj.write(text + "\n")

    print("\n\n")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        log_both(f, "="*80)
        log_both(f, f"GRAND SUMMARY REPORT - CIFAR10 - {session_timestamp}")
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
        log_both(f, f"Report saved to: {summary_path}")
