import os
import torch
from transformers import CLIPProcessor, CLIPModel

_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_current_dir)

MODEL_CONFIGS = {
    "base": os.path.join(PROJECT_ROOT, "models", "clip-vit-base-laion2b"),
    "large": os.path.join(PROJECT_ROOT, "models", "clip-vit-large-laion2b"),
    "huge": os.path.join(PROJECT_ROOT, "models", "clip-vit-huge-laion2b"),
}

def load_clip_model(model_name="large", use_fp16=True):
    """
    load CLIP model.
    Args:
        model_name (str): 'base', 'large' or 'huge'
        use_fp16 (bool): True for semi-precision
    Returns:
        model, processor, device
    """
    model_path = MODEL_CONFIGS.get(model_name)
    valid_keys = list(MODEL_CONFIGS.keys())
    if not model_path:
        raise ValueError(f"Unknown model name: {model_name}. Optional: {valid_keys}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"can not find model file: {model_path}"
                                "plz execute scripts/download_models.py firstly.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ModelLoader] loading: {model_name} ({device})")

    try:
        dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

        model = CLIPModel.from_pretrained(model_path, dtype=dtype).to(device)
        processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)

        print(f"loading success! (FP16={use_fp16})")
        return model, processor, device

    except Exception as e:
        print(f"loading fail: {e}")
        raise e
