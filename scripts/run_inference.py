import os
import sys
import torch
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.model_loader import load_clip_model

SELECTED_MODEL = "huge"  # option: "base", "large", "huge"
TEST_IMAGE_PATH = os.path.join(project_root, "assets","test.jpg")

TEXT_LABELS = [
    "a photo of a cat",
    "a photo of a dog",
    "scenery with moutains",
    "cyberpunk city",
    "delicious food",
    "anime style art",
]

def run_inference():
    print(f"initialization procession (Model: {SELECTED_MODEL})...")

    try:
        model, processor, device = load_clip_model(SELECTED_MODEL, use_fp16=True)
    except Exception as e:
        print(f"model load fails: {e}")
        return
    
    if os.path.exists(TEST_IMAGE_PATH):
        print(f"load local image: {TEST_IMAGE_PATH}")
        image = Image.open(TEST_IMAGE_PATH)
    else:
        print(f"can not find image: {TEST_IMAGE_PATH}")
        print("auto generate a pure red test image.")
        image = Image.new('RGB', (500, 500), color='red')
        TEXT_LABELS.append("somthing red")
    
    # preprocess
    inputs = processor(
        text=TEXT_LABELS,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    # result, transfer tensor to numpy
    scores = probs.cpu().numpy()[0]

    results = list(zip(TEXT_LABELS, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 40)
    print(f"{'label': <25} | {'Conf'}")
    print("-" * 40)

    for label, score in results:
        marker = ""
        if score == max(scores):
            marker = "Winner"
        elif score > 0.5:
            marker = "True"

        print(f"{label:<25} | {score:>7.2%} {marker}")

    print("=" * 40 + "\n")

if __name__ == "__main__":
    run_inference()