import os
from transformers import CLIPProcessor, CLIPModel

# for model files download.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_and_save_model():
    # config path
    model_name = "laion/CLIP-VIT-H-14-laion2B-s32B-b79K"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    save_path = os.path.join(project_root, "models", "clip-vit-huge-laion2b")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"ready for downloading model: {model_name}")
    print(f"target path: {save_path}")

    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)

        print("\n download success.")
        print(f"model file saved at: {save_path}")
    except Exception as e:
        print(f"\n download fails: {e}")

if __name__ == "__main__":
    download_and_save_model()