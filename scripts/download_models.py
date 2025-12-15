import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: can not find 'huggingface_hub' module.")
    print("plz exe: pip install huggingface-hub")
    sys.exit(1)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

MODELS_TO_DOWNLOAD = {
    "clip-vit-base-laion2b": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "clip-vit-large-laion2b": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "clip-vit-huge-laion2b": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}

def main():
    print(f"project root directory: {PROJECT_ROOT}")
    print(f"model save at: {MODELS_ROOT}")
    print("-" * 60)

    for folder_name, repo_id in MODELS_TO_DOWNLOAD.items():
        save_path = os.path.join(MODELS_ROOT, folder_name)

        print(f"\n checking/downloading: {folder_name}")
        print(f"remote repo: {repo_id}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=save_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=8,

                ignore_patterns=["*.h5", "*.ot", "*.msgpack"]
            )
            print(f"{folder_name} ready" )
        
        except Exception as e:
            print(f"[{folder_name}] download fails: {e}")

    print("\n" + "=" * 60)
    print("all models ready")

if __name__ == "__main__":
    main()