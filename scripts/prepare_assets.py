import os
import urllib.request

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
assets_dir = os.path.join(project_root, "assets")

os.makedirs(assets_dir, exist_ok=True)

IMAGES_TO_DOWNLOAD = [
    ("cat_dog.jpg", "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"),
    ("cyberpunk.jpg", "https://images.unsplash.com/photo-1601042879364-f3947d3f9c16?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"),
    ("emotion.jpg", "https://images.unsplash.com/photo-1521075486433-bf4052bb37bc?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"),
    ("red_envelop1.jpg", "https://images.unsplash.com/photo-1612201598945-f66a763965bd?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"),
    ("red_envelop2.jpg", "https://images.unsplash.com/photo-1651471948200-a9974e56fd71?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"),
]

def download_images():
    print(f"preparing test image at: {assets_dir} ...")
    
    for filename, url in IMAGES_TO_DOWNLOAD:
        save_path = os.path.join(assets_dir, filename)
        
        if os.path.exists(save_path):
            print(f"already existence: {filename}")
            continue
            
        print(f"downloading: {filename} ...")
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"download fails {filename}: {e}")

    print("\n all ready!")

if __name__ == "__main__":
    download_images()
