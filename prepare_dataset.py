import os, json, urllib.request, zipfile, shutil, random
import ssl
from pathlib import Path

# Bypass macOS Python SSL Certificate verification errors
ssl._create_default_https_context = ssl._create_unverified_context

def report_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    if count == 0 or percent % 10 == 0:
        print(f"\rDownloading... {percent}%", end="", flush=True)

def download_and_prepare():
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    coco_bg_url = "http://images.cocodataset.org/zips/val2017.zip"
    
    zip_path = "kvasir-seg.zip"
    coco_zip_path = "coco_bg.zip"
    dataset_dir = Path("dataset").absolute()
    
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")

    # 1. Download Kvasir-SEG
    if not os.path.exists(zip_path):
        print("Downloading Kvasir-SEG dataset (~462MB)... This may take a few minutes.")
        urllib.request.urlretrieve(url, zip_path, reporthook=report_hook)
        print("\nKvasir Download complete.")
    else:
        print("Found existing Kvasir zip file. Skipping download.")
        
    # 2. Download COCO Background Images (to prevent OOD hallucinations)
    if not os.path.exists(coco_zip_path):
        print("\nDownloading COCO Background Images (~778MB) to teach the AI what NOT to detect...")
        urllib.request.urlretrieve(coco_bg_url, coco_zip_path, reporthook=report_hook)
        print("\nCOCO Download complete.")
    else:
        print("Found existing COCO zip file. Skipping download.")
        
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    with zipfile.ZipFile(coco_zip_path, 'r') as zip_ref:
        zip_ref.extractall("coco_raw")
        
    print("Formatting bounding boxes for YOLOv8...")
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    # Read JSON (note the typo 'kavsir' in the official dataset zip)
    with open("Kvasir-SEG/kavsir_bboxes.json") as f:
        bboxes = json.load(f)
        
    images = list(bboxes.keys())
    random.seed(42)
    random.shuffle(images)
    split_idx = int(0.8 * len(images))
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]
    
    def process_split(split_imgs, split_name):
        for img_id in split_imgs:
            img_data = bboxes[img_id]
            img_path = Path("Kvasir-SEG/images") / f"{img_id}.jpg"
            if not img_path.exists():
                continue
            
            shutil.copy(img_path, dataset_dir / 'images' / split_name / f"{img_id}.jpg")
            
            w, h = img_data['width'], img_data['height']
            label_path = dataset_dir / 'labels' / split_name / f"{img_id}.txt"
            with open(label_path, "w") as lf:
                for bbox in img_data['bbox']:
                    xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
                    xc = (xmin + xmax) / 2.0 / w
                    yc = (ymin + ymax) / 2.0 / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    lf.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    
    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')
    
    print("Adding 150 Background (Negative) Images to prevent hallucinations...")
    coco_images = list(Path("coco_raw/val2017").glob("*.jpg"))
    random.shuffle(coco_images)
    bg_images = coco_images[:150]
    
    for i, bg_path in enumerate(bg_images):
        split_name = 'train' if i < 120 else 'val'
        bg_id = f"bg_{i}"
        
        # Copy image
        shutil.copy(bg_path, dataset_dir / 'images' / split_name / f"{bg_id}.jpg")
        
        # Write completely empty text file (YOLO standard for negative/background image)
        label_path = dataset_dir / 'labels' / split_name / f"{bg_id}.txt"
        with open(label_path, "w") as lf:
            pass
    
    yaml_content = f"""path: {dataset_dir}
train: images/train
val: images/val
names:
  0: polyp
"""
    with open("data.yaml", "w") as f:
        f.write(yaml_content)
        
    print("Cleaning up raw files...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if os.path.exists("Kvasir-SEG"):
        shutil.rmtree("Kvasir-SEG")
    if os.path.exists(coco_zip_path):
        os.remove(coco_zip_path)
    if os.path.exists("coco_raw"):
        shutil.rmtree("coco_raw")
    print("Dataset prepared successfully in 'dataset/' directory!")

if __name__ == "__main__":
    download_and_prepare()
