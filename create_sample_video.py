import os
import cv2
import random
import glob

def create_sample():
    os.makedirs("data", exist_ok=True)
    images = glob.glob("dataset/images/val/*.jpg")
    
    if not images:
        print("No validation images found!")
        return

    random.seed(42)
    random.shuffle(images)
    selected_images = images[:60] # 60 images

    # Create video writer
    # MP4 writer (often h264 or mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/sample_colonoscopy.mp4', fourcc, 30.0, (640, 480))

    print(f"Creating sample video from {len(selected_images)} images...")
    
    for img_path in selected_images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        # Resize to 640x480
        frame = cv2.resize(frame, (640, 480))
        
        # Write each image for 15 frames (0.5 seconds)
        for _ in range(15):
            out.write(frame)

    out.release()
    print("✅ Created data/sample_colonoscopy.mp4 successfully!")

if __name__ == "__main__":
    create_sample()
