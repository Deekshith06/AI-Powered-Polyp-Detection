from ultralytics import YOLO
import os, sys, shutil

if not os.path.exists("data.yaml"):
    print("Error: data.yaml not found. Please run prepare_dataset.py first.")
    sys.exit(1)

print("Loading YOLOv8n base model...")
model = YOLO("yolov8n.pt")

print("Starting training on Kvasir-SEG dataset for 10 epochs...")
# Automatically uses Apple Silicon GPU (MPS) if available!
model.train(
    data="data.yaml",
    epochs=10,        # 10 epochs is a quick MVP prototype cycle
    imgsz=480,        # Match the resolution of our real-time UI
    batch=8,
    name="polyp_model_v1"
)

print("\nTraining complete! Exporting model...")
model_path = "runs/detect/polyp_model_v1/weights/best.pt"

if os.path.exists(model_path):
    shutil.copy(model_path, "polyp_model.pt")
    print("✅ Successfully exported 'polyp_model.pt' to the root directory!")
    print("🔥 The Streamlit app will now automatically detect it and use the real medical model.")
else:
    print("⚠️ Could not find the trained weights. Check the runs/ folder.")
