import os
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset" / "raw"

FEATURES_DIR = BASE_DIR / "features"
FEATURES_FILE = FEATURES_DIR / "features.pkl"

# Create folder if not exists
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Load pretrained ResNet50 model
try:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
except Exception:
    model = models.resnet50(pretrained=True)

model = nn.Sequential(*list(model.children())[:-1])  # remove final layer
model.eval()

# ✅ Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image_path):
    """Extract and normalize feature vector for an image."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    features = features.reshape(-1)
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features

def main():
    if not DATASET_DIR.exists():
        print(f"❌ Dataset folder not found: {DATASET_DIR}")
        return

    features_list = []
    image_paths = []

    print(f"🔍 Extracting features from images in: {DATASET_DIR}")

    for file in sorted(DATASET_DIR.iterdir()):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                feat = extract_features(str(file))
                features_list.append(feat)
                image_paths.append(str(file))
                print(f"✅ Processed: {file.name}")
            except Exception as e:
                print(f"⚠️ Skipped {file.name}: {e}")

    # Save features and paths together
    with open(FEATURES_FILE, "wb") as f:
        pickle.dump((features_list, image_paths), f)

    print(f"\n🎯 Features extracted and saved to: {FEATURES_FILE}")
    print(f"Total images processed: {len(features_list)}")

if __name__ == "__main__":
    main()
