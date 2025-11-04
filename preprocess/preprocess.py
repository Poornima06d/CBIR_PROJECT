import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# -----------------------------
# ✅ Folder setup
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
FEATURES_PATH = os.path.join(BASE_DIR, "features", "features.pkl")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)

# -----------------------------
# ✅ Load pretrained ResNet50
# -----------------------------
print("🔹 Loading ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classification layer
model.eval()

# -----------------------------
# ✅ Image preprocessing pipeline
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image_path):
    """Extract normalized 2048-D feature vector for one image"""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = model(img_tensor).squeeze().numpy()
        features = features / np.linalg.norm(features)  # normalize

        return features
    except Exception as e:
        print(f"⚠️ Skipping {image_path} due to error: {e}")
        return None

# -----------------------------
# ✅ Extract and save features
# -----------------------------
features_dict = {}

print("🔹 Extracting features from images...")

for file in tqdm(os.listdir(RAW_DIR)):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(RAW_DIR, file)
        feat = extract_features(img_path)
        if feat is not None and len(feat) == 2048:
            features_dict[file] = feat
        else:
            print(f"⚠️ Skipped {file}: invalid feature shape.")

# -----------------------------
# ✅ Save to pickle
# -----------------------------
with open(FEATURES_PATH, "wb") as f:
    pickle.dump(features_dict, f)

print(f"✅ Features extracted and saved to '{FEATURES_PATH}'")
print(f"📦 Total valid images processed: {len(features_dict)}")
