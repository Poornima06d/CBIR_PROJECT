import os
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import shutil

# ✅ Load features and image paths
FEATURES_PATH = "features/features.pkl"

if not os.path.exists(FEATURES_PATH):
    print("❌ No features found. Please run 'python preprocess/preprocess.py' first.")
    exit()

with open(FEATURES_PATH, "rb") as f:
    features, image_paths = pickle.load(f)

features = np.array(features)

print("✅ Features loaded successfully.")
print("Total images in dataset:", len(features))

# ✅ Load pretrained ResNet50 (same as used in preprocessing)
model = models.resnet50(weights="IMAGENET1K_V1")
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# ✅ Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image_path):
    """Extract feature vector from a single image"""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        feat = model(image).squeeze().numpy()
    return feat

# === Main Logic ===
query_path = input("\nEnter the path of query image (e.g., dataset/raw/flower1.jpg): ").strip()

if not os.path.exists(query_path):
    print("❌ Image not found. Check your path.")
    exit()

top_n = int(input("\nEnter how many top similar images to display: "))

# Extract features for query image
query_features = extract_features(query_path)

# Calculate cosine similarity
similarities = cosine_similarity([query_features], features)[0]
sorted_indices = np.argsort(similarities)[::-1][:top_n]

# Display results
print("\n🔍 Top", top_n, "most similar images:\n")
for i, idx in enumerate(sorted_indices):
    print(f"{i+1}. {image_paths[idx]} (similarity: {similarities[idx]:.4f})")

# === ⭐ Feedback Section ===
feedback = input("\nWould you like to give feedback? (y/n): ").strip().lower()
if feedback == "y":
    print("\nPlease rate each retrieved image from 1⭐ to 5⭐:")
    for i, idx in enumerate(sorted_indices):
        rating = input(f"⭐ {image_paths[idx]}: ")
        print(f"✅ You rated {rating} stars for {os.path.basename(image_paths[idx])}")

# === 📥 Download Option ===
download_choice = input("\nDo you want to save these similar images? (y/n): ").strip().lower()
if download_choice == "y":
    os.makedirs("downloaded_results", exist_ok=True)
    for i, idx in enumerate(sorted_indices):
        src = image_paths[idx]
        dest = os.path.join("downloaded_results", os.path.basename(src))
        shutil.copy(src, dest)
    print("📥 All similar images saved in 'downloaded_results' folder.")

print("\n🎉 Search completed successfully!")
