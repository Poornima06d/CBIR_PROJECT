import os
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
FEATURES_PATH = os.path.join(BASE_DIR, "features", "features.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load pre-extracted features
with open(FEATURES_PATH, "rb") as f:
    features_dict = pickle.load(f)

# CNN model for feature extraction
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

model = load_model()

# Preprocess image
def extract_features(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(img_tensor).squeeze().numpy()
    return feature

# Find top similar images
def find_similar_images(query_img_path, top_n=10):
    query_feature = extract_features(query_img_path)
    all_features = np.array(list(features_dict.values()))
    all_image_paths = list(features_dict.keys())

    similarities = cosine_similarity([query_feature], all_features)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    top_images = [(all_image_paths[i], similarities[i]) for i in sorted_indices[:top_n]]
    return top_images

# Download selected similar images
def download_similar_images(top_images, count=5):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for i, (img_path, sim) in enumerate(top_images[:count]):
        dest_path = os.path.join(RESULTS_DIR, f"similar_{i+1}.jpg")
        shutil.copy(img_path, dest_path)
    print(f"✅ {count} similar images saved in 'results/' folder.")

# --- Main Program ---
if __name__ == "__main__":
    print("🔍 Content-Based Image Retrieval System")
    query_img = input("Enter path of the query image (from dataset/raw): ").strip()
    top_n = int(input("Enter how many top similar images to search (e.g., 10): "))
    download_count = int(input("How many images to download (e.g., 5): "))

    results = find_similar_images(query_img, top_n=top_n)

    print("\nTop similar images:")
    for i, (path, sim) in enumerate(results):
        print(f"{i+1}. {path} (Similarity: {sim:.4f})")

    download_similar_images(results, download_count)
