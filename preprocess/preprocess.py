import os
import cv2
import numpy as np
import pickle

# Paths
dataset_path = "dataset/raw"
features_dir = "preprocess/features"
features_path = os.path.join(features_dir, "features.pkl")

# Ensure feature directory exists
os.makedirs(features_dir, exist_ok=True)

# Check dataset
if not os.path.exists(dataset_path):
    print(f"❌ Dataset folder not found: {os.path.abspath(dataset_path)}")
    exit()

features = []
image_paths = []

print(f"🔍 Extracting features from images in: {os.path.abspath(dataset_path)}")

# Loop through dataset images
for img_name in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue

    # Resize and extract HSV color histogram
    image = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    features.append(hist.flatten())
    image_paths.append(img_path)
    print(f"✅ Processed: {img_name}")

# Save features
with open(features_path, "wb") as f:
    pickle.dump({"features": np.array(features), "image_paths": image_paths}, f)

print(f"\n🎯 Features extracted and saved to: {os.path.abspath(features_path)}")
print(f"Total images processed: {len(image_paths)}")
