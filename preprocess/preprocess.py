# preprocess/preprocess.py
import os
import cv2
import numpy as np
import pickle

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
OUT_DIR = os.path.join(BASE_DIR, "preprocess", "features")
os.makedirs(OUT_DIR, exist_ok=True)
FEATURES_PATH = os.path.join(OUT_DIR, "features.pkl")

def extract_histogram(image_path, size=(256,256)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def main():
    if not os.path.exists(DATASET_DIR):
        print("Dataset folder not found:", DATASET_DIR)
        return

    features = []
    image_paths = []

    files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"Found {len(files)} images in dataset.")
    for fname in files:
        p = os.path.join(DATASET_DIR, fname)
        feat = extract_histogram(p)
        if feat is None:
            print("Skipping unreadable:", fname)
            continue
        features.append(feat)
        image_paths.append(fname)   # store basename for convenience
        print("Processed:", fname)

    with open(FEATURES_PATH, "wb") as f:
        # Save as tuple: (features_list, image_paths_list)
        pickle.dump((features, image_paths), f)

    print("Saved features to:", FEATURES_PATH)
    print("Total images processed:", len(image_paths))

if __name__ == "__main__":
    main()
