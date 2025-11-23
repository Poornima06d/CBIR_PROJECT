import os
import cv2
import pickle
import numpy as np
import shutil
from sklearn.metrics.pairwise import cosine_similarity

# ---------- load features (supports dict or tuple) ----------
FEATURES_PATH = "preprocess/features/features.pkl"
if not os.path.exists(FEATURES_PATH):
    print("❌ No features found. Run 'python preprocess/preprocess.py' first.")
    exit()

with open(FEATURES_PATH, "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    features = np.array(data.get("features"))
    image_paths = data.get("image_paths", [])
else:
    features, image_paths = data
    features = np.array(features)

if features is None or len(image_paths) == 0:
    print("❌ features.pkl missing expected data. Regenerate features.")
    exit()

# ---------- helper: histogram extractor (must match preprocessing) ----------
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Unable to read image: {image_path}")
        return None

    image = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# ---------- read query path (strip quotes) ----------
raw = input("📸 Enter full path of query image: ").strip()
query_path = raw.strip('"').strip("'")
if not os.path.exists(query_path):
    print("❌ Query image not found:", query_path)
    exit()

query_features = extract_features(query_path)

if query_features is None:
    print("⚠️ Could not read or process query image.")
    exit()

# ---------- compute similarities and show top-K preview ----------
sims = cosine_similarity([query_features], features)[0]
sorted_idx = np.argsort(sims)[::-1]

preview_k = min(20, len(sorted_idx))
print(f"\n🔍 Top {preview_k} matches (score range {sims.min():.3f} .. {sims.max():.3f}):\n")
for i in range(preview_k):
    idx = sorted_idx[i]
    print(f"{i+1:2d}. {os.path.basename(image_paths[idx])}  —  score: {sims[idx]:.4f}")

# ---------- let user choose mode ----------
print("\nChoose retrieval mode:")
print("1) Threshold mode — return all images with similarity >= T")
print("2) Top-K mode — pick the top N images (regardless of score)")
mode = input("Enter 1 or 2 (default 2): ").strip() or "2"

chosen_indices = []
if mode == "1":
    # suggest a reasonable default based on distribution
    suggested = float(np.median(sims) + 0.1)
    print(f"Suggested threshold (heuristic): {suggested:.3f}")
    t_str = input(f"Enter threshold T (0..1), e.g. {suggested:.3f}: ").strip()
    try:
        T = float(t_str) if t_str != "" else suggested
    except:
        T = suggested
    chosen_indices = [i for i, s in enumerate(sims) if s >= T]
    chosen_indices = sorted(chosen_indices, key=lambda i: sims[i], reverse=True)
    print(f"\n🔎 Found {len(chosen_indices)} images with similarity >= {T:.3f}")
    if len(chosen_indices) == 0:
        print("No images passed the threshold. Try lowering T or use top-K mode.")
        cont = input("Switch to top-K mode? (y/n) ").strip().lower()
        if cont == "y":
            mode = "2"
        else:
            print("Exiting.")
            exit()

if mode == "2":
    n_str = input("Enter how many top images you want to retrieve (e.g. 5): ").strip()
    try:
        N = int(n_str)
        if N <= 0:
            raise ValueError
    except:
        N = 5
        print("Invalid input — defaulting to top 5.")
    chosen_indices = list(sorted_idx[:min(N, len(sorted_idx))])

# ---------- show chosen results ----------
if not chosen_indices:
    print("No images selected. Exiting.")
    exit()

print(f"\n✅ Returning {len(chosen_indices)} images. First 10:")
for i, idx in enumerate(chosen_indices[:10], 1):
    print(f"{i}. {image_paths[idx]}  (score: {sims[idx]:.4f})")

# ---------- optional visual preview ----------
show = input("\nShow thumbnails now? (y/n, default n): ").strip().lower() == "y"
if show:
    for idx in chosen_indices:
        img = cv2.imread(image_paths[idx])
        if img is None:
            continue
        thumb = cv2.resize(img, (400, 400))
        cv2.imshow("Result", thumb)
        key = cv2.waitKey(600)
        if key == 27:
            break
    cv2.destroyAllWindows()

# ---------- save selected images ----------
save = input("\nSave these images to 'downloaded_results/'? (y/n): ").strip().lower() == "y"
if save:
    os.makedirs("downloaded_results", exist_ok=True)
    for idx in chosen_indices:
        src = image_paths[idx]
        dst = os.path.join("downloaded_results", os.path.basename(src))
        shutil.copy(src, dst)
    print(f"📥 Saved {len(chosen_indices)} images to 'downloaded_results/'")

# ---------- feedback ----------
fb = input("\nAre you satisfied with the results? (yes/no): ").strip().lower()
if fb == "yes":
    print("🙂 Thanks for the feedback!")
elif fb == "no":
    reason = input("Please type a short reason (optional): ").strip()
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Query: {os.path.basename(query_path)}  — Feedback: {reason}\n")
    print("Saved feedback to feedback_log.txt")

print("\nAll done — run again to search another image.")
