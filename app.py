# app.py
import os
import pickle
import numpy as np
import shutil
import zipfile
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
FEATURES_PATH = os.path.join(BASE_DIR, "preprocess", "features", "features.pkl")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED

# histogram feature extractor (must match preprocess)
def extract_histogram_from_bgr_bytedata(image_path, size=(256,256)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# load features (tuple format saved by preprocess.py)
if not os.path.exists(FEATURES_PATH):
    print("⚠ features.pkl not found. Run: python preprocess/preprocess.py")
    features_list, image_paths = [], []
else:
    with open(FEATURES_PATH, "rb") as f:
        data = pickle.load(f)
    # expected format: (features_list, image_paths_list)
    if isinstance(data, tuple) and len(data) == 2:
        features_list, image_paths = data
        features_arr = np.array(features_list)
    else:
        # fallback: try to parse dict style
        try:
            # if dict mapping basename -> vector
            features_arr = np.array(list(data.values()))
            image_paths = list(data.keys())
        except Exception:
            features_arr, image_paths = np.array([]), []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    if "query_image" not in request.files:
        return "No file part", 400

    file = request.files["query_image"]
    if file.filename == "":
        return "No selected file", 400
    if not allowed_file(file.filename):
        return "File type not allowed", 400

    top_n = int(request.form.get("top_n", 10))
    filename = secure_filename(file.filename)
    query_save_path = os.path.join(RESULTS_FOLDER, filename)
    file.save(query_save_path)

    query_feat = extract_histogram_from_bgr_bytedata(query_save_path)
    if query_feat is None:
        return "Unable to read query image", 400

    if len(image_paths) == 0 or len(features_arr) == 0:
        return "No features available. Run preprocess/preprocess.py", 500

    # cosine similarity
    sims = np.dot(features_arr, query_feat) / (np.linalg.norm(features_arr, axis=1) * np.linalg.norm(query_feat) + 1e-10)
    idxs = np.argsort(sims)[::-1][:top_n]

    results = []
    for i in idxs:
        name = image_paths[i]
        score = float(sims[i])
        src = os.path.join(DATASET_DIR, name)
        dst = os.path.join(RESULTS_FOLDER, name)
        if os.path.exists(src):
            shutil.copy(src, dst)
        results.append({"name": name, "score": round(score, 4)})

    return render_template("results.html", query=filename, results=results, total=len(results))

@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()
    images = data.get("images", [])
    if not images:
        return jsonify({"error": "no images selected"}), 400

    zip_path = os.path.join(RESULTS_FOLDER, "similar_images.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for img in images:
            p = os.path.join(RESULTS_FOLDER, img)
            if os.path.exists(p):
                z.write(p, arcname=img)
    return send_file(zip_path, as_attachment=True)

@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.get_json()
    # store simple feedback log as JSON lines
    fb_path = os.path.join(BASE_DIR, "feedback_log.jsonl")
    with open(fb_path, "a", encoding="utf-8") as f:
        f.write(jsonify(payload).get_data(as_text=True) + "\n")
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
