import os
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import json
import zipfile

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, "features", "features.pkl")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_t).squeeze().numpy()
    return features / np.linalg.norm(features)

# Load features
with open(FEATURES_PATH, "rb") as f:
    features_dict = pickle.load(f)

# Convert dict to lists for easier search
file_names = list(features_dict.keys())
feature_vectors = np.array(list(features_dict.values()))

def find_similar_images(query_path, top_n=10):
    query_feat = extract_features(query_path)
    similarities = np.dot(feature_vectors, query_feat)
    indices = np.argsort(similarities)[::-1][:top_n]
    return [os.path.basename(file_names[i]) for i in indices]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    image = request.files["query_image"]
    top_n = int(request.form.get("top_n", 10))

    query_path = os.path.join(RESULTS_FOLDER, image.filename)
    image.save(query_path)

    results = find_similar_images(query_path, top_n)

    for res in results:
        src = os.path.join(DATASET_DIR, res)
        dst = os.path.join(RESULTS_FOLDER, res)
        if os.path.exists(src):
            Image.open(src).save(dst)

    return render_template("results.html", query=image.filename, results=results)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    image = data["image"]
    is_relevant = data["feedback"]

    feedback_file = os.path.join(BASE_DIR, "feedback.json")
    feedback_data = []

    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)

    feedback_data.append({"image": image, "relevant": is_relevant})

    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)

    return jsonify({"status": "success", "message": f"Feedback saved for {image}"})

@app.route("/download", methods=["POST"])
def download():
    data = request.get_json()
    images = data.get("images", [])

    zip_path = os.path.join(RESULTS_FOLDER, "similar_images.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for img in images:
            img_path = os.path.join(RESULTS_FOLDER, img)
            if os.path.exists(img_path):
                zipf.write(img_path, img)
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
