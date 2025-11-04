import pickle

with open("features/features.pkl", "rb") as f:
    features = pickle.load(f)

print("✅ Total images:", len(features))
print("🔹 Checking first few feature vectors...\n")

for k, v in list(features.items())[:5]:
    print(f"Image: {k}")
    print(f"Type: {type(v)} | Length: {len(v) if hasattr(v, '__len__') else 'N/A'}\n")

# Check if all feature lengths are the same
lengths = [len(v) for v in features.values() if hasattr(v, '__len__')]
if len(set(lengths)) == 1:
    print(f"✅ All feature vectors have consistent length: {lengths[0]}")
else:
    print("⚠️ Inconsistent feature vector lengths detected:", set(lengths))
