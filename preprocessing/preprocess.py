import os
import cv2

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raw_dir = os.path.join(base_dir, 'dataset', 'raw')
processed_dir = os.path.join(base_dir, 'dataset', 'processed')

# Ensure the processed directory exists
os.makedirs(processed_dir, exist_ok=True)

# List all image files in the raw directory (including subfolders)
image_files = []
for root, dirs, files in os.walk(raw_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(root, f))

# Check if there are any images to process
if not image_files:
    print("No images found in the raw directory.")
else:
    print(f"Found {len(image_files)} images to process.")

# Resize dimensions
resize_width = 256
resize_height = 256

for image_path in image_files:
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to read {image_path}. It may be corrupted or unsupported.")
        continue

    # Resize the image
    resized_image = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    # Create a consistent filename to save (flatten subdirectories)
    filename = os.path.basename(image_path)
    processed_image_path = os.path.join(processed_dir, filename)

    # Save the resized image
    success = cv2.imwrite(processed_image_path, resized_image)
    if success:
        print(f"Processed and saved {filename}")
    else:
        print(f"Failed to save {filename}")
