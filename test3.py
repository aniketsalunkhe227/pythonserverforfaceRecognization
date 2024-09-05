import os
import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

print("All libraries imported successfully!")

app = Flask(__name__)

def load_images_from_folder(folder):
    """Load all images from a specified folder."""
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def encode_face(image, image_name):
    """Encode faces detected in an image using DeepFace."""
    try:
        faces = DeepFace.extract_faces(img_path=image, detector_backend='opencv', enforce_detection=False)
        num_faces = len(faces)

        if num_faces == 0:
            print(f"No faces detected in {image_name}.")
            return None, 0
        else:
            print(f"Detected {num_faces} face(s) in {image_name}. Encoding the first detected face...")
            embedding = DeepFace.represent(img_path=image, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            return np.array(embedding), num_faces

    except Exception as e:
        print(f"Error encoding face in {image_name}: {e}")
        return None, 0

def match_faces(user_encoding, folder_encodings, folder_filenames, tolerance=10.0):
    """Match the user face encoding with faces from the folder encodings."""
    matched_filenames = []
    for encoding, filename in zip(folder_encodings, folder_filenames):
        distance = np.linalg.norm(user_encoding - encoding)
        if distance < tolerance:
            matched_filenames.append(filename)
    return matched_filenames

def download_image(image_url):
    """Download image from URL and return it as a NumPy array."""
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)
        return img
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None

@app.route('/match-images', methods=['POST'])
def match_images():
    data = request.json

    reference_image_url = data.get('reference_image')
    image_urls = data.get('image_urls')

    if not reference_image_url or not image_urls:
        return jsonify({"error": "Please provide both reference_image and image_urls."}), 400

    reference_image = download_image(reference_image_url)
    if reference_image is None:
        return jsonify({"error": "Failed to download reference image."}), 400

    reference_encoding, num_faces = encode_face(reference_image, "reference_image")
    if reference_encoding is None:
        return jsonify({"error": "No faces detected in the reference image."}), 400

    matched_image_urls = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, url) for url in image_urls]
        for future in as_completed(futures):
            image_url = image_urls[futures.index(future)]
            image = future.result()
            if image is None:
                print(f"Skipping {image_url}")
                continue

            encoding, num_faces = encode_face(image, image_url)
            if encoding is not None:
                distance = np.linalg.norm(reference_encoding - encoding)
                if distance < 10.0:
                    matched_image_urls.append(image_url)

    return jsonify({"matched_images": matched_image_urls})

def main():
    user_image_path = "./gallery/IMG_4928.jpg"
    user_image = cv2.imread(user_image_path)
    if user_image is None:
        print(f"User image not found at {user_image_path}. Please check the path.")
        return

    user_encoding, num_faces = encode_face(user_image, user_image_path)
    if user_encoding is None or num_faces == 0:
        print("No face detected in the user image.")
        return
    elif num_faces > 1:
        print("Multiple faces detected in the user image. Please provide an image with a single face.")
        return
    else:
        print(f"Number of faces detected in user image: {num_faces}")

    folder_path = "./gallery"
    folder_images, folder_filenames = load_images_from_folder(folder_path)

    folder_encodings = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(encode_face, img, filename) for img, filename in zip(folder_images, folder_filenames)]
        for future in as_completed(futures):
            encoding, num_faces = future.result()
            if encoding is not None:
                folder_encodings.append(encoding)
            else:
                print(f"No face detected in {folder_filenames[futures.index(future)]}.")

    matched_filenames = match_faces(user_encoding, folder_encodings, folder_filenames)

    if matched_filenames:
        print("Matched images containing the same person as in user image:")
        for filename in matched_filenames:
            print(filename)
    else:
        print("No matching images found.")

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
