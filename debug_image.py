import requests
from PIL import Image
import io
import numpy as np
import face_recognition

# The specific URL for the problematic image
IMAGE_URL = 'https://www.mmmut.ac.in/News_content/IMGFaculty29.jpg?637507973256744939'

print(f"--- Starting Debug for A. K. Mishra's Image ---")
print(f"URL: {IMAGE_URL}")

try:
    # 1. Download the image
    img_response = requests.get(IMAGE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    img_response.raise_for_status()
    print("[SUCCESS] Image downloaded.")

    # 2. Open from memory with Pillow
    image_stream = io.BytesIO(img_response.content)
    pil_image = Image.open(image_stream)
    print(f"[INFO] Pillow opened image. Original mode: '{pil_image.mode}', Format: '{pil_image.format}'")

    # 3. Convert to RGB
    rgb_pil_image = pil_image.convert('RGB')
    print(f"[INFO] Pillow converted image to mode: '{rgb_pil_image.mode}'")

    # 4. Convert to NumPy array
    image_np = np.array(rgb_pil_image)
    print(f"[INFO] Converted to NumPy array. Shape: {image_np.shape}, Data type: {image_np.dtype}")

    # 5. Check and enforce data type (just in case)
    if image_np.dtype != np.uint8:
        print(f"[WARNING] Data type is not uint8, converting from {image_np.dtype}...")
        image_np = image_np.astype(np.uint8)
        print(f"[INFO] Converted data type to: {image_np.dtype}")

    # 6. Attempt face encoding (the step that fails)
    print("\n>>> Attempting face_recognition.face_encodings()...")
    encodings = face_recognition.face_encodings(image_np)

    if encodings:
        print("\n[SUCCESS] Face encoding was successful!")
        print(f"Found {len(encodings)} face(s).")
    else:
        print("\n[FAILURE] No face was found in the image, but the processing did not crash.")

except Exception as e:
    print(f"\n[CRITICAL ERROR] The script crashed. Error message below:")
    print(e)