from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import base64
import io
from PIL import Image
import numpy as np
import os
import json

# --- ROBUST PATHING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(base_dir, "..", "faculty_db")
DATA_PATH = os.path.join(base_dir, "..", "faculty_data.json")

app = Flask(__name__, template_folder=os.path.join(base_dir, "..", "static"))

# --- CONFIGURATION ---
MODEL_NAME = "SFace" # Using the smallest and fastest model
DETECTOR_BACKEND = "opencv"

# Pre-load faculty data (this is small and safe to load at startup)
try:
    with open(DATA_PATH, 'r') as f:
        faculty_data = json.load(f)
    print("Successfully loaded faculty_data.json")
except FileNotFoundError:
    faculty_data = {}
    print(f"CRITICAL WARNING: {DATA_PATH} not found.")

# --- ROUTES ---

@app.route('/')
def home():
    """Serves the frontend HTML page."""
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """
    Receives an image, loads the model on-demand, finds a match, and returns the data.
    """
    if not request.json or 'image_data' not in request.json:
        return jsonify({'status': 'error', 'message': 'No image data received.'}), 400

    try:
        # Decode the Base64 image
        base64_string = request.json['image_data'].split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(pil_image)

        # *** MEMORY FIX: The model is now loaded and used only inside this function ***
        # This uses more CPU on first request, but keeps resting memory low.
        dfs = DeepFace.find(
            img_path=image_np,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            silent=True
        )

        # Process the results
        if isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
            df = dfs[0]
            top_match = df.iloc[0]
            identity_path = top_match['identity']
            
            safe_name = os.path.basename(identity_path).split('.')[0]
            details = faculty_data.get(safe_name)

            if details:
                return jsonify({'status': 'success', 'data': details})
            else:
                return jsonify({'status': 'not_found', 'message': 'Match found, but no details in JSON.'})
        else:
            return jsonify({'status': 'not_found', 'message': 'No match found in the database.'})

    except ValueError:
        return jsonify({'status': 'no_face', 'message': 'Could not detect a face in the image. Please try again.'})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
