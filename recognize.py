import cv2
from deepface import DeepFace
import os
import time
import json
import numpy as np

# --- CONFIGURATION ---
DB_PATH = "faculty_db"
MODEL_NAME = "VGG-Face" 
DETECTOR_BACKEND = "opencv"

# --- END OF CONFIGURATION ---

# Load faculty data from JSON file
try:
    with open('faculty_data.json', 'r') as f:
        faculty_data = json.load(f)
except FileNotFoundError:
    print("Error: 'faculty_data.json' not found. Please run the 'build_database.py' script first.")
    exit()

if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
    print(f"Error: Database folder '{DB_PATH}' is empty or does not exist.")
    exit()

print("Loading face recognition model...")
try:
    DeepFace.find(img_path=np.zeros((100, 100, 3), dtype=np.uint8), db_path=DB_PATH, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
except Exception:
    pass
print("Model loaded successfully.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_check_time = time.time()
check_interval = 2 # seconds
last_known_faces = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    current_time = time.time()
    
    if current_time - last_check_time > check_interval:
        last_check_time = current_time
        try:
            dfs = DeepFace.find(
                img_path=frame,
                db_path=DB_PATH,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True
            )
            
            current_faces = []
            if isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
                df = dfs[0]
                for _, row in df.iterrows():
                    identity_path = row['identity']
                    safe_name = os.path.basename(identity_path).split('.')[0]
                    
                    # Look up details from our loaded JSON data
                    details = faculty_data.get(safe_name)
                    if details:
                        x, y, w, h = row['source_x'], row['source_y'], row['source_w'], row['source_h']
                        current_faces.append((details, (x, y, w, h)))
            
            last_known_faces = current_faces

        except Exception as e:
            last_known_faces = []
            pass

    # Draw the boxes and names from the last successful check
    for details, (x, y, w, h) in last_known_faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Create a background for the text
        text_y = y + h + 45 # Position text below the box
        cv2.rectangle(frame, (x, y + h), (x + w, text_y), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Display the details
        cv2.putText(frame, details['full_name'], (x + 6, y + h + 15), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, details['designation'], (x + 6, y + h + 30), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, details['department'], (x + 6, y + h + 42), font, 0.4, (255, 255, 255), 1)


    cv2.imshow('Real-Time Faculty Recognition (DeepFace)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
