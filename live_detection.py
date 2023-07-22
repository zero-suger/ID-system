import torch
import cv2
import time
import numpy as np
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase
cred = credentials.Certificate("Firebase-admin-conifs.json")  # Replace with your service account key file
firebase_admin.initialize_app(cred, {'databaseURL': 'https://id-verify-3bc2f-default-rtdb.asia-southeast1.firebasedatabase.app/'})

# Load stored face embeddings from Firebase
ref = db.reference('face_embeddings')  # Replace with the path to your face embeddings in Firebase
stored_embeddings = ref.get()

# Load the YOLO model for object detection
model = YOLO('/home/suger01/Desktop/YOLOv8_FD/ultralytics/runs/detect/train9/weights/merged_data.pt')

# Set up camera details
camera_ip = '192.168.0.22'
username = 'root'
password = 'kiicti'

# Construct the RTSP URL
rtsp_url = f'rtsp://{username}:{password}@{camera_ip}/axis-media/media.amp'
my_camera = cv2.VideoCapture(rtsp_url)

if not my_camera.isOpened():
    raise Exception("No Camera")

while True:
    ret, image = my_camera.read()
    if not ret:
        break

    _time_mulai = time.time()

    # Perform object detection on the image
    result = model.predict(image, show=False)

    # Retrieve detected faces from the object detection results
    faces = result.xyxy[0].detach().numpy()

    # Iterate over detected faces
    for face in faces:
        # Extract face region coordinates
        x, y, w, h, _ = face

        # Extract the face region from the frame
        face_img = image[int(y):int(y+h), int(x):int(x+w)]

        # Preprocess the face image (resize, normalize, etc.)
        # ...

        # Extract face embedding from the preprocessed image using your face embedding model
        # ...

        # Compare the extracted embedding with stored embeddings
        for name, stored_embedding in stored_embeddings.items():
            # Calculate similarity score between the extracted embedding and stored embeddings
            similarity_score = np.dot(extracted_embedding, stored_embedding)

            # Set a threshold for matching
            if similarity_score > 0.8:  # Adjust the threshold as per your requirements
                # Face matched! Do something (e.g., display a match notification, perform an action associated with the recognized individual)
                print(f"Match found: {name}")

    print('Urinov Azizbek', time.time()-_time_mulai)

    _key = cv2.waitKey(1)

    if _key == ord('q'):
        break

my_camera.release()
cv2.destroyAllWindows()
