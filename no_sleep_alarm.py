import cv2
import numpy as np
import mediapipe as mp
import pygame
from keras.models import load_model

# Load the trained model
model = load_model("eye_state_model.keras")

# Initialize Pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Add a short beep sound file

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Define eye landmark indexes for cropping
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]

# Capture webcam
cap = cv2.VideoCapture(0)

def preprocess_eye(eye_img):
    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(eye_gray, (82, 82))  # Match model input size
    normalized = resized / 255.0
    return normalized.reshape(1, 82, 82, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            # Get coordinates for both eyes
            left_eye_pts = []
            right_eye_pts = []

            for idx in LEFT_EYE_IDX:
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                left_eye_pts.append((x, y))

            for idx in RIGHT_EYE_IDX:
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                right_eye_pts.append((x, y))

            def crop_eye(pts):
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                margin = 10
                return frame[max(0, y1-margin):min(h, y2+margin),
                             max(0, x1-margin):min(w, x2+margin)]

            left_eye_img = crop_eye(left_eye_pts)
            right_eye_img = crop_eye(right_eye_pts)

            # Predict if both eyes are closed
            left_pred = model.predict(preprocess_eye(left_eye_img), verbose=0)[0][0]
            right_pred = model.predict(preprocess_eye(right_eye_img), verbose=0)[0][0]

            # Assume threshold: closed if prediction < 0.5
            if left_pred < 0.5 and right_pred < 0.5:
                cv2.putText(frame, "DROWSY!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                pygame.mixer.music.stop()

    cv2.imshow("No Sleep Alarm", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
