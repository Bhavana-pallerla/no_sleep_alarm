# Top-level imports
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from PIL import Image
from tensorflow.keras.models import load_model
import pygame

# Load the trained model
model = tf.keras.models.load_model("eye_detection_model.keras")

# Initialize Pygame mixer for alarm
pygame.mixer.init()
pygame.mixer.music.load("C:/Users/Bhavana/no_sleep_alarm/alarm.wav")  # Make sure this file exists

# Mediapipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Eye landmark indices
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]

# Alarm logic variables
closed_counter = 0
CLOSED_THRESHOLD = 10  # Adjust based on sensitivity

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        face_landmarks = results.multi_face_landmarks[0]

        # Get eye bounding boxes
        left_eye_pts = [(int(face_landmarks.landmark[idx].x * w),
                         int(face_landmarks.landmark[idx].y * h)) for idx in LEFT_EYE_IDX]
        right_eye_pts = [(int(face_landmarks.landmark[idx].x * w),
                          int(face_landmarks.landmark[idx].y * h)) for idx in RIGHT_EYE_IDX]

        def crop_eye(img, eye_pts):
            x1, y1 = eye_pts[0]
            x2, y2 = eye_pts[1]
            margin = 10
            x_min = max(min(x1, x2) - margin, 0)
            y_min = max(min(y1, y2) - margin, 0)
            x_max = min(max(x1, x2) + margin, img.shape[1])
            y_max = min(max(y1, y2) + margin, img.shape[0])
            return img[y_min:y_max, x_min:x_max]

        left_eye_img = crop_eye(frame, left_eye_pts)
        right_eye_img = crop_eye(frame, right_eye_pts)

        def preprocess_eye(eye_img):
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (82, 82))  # Match training size
            normalized = resized / 255.0
            return np.expand_dims(normalized, axis=(0, -1))  # Shape: (1, 82, 82, 1)

        try:
            left_input = preprocess_eye(left_eye_img)
            right_input = preprocess_eye(right_eye_img)

            left_prob = model.predict(left_input, verbose=0)[0]
            right_prob = model.predict(right_input, verbose=0)[0]

            left_state = "Open" if left_prob[1] > left_prob[0] else "Closed"
            right_state = "Open" if right_prob[1] > right_prob[0] else "Closed"

            # Display predictions
            cv2.putText(frame, f"Left: {left_state} ({left_prob[1]:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Right: {right_state} ({right_prob[1]:.2f})", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Alarm logic
            if left_state == "Closed" or right_state == "Closed":
                closed_counter += 1
            else:
                closed_counter = 0
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

            if closed_counter > CLOSED_THRESHOLD:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()

        except Exception as e:
            print(f"Error processing eye: {e}")

    cv2.imshow("Eye State Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
