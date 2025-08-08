import cv2
import numpy as np
import mediapipe as mp
import math
import serial
import time
import requests

# EAR calculation function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# EAR threshold and frame count
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 30
frame_counter = 0
drowsiness_triggered = False

# Start video
cap = cv2.VideoCapture(0)

# Indices for eye landmarks (Mediapipe FaceMesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Telegram Bot Setup
BOT_TOKEN = '8361461500:AAEWF7UszvsSDeLHbeUEKiLHhkHOB_u6Lvw'
CHAT_ID = '6743445373'

def send_telegram_alert():
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': '⚠️ Drowsiness detected! Please take a break!'}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Telegram Error:", e)

# Serial Communication
try:
    arduino = serial.Serial('COM6', 9600, timeout=1)
    time.sleep(2)
except Exception as e:
    print("Serial Connection Error:", e)
    arduino = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            for pt in left_eye + right_eye:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not drowsiness_triggered:
                        if arduino:
                            arduino.write(b'1')  # Trigger buzzer/light
                        send_telegram_alert()
                        drowsiness_triggered = True
            else:
                frame_counter = 0
                drowsiness_triggered = False
                if arduino:
                    arduino.write(b'0')  # Turn off alert

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        brea

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()