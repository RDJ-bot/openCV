from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import numpy as np
import os
from PIL import Image
import pyttsx3
from datetime import datetime
import serial
import requests
import time

# Voice setup
engine = pyttsx3.init()
engine.setProperty('rate', 140)
engine.say("GuardianEye activated. Monitoring zone.")
engine.runAndWait()

# Telegram setup
BOT_TOKEN = '8361461500:AAEWF7UszvsSDeLHbeUEKiLHhkHOB_u6Lvw'
CHAT_ID = '6743445373'

def send_telegram_alert(image_path=None):
    url_msg = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    url_photo = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'

    try:
        # Send alert message
        msg_data = {'chat_id': CHAT_ID, 'text': '🚨 Intruder Detected by GuardianEye!'}
        msg_response = requests.post(url_msg, data=msg_data)

        # Optionally send the image
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as photo:
                photo_data = {'chat_id': CHAT_ID}
                photo_response = requests.post(url_photo, data=photo_data, files={'photo': photo})
                if not photo_response.ok:
                    print("❌ Failed to send photo.")
        if not msg_response.ok:
            print("❌ Failed to send Telegram alert.")
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# Serial communication setup
try:
    arduino = serial.Serial('COM6', 9600, timeout=1)  # Adjust as per your port
    time.sleep(2)
except Exception as e:
    print(f"Serial connection failed: {e}")
    arduino = None

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known faces
known_faces_dir = 'known_faces'
known_embeddings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        name = os.path.splitext(filename)[0]
        img = Image.open(os.path.join(known_faces_dir, filename)).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0).to(device))
            known_embeddings.append(embedding)
            known_names.append(name)

# Create intruder folder if needed
os.makedirs("intruder", exist_ok=True)

def recognize_face(face_img):
    face_img = face_img.convert('RGB')
    face_tensor = mtcnn(face_img)
    if face_tensor is not None:
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(device))
        min_dist = float('inf')
        identity = "Unknown"

        for i, known_embedding in enumerate(known_embeddings):
            dist = (embedding - known_embedding).norm().item()
            if dist < min_dist:
                min_dist = dist
                identity = known_names[i]

        return identity if min_dist < 0.9 else "Intruder"
    return "No Face Detected"

# Start webcam
cap = cv2.VideoCapture(0)
intruder_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()  # Keep original BGR for display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Use RGB for recognition
    img = Image.fromarray(rgb_frame)
    identity = recognize_face(img)

    if identity != "No Face Detected":
        if identity != "Intruder":
            print(f"{identity} detected. Searching for intruders.")
            engine.say(f"{identity} detected.")
            intruder_triggered = False
            if arduino:
                arduino.write(b'0')  # No intrusion
        else:
            print("🚨 Intruder detected!")
            engine.say("Intruder detected!")

            if not intruder_triggered:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"intruder/intruder_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Intruder face saved: {filename}")

                if arduino:
                    arduino.write(b'1')

                send_telegram_alert(filename)
                intruder_triggered = True

        engine.runAndWait()

    # Draw label
    cv2.putText(display_frame, identity, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if identity == "Intruder" else (0, 255, 0), 2)

    cv2.imshow('GuardianEye Surveillance', display_frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()