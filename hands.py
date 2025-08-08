import cv2
import mediapipe as mp
import serial
import sys
import time

# === CONFIGURATION ===
USE_HARDWARE = True  # Change to True if Arduino is connected
SERIAL_PORT = 'COM7'  # Update this to match your Arduino's port
BAUD_RATE = 9600

# === SERIAL SETUP ===
if USE_HARDWARE:
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to Arduino on {SERIAL_PORT}")
        time.sleep(2)
    except Exception as e:
        print(f"[ERROR] Failed to connect to Arduino: {e}")
        sys.exit(1)

# === MEDIAPIPE SETUP ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# === CAMERA SETUP ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    sys.exit(1)

# === FUNCTION TO DETECT FINGER STATES ===
def get_finger_states(hand_landmarks, hand_label):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other fingers (Index to Pinky)
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# === CLASSIFY GESTURE ===
def classify_gesture(fingers):
    gestures = {
        (1, 1, 1, 1, 1): ('Palm', '1'),
        (0, 0, 0, 0, 0): ('Fist', '2'),
        (0, 1, 1, 0, 0): ('Two Fingers', '3'),
        (1, 0, 0, 0, 0): ('Thumbs Up', '4'),
        (0, 1, 0, 0, 0): ('Index Finger', '5'),
        (1, 1, 0, 0, 1): ('Rock Sign', '6'),
        (1, 0, 0, 0, 1): ('Call Me', '7')
    }
    return gestures.get(tuple(fingers), ('Unknown', '0'))

# === MAIN LOOP ===
last_command_sent = '0'

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = get_finger_states(hand_landmarks, hand_label)
                gesture, command = classify_gesture(fingers)

                cv2.putText(frame, f'{gesture}', (10, 50 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                if command != last_command_sent:
                    last_command_sent = command

                    if command != '0':
                        if USE_HARDWARE:
                            try:
                                arduino.write(command.encode())
                                print(f"[INFO] Sent Command ➝ {gesture} ({command})")
                            except Exception as e:
                                print(f"[ERROR] Serial write failed: {e}")
                        else:
                            print(f"[SIM] Gesture ➝ {gesture} | Command ➝ {command}")

        else:
            if last_command_sent != '0':
                last_command_sent = '0'
                if USE_HARDWARE:
                    try:
                        arduino.write(b'0')
                        print("[INFO] No hand detected, sent reset command.")
                    except Exception as e:
                        print(f"[ERROR] Serial write failed: {e}")

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Program interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if USE_HARDWARE and 'arduino' in locals():
        arduino.close()
        print("[INFO] Arduino connection closed.")