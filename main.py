import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load model
model = joblib.load("models/number_gesture_model.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)
predictions = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        results = hands.process(image_rgb)
    except Exception as e:
        print("MediaPipe error:", e)
        continue

    number = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                    [lm.y for lm in hand_landmarks.landmark] + \
                    [lm.z for lm in hand_landmarks.landmark]
        X = np.array(landmarks).reshape(1, -1)

        try:
            number = int(model.predict(X)[0])
            predictions.append(number)
            if predictions:
                number = max(set(predictions), key=predictions.count)
            cv2.putText(frame, f"Number: {number}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        except Exception as e:
            print("Prediction error:", e)
    else:
        predictions.clear()
        cv2.putText(frame, "Hand not detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Hand Number Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
