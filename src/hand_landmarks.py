import cv2
import mediapipe as mp
import os
import csv

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

DATA_DIR = "data/raw"
OUTPUT_FILE = "data/processed/landmarks.csv"

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)

    # CSV Header
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")
    writer.writerow(header)

    for label in range(10):
        folder = os.path.join(DATA_DIR, str(label))

        if not os.path.exists(folder):
            continue

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                row = []

                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)
                writer.writerow(row)

print("✅ Landmarks extracted and saved to CSV")
