import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

DATASET_DIR = "data/raw"
OUTPUT_CSV = "data/processed/landmarks.csv"

os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)

    # CSV header
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    writer.writerow(header)

    for label in range(10):
        folder_path = os.path.join(DATASET_DIR, str(label))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                row = []
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(label)
                writer.writerow(row)

print("✅ Landmark extraction completed!")