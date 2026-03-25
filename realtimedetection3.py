import cv2
import os
from datetime import datetime
import time
import csv
import psycopg2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import numpy as np

# -------------------------
# PostgreSQL connection
# -------------------------
conn = psycopg2.connect(
    host="localhost",
    database="face_monitoring",
    user="postgres",
    password="2005"
)
cursor = conn.cursor()

# -------------------------
# Logging functions
# -------------------------
def log_leave(person_name, leave_time):
    print(f"[LEAVE] {person_name} at {leave_time}")
    cursor.execute(
        "INSERT INTO break_logs (name, leave_time) VALUES (%s, %s)",
        (person_name, leave_time)
    )
    conn.commit()

def log_return(person_name, return_time, duration):
    print(f"[RETURN] {person_name} at {return_time}, duration={duration}s")
    cursor.execute(
        """
        UPDATE break_logs
        SET return_time=%s, duration_seconds=%s
        WHERE name=%s AND return_time IS NULL
        """,
        (return_time, duration, person_name)
    )
    conn.commit()

# -------------------------
# Load user embeddings (multiple images per user)
# -------------------------
dataset_path = "user_photos"
user_embeddings = {}  # person_name -> list of embeddings

print("Loading user embeddings...")
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    images = os.listdir(person_folder)
    if len(images) == 0:
        print(f"Warning: No images found for {person_name}, skipping...")
        continue

    embeddings_list = []
    for img_file in images:
        img_path = os.path.join(person_folder, img_file)
        try:
            embedding = DeepFace.represent(img_path, model_name="Facenet")[0]["embedding"]
            embeddings_list.append(np.array(embedding))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    if embeddings_list:
        user_embeddings[person_name] = embeddings_list
        print("Loaded:", person_name)

# Optionally, compute average embeddings for more stable recognition
user_avg_embedding = {u: np.mean(embs, axis=0) for u, embs in user_embeddings.items()}

# -------------------------
# CSV FILE SETUP
# -------------------------
csv_file = "break_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Lefted", "Returned", "Time Taken (seconds)"])

# -------------------------
# Chair zone coordinates
# -------------------------
ZONE_X1, ZONE_Y1 = 150, 120
ZONE_X2, ZONE_Y2 = 420, 350

cap = cv2.VideoCapture(0)

# -------------------------
# Tracking dictionaries
# -------------------------
person_in_chair = {}  # True = in chair
leave_times = {}
last_event = {}
face_trackers = {}  # recognized_name -> tracker
face_positions = {}  # recognized_name -> bbox
cooldown = 5  # seconds
frame_skip = 5
frame_count = 0
min_away_duration = 5  # seconds, for return logging

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Helper function: IoU for tracker overlap
# -------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    if (boxAArea + boxBArea - interArea) == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

print("Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    recognized_this_frame = []

    # Update existing trackers
    to_remove = []
    for name, tracker in face_trackers.items():
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            face_positions[name] = (x, y, w, h)
            recognized_this_frame.append(name)
        else:
            to_remove.append(name)

    for name in to_remove:
        face_trackers.pop(name)
        face_positions.pop(name)

    # Detect new faces every N frames
    if frame_count % frame_skip == 0:
        detected_faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            # Skip if overlaps significantly with existing tracker
            if any(iou((x, y, w, h), pos) > 0.3 for pos in face_positions.values()):
                continue

            face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

            recognized_name = "Unknown"
            try:
                unknown_embedding = np.array(
                    DeepFace.represent(face_rgb, model_name="Facenet")[0]["embedding"]
                )
                closest_distance = 0.6
                for user_name, avg_emb in user_avg_embedding.items():
                    distance = cosine(avg_emb, unknown_embedding)
                    if distance < closest_distance:
                        closest_distance = distance
                        recognized_name = user_name
            except Exception as e:
                print("Face recognition error:", e)

            # Initialize tracker for this face
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))

            # For multiple unknown faces, append counter for display only
            display_name = recognized_name
            if recognized_name == "Unknown":
                count = 1
                while display_name in face_trackers:
                    count += 1
                    display_name = f"Unknown_{count}"
            face_trackers[display_name] = tracker
            face_positions[display_name] = (x, y, w, h)
            recognized_this_frame.append(display_name)

    # Chair logic for all recognized faces
    now = datetime.now()
    current_time = time.time()
    for recognized_name in recognized_this_frame:
        base_name = recognized_name.split("_")[0]  # For unknown display handling

        if base_name not in person_in_chair:
            person_in_chair[base_name] = True
            leave_times[base_name] = None
            last_event[base_name] = 0

        x, y, w, h = face_positions[recognized_name]
        cx = x + w // 2
        cy = y + h // 2
        inside_zone = (ZONE_X1 < cx < ZONE_X2) and (ZONE_Y1 < cy < ZONE_Y2)

        if inside_zone:
            if not person_in_chair[base_name] and current_time - last_event[base_name] > cooldown:
                person_in_chair[base_name] = True
                last_event[base_name] = current_time

                if leave_times[base_name]:
                    returned_time = now
                    duration = (returned_time - leave_times[base_name]).seconds
                    if duration >= min_away_duration:
                        with open(csv_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([base_name,
                                             leave_times[base_name].strftime("%H:%M:%S"),
                                             returned_time.strftime("%H:%M:%S"),
                                             duration])
                        if base_name != "Unknown":
                            log_return(base_name, returned_time, duration)
                    leave_times[base_name] = None
        else:
            if person_in_chair[base_name] and current_time - last_event[base_name] > cooldown:
                leave_times[base_name] = now
                person_in_chair[base_name] = False
                last_event[base_name] = current_time
                if base_name != "Unknown":
                    log_leave(base_name, now)

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw chair zone
    cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 2)
    cv2.putText(frame, "Chair Zone", (ZONE_X1, ZONE_Y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Office Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
conn.close()