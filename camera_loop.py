import cv2
import os
import time
from datetime import datetime
import threading
import psycopg2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import numpy as np

# -------------------------
# GLOBALS
# -------------------------
output_frame = None
lock = threading.Lock()
camera_on = False

attendance = {}
last_seen = {}
person_in_chair = {}
leave_times = {}
last_event = {}
face_trackers = {}
face_positions = {}

# -------------------------
# CONFIG
# -------------------------
ABSENCE_LIMIT = 600        # 10 min for logout
COOLDOWN = 10              # seconds before next leave/return event
FRAME_SKIP = 5             # process every 5th frame
MIN_AWAY_DURATION = 10     # minimum duration to count as leave (seconds)
ZONE_X1, ZONE_Y1 = 150, 120
ZONE_X2, ZONE_Y2 = 420, 350

# -------------------------
# DATABASE CONNECTION
# -------------------------
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="face_monitoring",
        user="postgres",
        password="2005"
    )

def execute_query(query, values=(), fetch=False):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query, values)
        result = cur.fetchall() if fetch else None
        conn.commit()
        cur.close()
        conn.close()
        return result
    except Exception as e:
        print("DB ERROR:", e)
        return None

# -------------------------
# LOGS
# -------------------------
def log_login(name, now):
    print(f"[LOGIN] {name}")
    execute_query(
        "INSERT INTO attendance_logs (name, date, login_time) VALUES (%s, %s, %s)",
        (name, now.date(), now)
    )

def log_logout(name, now):
    print(f"[LOGOUT] {name}")
    execute_query(
        """
        UPDATE attendance_logs
        SET logout_time = %s,
            total_work_seconds = EXTRACT(EPOCH FROM (%s - login_time))
        WHERE name = %s AND date = %s AND logout_time IS NULL
        """,
        (now, now, name, now.date())
    )

def log_leave(name, leave_time):
    print(f"[LEAVE] {name}")
    execute_query(
        "INSERT INTO break_logs (name, leave_time) VALUES (%s, %s)",
        (name, leave_time)
    )

def log_return(name, return_time, duration):
    print(f"[RETURN] {name} ({duration}s)")
    execute_query(
        """
        UPDATE break_logs
        SET return_time = %s,
            duration_seconds = %s
        WHERE ctid = (
            SELECT ctid
            FROM break_logs
            WHERE name = %s AND return_time IS NULL
            ORDER BY leave_time DESC
            LIMIT 1
        )
        """,
        (return_time, duration, name)
    )

# -------------------------
# LOAD USER EMBEDDINGS
# -------------------------
dataset_path = "user_photos"
user_embeddings = {}

print("Loading embeddings...")
for person in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, person)
    if not os.path.isdir(folder):
        continue
    embs = []
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        try:
            emb = DeepFace.represent(path, model_name="Facenet")[0]["embedding"]
            embs.append(np.array(emb))
        except:
            pass
    if embs:
        user_embeddings[person] = embs
        print("Loaded:", person)

user_avg_embedding = {u: np.mean(e, axis=0) for u, e in user_embeddings.items()}

# -------------------------
# FACE DETECTOR
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise Exception("Haarcascade not loaded. Fix OpenCV installation.")

# -------------------------
# UTILITY
# -------------------------
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0]+a[2], b[0]+b[2])
    yB = min(a[1]+a[3], b[1]+b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    return inter / (a[2]*a[3] + b[2]*b[3] - inter)

# -------------------------
# CAMERA LOOP
# -------------------------
def start_camera_loop():
    global camera_on, output_frame
    camera_on = True
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    frame_count = 0

    while camera_on:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recognized_this_frame = []

        # -------------------------
        # FACE DETECTION
        # -------------------------
        if frame_count % FRAME_SKIP == 0:
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                name = "Unknown"

                try:
                    emb = np.array(DeepFace.represent(face_img, model_name="Facenet")[0]["embedding"])
                    best = 0.6
                    for u, avg in user_avg_embedding.items():
                        d = cosine(avg, emb)
                        if d < best:
                            best = d
                            name = u
                except:
                    pass

                recognized_this_frame.append(name)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # -------------------------
        # LOGIC
        # -------------------------
        now = datetime.now()
        current_time = time.time()

        for name in recognized_this_frame:
            if name == "Unknown":
                continue

            # LOGIN
            if name not in attendance:
                attendance[name] = True
                log_login(name, now)

            last_seen[name] = current_time

            if name not in person_in_chair:
                person_in_chair[name] = True
                leave_times[name] = None
                last_event[name] = 0

            # ZONE CHECK
            pos = next(((x,y,w,h) for (x,y,w,h) in faces), None)
            if pos:
                cx, cy = pos[0]+pos[2]//2, pos[1]+pos[3]//2
                inside = ZONE_X1 < cx < ZONE_X2 and ZONE_Y1 < cy < ZONE_Y2
            else:
                inside = False

            # -------------------------
            # STABLE LEAVE/RETURN LOGIC
            # -------------------------
            # Person leaves
            if not inside:
                if person_in_chair[name] and leave_times[name] is None:
                    leave_times[name] = now
                elif leave_times[name]:
                    duration = (now - leave_times[name]).seconds
                    if duration >= MIN_AWAY_DURATION:
                        log_leave(name, leave_times[name])
                        person_in_chair[name] = False
                        last_event[name] = current_time
                        leave_times[name] = None
            # Person returns
            else:
                if not person_in_chair[name] and last_event[name] + COOLDOWN < current_time:
                    log_return(name, now, duration)
                    person_in_chair[name] = True
                    last_event[name] = current_time

        # -------------------------
        # AUTO LOGOUT
        # -------------------------
        for user in list(last_seen.keys()):
            if current_time - last_seen[user] > ABSENCE_LIMIT:
                log_logout(user, datetime.now())
                last_seen.pop(user)
                attendance.pop(user, None)

        cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255,0,0), 2)
        with lock:
            output_frame = frame.copy()

    cap.release()

# -------------------------
# CONTROL
# -------------------------
def stop_camera_loop():
    global camera_on
    camera_on = False

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')