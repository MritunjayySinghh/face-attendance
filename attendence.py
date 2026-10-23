import glob
import os
import time
from datetime import datetime

import cv2
import numpy as np

import database
import voice

# ---------- Settings ----------
# Small numbers so you can test at your desk. Real lecture values are in the comments.
LECTURE_MINUTES = 3        # real: 60
FREE_TO_GO_MINUTES = 1     # real: 20 -> in the last 20 minutes, leaving is fine
CHECK_EVERY_SECONDS = 5    # real: 30 -> how often we note who is in the room
LEFT_AFTER_SECONDS = 20    # real: 420 (7 minutes) -> not seen for this long = "left"
PRESENT_PERCENT = 80       # seen in at least 80% of checks = Present
THRESHOLD = 0.5            # stricter than recognize.py (0.363), so fewer wrong matches
CONFIRM_FRAMES = 5         # must be recognised in 5 frames in a row before we count them
EARLIEST_END_MINUTES = 1   # real: 30 -> ignore "see you next week" before this (no early false alarms)

# ---------- Face recognition (same idea as recognize.py) ----------
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2026may.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

people = {}
for path in glob.glob("data/*.npy"):
    name = os.path.splitext(os.path.basename(path))[0]
    people[name] = np.load(path)
print("Known people:", list(people))

# ---------- Which lecture is this, and who is supposed to be here? ----------
module = input("Module code (e.g. COMP1312): ").strip().upper()
expected = database.class_list(module)
if not expected:
    print(f"No class list for {module} in data/class_lists.csv, so everyone enrolled is expected")
    expected = list(people)
for name in expected:
    if name not in people:
        print(f"Warning: {name} is on the class list but hasn't enrolled their face yet")
print("Supposed to be here:", expected)


def recognise(frame):
    """Find every face in the frame. Returns a list of (name, box)."""
    height, width, _ = frame.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    results = []
    if faces is None:
        return results
    for face in faces:
        feature = recognizer.feature(recognizer.alignCrop(frame, face))
        best_name, best_score = "Unknown", 0.0
        for name, samples in people.items():
            for sample in samples:
                score = recognizer.match(feature, sample, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_name, best_score = name, score
        if best_score < THRESHOLD:
            best_name = "Unknown"
        results.append((best_name, face[:4].astype(int)))
    return results


def clock(t):
    """Turn a time.time() value into '09:41:05'."""
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def mmss(seconds):
    """Turn 125 seconds into '02:05'."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def write(frame, text, row):
    """Write a line of text in the top-left corner, with a black outline so it's readable."""
    position = (10, 30 + row * 28)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ---------- Lecture state ----------
start = time.time()
end = start + LECTURE_MINUTES * 60
free_to_go = end - FREE_TO_GO_MINUTES * 60    # after this, leaving doesn't count

first_seen = {}                               # name -> when they arrived
last_seen = {}                                # name -> when we last saw them
seen_checks = {name: 0 for name in people}    # name -> how many checks they were seen in
streak = {name: 0 for name in people}         # name -> frames in a row they've been recognised
total_checks = 0
seen_since_last_check = set()
next_check = start + CHECK_EVERY_SECONDS

voice.start()                                 # start listening to the microphone
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    now = time.time()

    # 1. Who is in the room right now?
    seen_now = set()
    for name, (x, y, w, h) in recognise(frame):
        colour = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        if name != "Unknown":
            seen_now.add(name)

    # Only trust a name after several frames in a row (ignores one-frame mistakes)
    for name in people:
        streak[name] = streak[name] + 1 if name in seen_now else 0
        if streak[name] >= CONFIRM_FRAMES:
            first_seen.setdefault(name, now)
            last_seen[name] = now
            seen_since_last_check.add(name)

    # 2. Every few seconds, note who was there (only until it's "free to go")
    if now >= next_check:
        if now <= free_to_go:
            total_checks += 1
            for name in seen_since_last_check:
                seen_checks[name] += 1
        seen_since_last_check = set()
        next_check = now + CHECK_EVERY_SECONDS

    # 3. Show the timer and everyone's live status
    write(frame, f"Lecture {mmss(now - start)} / {mmss(end - start)}", 0)
    if now < free_to_go:
        write(frame, f"Free to go in {mmss(free_to_go - now)}", 1)
    else:
        write(frame, "FREE TO GO", 1)
    for row, name in enumerate(people, start=2):
        if name not in first_seen:
            status = "not arrived"
        elif now - last_seen[name] > LEFT_AFTER_SECONDS:
            status = f"left at {clock(last_seen[name])}"
        else:
            status = "in the room"
        percent = round(100 * seen_checks[name] / total_checks) if total_checks else 0
        write(frame, f"{name}: {status} ({percent}%)", row)
    write(frame, "Mic: " + " ".join(voice.last_heard.split()[-6:]), len(people) + 2)

    cv2.imshow("Attendance", frame)

    # 4. Did the lecturer say "see you next week"? (Pressing E does the same)
    key = cv2.waitKey(1)
    if voice.lecture_over.is_set():
        voice.lecture_over.clear()
        if now - start >= EARLIEST_END_MINUTES * 60:
            free_to_go = min(free_to_go, now)
            print(f'Heard "{voice.last_heard}" -> FREE TO GO')
        else:
            print(f'Heard "{voice.last_heard}" but it is too early, ignoring it')
    if key in (ord('e'), ord('E')):
        free_to_go = min(free_to_go, now)

    # 5. Q = stop now
    if key in (ord('q'), ord('Q')) or now >= end:
        break

cap.release()
cv2.destroyAllWindows()

# ---------- Final attendance ----------
free_to_go = min(free_to_go, time.time())   # if we stopped early, the lecture ended now
everyone = expected + [name for name in first_seen if name not in expected]
rows = []
for name in everyone:
    percent = round(100 * seen_checks.get(name, 0) / total_checks) if total_checks else 0
    if name not in first_seen or percent == 0:
        status = "Absent"
    elif percent >= PRESENT_PERCENT:
        status = "Present"
    elif last_seen[name] < free_to_go - LEFT_AFTER_SECONDS:
        status = "Left early"
    else:
        status = "Partial"
    on_list = 1 if name in expected else 0
    arrived = clock(first_seen[name]) if name in first_seen else None
    left = clock(last_seen[name]) if name in last_seen else None
    rows.append((name, on_list, status, percent, arrived, left))
    print(f"{name}: {status} ({percent}%), arrived {arrived or '-'}, last seen {left or '-'}"
          + ("" if on_list else "  <- not on the class list"))

started = datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M")
ended = datetime.now().strftime("%Y-%m-%d %H:%M")
lecture_id = database.save_lecture(module, started, ended, rows)
print(f"Saved lecture #{lecture_id} ({module}) to {database.DB_PATH}")