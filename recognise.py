import glob
import os
import cv2
import numpy as np

# Cosine similarity above this means "same person" (OpenCV's recommended value for SFace)
THRESHOLD = 0.363

# 1. Load the face detector and the face recognizer
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2026may.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

# 2. Load everyone who has enrolled (one data/<name>.npy file per person)
people = {}
for path in glob.glob("data/*.npy"):
    name = os.path.splitext(os.path.basename(path))[0]
    people[name] = np.load(path)
print("Known people:", list(people))

# 3. Open the camera
cap = cv2.VideoCapture(0)

# 4. Loop forever
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Find faces
    height, width, _ = frame.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    # Draw on a copy, so boxes don't get into the next face's crop
    display = frame.copy()
    if faces is not None:
        for face in faces:
            # Turn this face into 128 numbers
            aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned)

            # Compare with every saved sample and keep the best match
            best_name = "Unknown"
            best_score = 0.0
            for name, samples in people.items():
                for sample in samples:
                    score = recognizer.match(feature, sample, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_name = name
                        best_score = score

            # Too different from everyone? Then we don't know this person
            if best_score < THRESHOLD:
                best_name = "Unknown"

            # Green box for known people, red for unknown
            colour = (0, 0, 255) if best_name == "Unknown" else (0, 255, 0)
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(display, f"{best_name} {best_score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    cv2.imshow("Recognize", display)
    if cv2.waitKey(1) == ord('q'):
        break

# 5. Release the camera and close the window
cap.release()
cv2.destroyAllWindows()