import os
import cv2
import numpy as np

SAMPLES_NEEDED = 5

# 1. Ask whose face we're saving
name = input("Name: ").strip()

# 2. Load the face detector and the face recognizer
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2026may.onnx", "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface_2021dec.onnx", "")

# 3. Open the camera
cap = cv2.VideoCapture(0)
samples = []

# 4. Keep going until we have enough samples
while len(samples) < SAMPLES_NEEDED:
    ret, frame = cap.read()
    if not ret:
        break

    # Find faces
    height, width, _ = frame.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    # Draw on a copy, so the boxes don't end up in the saved face
    display = frame.copy()
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(display, f"Press S to save ({len(samples)}/{SAMPLES_NEEDED})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Enroll", display)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        if faces is None or len(faces) != 1:
            print("Make sure exactly one face is in view")
        else:
            # Turn the face into 128 numbers
            aligned = recognizer.alignCrop(frame, faces[0])
            samples.append(recognizer.feature(aligned))
            print(f"Saved sample {len(samples)}/{SAMPLES_NEEDED}")

# 5. Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# 6. Save the numbers (not the photo!) in the data folder
if len(samples) == SAMPLES_NEEDED:
    os.makedirs("data", exist_ok=True)
    np.save(f"data/{name}.npy", np.vstack(samples))
    print(f"Done! Saved to data/{name}.npy")
else:
    print("Stopped early, nothing saved")