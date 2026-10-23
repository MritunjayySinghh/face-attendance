import cv2

# 1. Load the face detector
detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2026may.onnx", "", (320, 320))

# 2. Open the camera
cap = cv2.VideoCapture(0)

# 3. Loop forever
while True:
    # a. Grab a frame (stop if it failed)
    ret, frame = cap.read()
    if not ret:
        break

    # b. Tell the detector the frame's width and height
    height, width, _ = frame.shape
    detector.setInputSize((width, height))

    # c. Detect faces
    _, faces = detector.detect(frame)

    # d. Draw a green box around each face
    if faces is not None:
        for face in faces:
            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # e. Show the frame
    cv2.imshow("Camera", frame)

    # f. Stop if 'q' was pressed
    if cv2.waitKey(1) == ord('q'):
        break

# 4. Release the camera and close the window
cap.release()
cv2.destroyAllWindows()