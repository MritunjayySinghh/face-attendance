import cv2

# 1. Open the camera
cap = cv2.VideoCapture(0)

# 2. Loop forever
while True:
    # a. Grab a frame
    ret, frame = cap.read()

    # b. If grabbing failed, stop the loop
    if not ret:
        break

    # c. Show the frame
    cv2.imshow("Camera", frame)

    # d. If 'q' was pressed, stop the loop
    if cv2.waitKey(1) == ord('q'):
        break

# 3. Release the camera and close the window
cap.release()
cv2.destroyAllWindows()