import cv2
from CustomDataGen import CustomDataGen

# Create a classifier for face detection
face_cascade_path = "haarcascade\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Open a connection to the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture an image from the camera
    ret, frame = video_capture.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Human Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check if the user pressed the 'q' key for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources and close all windows
video_capture.release()
cv2.destroyAllWindows()
