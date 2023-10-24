import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Create a classifier for face detection
face_cascade_path = "haarcascade\\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)


# Load the model for age estimation
age_model = tf.keras.models.load_model("checkpoints\\model_checkpoint.h5")


# Open a connection to the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture an image from the camera
    ret, frame = video_capture.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around faces and estimate age
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face region
        face_img = frame[y:y+h, x:x+w]
        
        # Resize and normalize the face image to match the model's input size and scale
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 255.0
        
        # Predict the age
        age_prediction = age_model.predict(np.expand_dims(face_img, axis=0))
        
        # Display the estimated age
        predicted_age = int(age_prediction[0][0])
        cv2.putText(frame, f"Age: {predicted_age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Video', frame)

    # Check for the 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
video_capture.release()
cv2.destroyAllWindows()
