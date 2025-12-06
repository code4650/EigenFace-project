import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as transforms

# Load Haar Cascade for face detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)


# Load the saved model and class mapping
recognition_model = joblib.load('recognition_model.pkl')
class_mapping = joblib.load('class_mapping.pkl')

# Set image size (should match the one used during training)
image_size = (90, 90)

# Define transformations for grayscale images (same as during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale (Haar Cascade works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        cropped_face = gray[y:y+h, x:x+w]  # Use grayscale image

        # Preprocess the cropped face for the model (same as during training)
        processed_face = transform(cropped_face)
        processed_face = processed_face.numpy().flatten().reshape(1, -1)

        # Predict the identity using the loaded recognition model
        predicted_probabilities = recognition_model.predict_proba(processed_face)
        predicted_label = np.argmax(predicted_probabilities)
        confidence_score = np.max(predicted_probabilities) * 100  # Convert to percentage

        # Get the predicted person's name from the class mapping
        predicted_person = class_mapping.get(predicted_label, "Unknown")

        # Draw the bounding box and predicted label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_person} ({confidence_score:.2f}%)", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame with the detected faces and predictions
    cv2.imshow('Webcam Face Recognition', frame)

    # Exit the webcam feed when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
