import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
# Function to extract faces and labels from images in a given directory
def extract_faces_and_labels(directory):
    faces = []
    labels = []
    label_encoder = LabelEncoder()
    label_encoder.fit([directory])

    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces_rect:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(directory)

    return faces, label_encoder.transform(labels)

# Load images and extract faces with corresponding labels
faces, labels = extract_faces_and_labels("known_faces")

# Convert lists to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Flatten the 2D images into 1D vectors
faces_flattened = faces.reshape(len(faces), -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces_flattened, labels, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each face detected, predict the label using the SVM classifier
    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]
        face_flattened = face_roi.reshape(1, -1)
        label = svm_classifier.predict(face_flattened)[0]

        # Draw a rectangle around the face and display the predicted label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_encoder.inverse_transform([label])[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
