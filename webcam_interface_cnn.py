import cv2
import torch
import numpy as np
from torchvision import transforms
from src import FacialEmotionCNN  # Adjust this if your model import path is different

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load model
model = FacialEmotionCNN()
model.load_state_dict(
    torch.load("output/cnn_best_model.pth", map_location=torch.device("cpu"))
)
model.eval()

# Correct emotion labels
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Preprocessing pipeline with correct normalization
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        try:
            # Add padding and ensure square crop
            padding = 20  # You can adjust this
            cx, cy = x + w // 2, y + h // 2
            size = max(w, h) + 2 * padding

            x1 = max(cx - size // 2, 0)
            y1 = max(cy - size // 2, 0)
            x2 = min(cx + size // 2, frame.shape[1])
            y2 = min(cy + size // 2, frame.shape[0])

            face = frame[int(y1) : int(y2), int(x1) : int(x2)]

            # Preprocess the face
            face_tensor = preprocess(face).unsqueeze(0)  # Shape: [1, 3, 48, 48]

            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze()

                # Print all probabilities for inspection
                for idx, prob in enumerate(probabilities):
                    print(f"{emotion_labels[idx]}: {prob.item() * 100:.1f}%")

                confidence, predicted = torch.max(probabilities, 0)
                emotion = emotion_labels[predicted.item()]

            # Display prediction and confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emotion} ({confidence.item() * 100:.1f}%)",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

        except Exception as e:
            print(f"Error processing face: {e}")
            continue

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
