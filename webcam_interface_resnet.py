import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from src.resnet_model import FacialEmotionResNet

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

# Load ResNet Model
model = FacialEmotionResNet(num_classes=5)
model.load_state_dict(
    torch.load("output/resnet_best_model.pth", map_location=torch.device("cpu"))
)
model.eval()

# Emotion Labels
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# ResNet Preprocessing
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            # Add padding and ensure square crop
            padding = 30
            cx, cy = x + w_box // 2, y + h_box // 2
            size = max(w_box, h_box) + 2 * padding

            x1 = max(cx - size // 2, 0)
            y1 = max(cy - size // 2, 0)
            x2 = min(cx + size // 2, frame.shape[1])
            y2 = min(cy + size // 2, frame.shape[0])

            face = frame[int(y1) : int(y2), int(x1) : int(x2)]

            try:
                face_tensor = preprocess(face).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1).squeeze()

                    for idx, prob in enumerate(probabilities):
                        print(f"{emotion_labels[idx]}: {prob.item() * 100:.1f}%")

                    confidence, predicted = torch.max(probabilities, 0)
                    emotion = emotion_labels[predicted.item()]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({confidence.item() * 100:.1f}%)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

                mp_drawing.draw_detection(frame, detection)

            except Exception as e:
                print(f"Error processing face: {e}")
                continue

    cv2.imshow("Facial Emotion Recognition (MediaPipe + ResNet)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()
