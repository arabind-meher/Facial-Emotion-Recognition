import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from src.cnn_model import FacialEmotionCNN
from src.resnet_model import FacialEmotionResNet


class ModelHandler:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self.load_model()
        self.preprocess = self.get_preprocess()

    def load_model(self):
        if self.model_type == "CNN":
            model = FacialEmotionCNN(num_classes=5)
            model.load_state_dict(
                torch.load(
                    "output/cnn_best_model.pth", map_location=torch.device("cpu")
                )
            )
        else:
            model = FacialEmotionResNet(num_classes=5)
            model.load_state_dict(
                torch.load(
                    "output/resnet_best_model.pth", map_location=torch.device("cpu")
                )
            )
        model.eval()
        return model

    def get_preprocess(self):
        if self.model_type == "CNN":
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def predict(self, face, emotion_labels):
        face_tensor = self.preprocess(face).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
            confidence, predicted = torch.max(probabilities, 0)
            return emotion_labels[predicted.item()], confidence.item() * 100


class WebcamApp:
    def __init__(self, model_handler):
        self.model_handler = model_handler
        self.emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        self.frame_window = st.image([])
        self.running = True

    def run(self):
        st.sidebar.write("Press 'Stop' to release webcam.")
        stop_button = st.sidebar.button("Stop Webcam")

        while self.cap.isOpened() and self.running and not stop_button:
            ret, frame = self.cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    w_box = int(bbox.width * w)
                    h_box = int(bbox.height * h)

                    padding = 30
                    cx, cy = x + w_box // 2, y + h_box // 2
                    size = max(w_box, h_box) + 2 * padding

                    x1 = max(cx - size // 2, 0)
                    y1 = max(cy - size // 2, 0)
                    x2 = min(cx + size // 2, frame.shape[1])
                    y2 = min(cy + size // 2, frame.shape[0])

                    face = frame[int(y1) : int(y2), int(x1) : int(x2)]

                    try:
                        emotion, confidence = self.model_handler.predict(
                            face, self.emotion_labels
                        )
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{emotion} ({confidence:.1f}%)",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (36, 255, 12),
                            2,
                        )
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

            self.frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        self.cap.release()
        self.face_detection.close()
        st.write("Webcam stopped.")


# -------------------------------
# Streamlit Main App
# -------------------------------
st.title("Facial Emotion Recognition App (OOP Version)")
st.sidebar.title("Model Selection")

model_option = st.sidebar.selectbox("Select Model", ["CNN", "ResNet"])

model_handler = ModelHandler(model_option)
webcam_app = WebcamApp(model_handler)

if st.sidebar.button("Start Webcam"):
    webcam_app.run()
