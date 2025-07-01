# Facial Emotion Recognition with CNN and RESNET

## Overview  
This project implements facial emotion recognition using CNN and ResNet models with real-time webcam integration. The application is built using Streamlit and MediaPipe with support for model switching.

---

## Features
- Real-time facial emotion detection using webcam.
- Model switching between CNN and ResNet.
- MediaPipe face detection.
- OOP-based Streamlit application.
- Class-weighted training for CNN.
- Early stopping and learning rate scheduler.
- Class count generation with JSON storage.
- Training notebooks for CNN and ResNet.

---

## Dataset

Link: https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset

Save this in `data/` directory

---

## Demo
![Demo](demo/Facial%20Emotion%20Recognition.gif)
<!-- <video src="demo/Facial Emotion Recognition.mp4" controls width="700"></video> -->

---

## Project Structure
```text
Facial-Emotion-Recognition/
│
├── data/                       # Image dataset
│
├── notebooks/                  # Model training notebooks
│   ├── train_cnn.ipynb
│   ├── train_resnet.ipynb
│
├── output/                     # Model outputs and class counts
│   ├── class_counts.json
│   ├── cnn_best_model.pth
│   ├── resnet_best_model.pth
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── cnn_model.py
│   ├── resnet_model.py
│   ├── dataloader.py
│   ├── count_labels.py
│
├── app.py                      # Streamlit application with model switching
│
├── webcam_interface_cnn.py     # Standalone webcam interface for CNN
├── webcam_interface_resnet.py  # Standalone webcam interface for ResNet
│
├── README.md
└── requirements.txt
```

---

## Setup Instructions
```bash
git clone https://github.com/arabind-meher/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
pip install -r requirements.txt
streamlit run app.py
```

---

## Models
- CNN: Trained with class weights.
- ResNet: Fine-tuned without class weights.

---

## Training
Training notebooks:
- `train_cnn.ipynb`
- `train_resnet.ipynb`

---

## Notes
- Class counts generated with `count_labels.py`.
- Real-time inference with MediaPipe and OpenCV.
- Separate webcam scripts provided for direct testing.

---

## Author
**Arabind Meher**  
[GitHub](https://github.com/arabind-meher) | [LinkedIn](https://www.linkedin.com/in/arabind-meher/)  
arabindmeher.99@gmail.com
