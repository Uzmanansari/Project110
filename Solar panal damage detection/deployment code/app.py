import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Function to load the model
@st.cache_resource
def load_model(weights_path):
    model = YOLO(weights_path)  # Load the model with ultralytics
    return model

# Function to perform object detection
def detect_objects(image, model):
    results = model(image)  # Perform inference
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, results, model):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls = int(box.cls.item())
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

# Streamlit UI
st.title('Object Detection with YOLOv8')
st.write('Upload an image to perform object detection using a trained YOLOv8 model.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Load model
    model = load_model('/content/drive/MyDrive/Colab Notebooks/solar panal object/best.pt')

    # Perform object detection
    results = detect_objects(image_np, model)

    # Draw bounding boxes
    image_with_boxes = draw_boxes(image_np.copy(), results, model)

    # Convert back to PIL image
    result_image = Image.fromarray(image_with_boxes)

    # Display the output
    st.image(result_image, caption='Detected Objects', use_column_width=True)
    st.write('Done!')
