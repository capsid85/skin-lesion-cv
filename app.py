import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide", page_icon="🔬")

# Premium Custom CSS
st.markdown("""
<style>
    /* Main background gradient and typography */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Upload Box */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #38bdf8;
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        border-radius: 8px;
    }
    
    /* Columns and Image styling */
    [data-testid="column"] {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    [data-testid="column"]:hover {
        transform: translateY(-5px);
    }
    
    img {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# Define classes (Standard HAM10000 labels)
CLASSES = ['Actinic keratoses (akiec)', 
           'Basal cell carcinoma (bcc)', 
           'Benign keratosis (bkl)', 
           'Dermatofibroma (df)', 
           'Melanoma (mel)', 
           'Melanocytic nevi (nv)', 
           'Vascular lesions (vasc)']

NUM_CLASSES = 7
MODEL_PATH = "model_weights/best_model.pth"

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            st.sidebar.success("Model weights loaded successfully.")
        except Exception as e:
            st.sidebar.warning(f"Could not load weights: {e}")
    else:
        st.sidebar.warning("Using uninitialized dummy model (train the model first).")
        
    model.eval()
    return model

model = load_model()

# Transforms for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("🔬 Skin Lesion Classifier with Grad-CAM")
st.markdown("""
Upload a dermoscopy image to classify the skin lesion into one of 7 categories (HAM10000).
The application uses an **EfficientNet-B0** model and highlights the areas of interest using **Grad-CAM**.
""")

uploaded_file = st.file_uploader("Choose a dermoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display columns
    col1, col2, col3 = st.columns(3)
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = CLASSES[predicted_idx.item()]
        
    with col2:
        st.subheader("Prediction")
        st.markdown(f"### **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence.item():.2%}")
        
        # Display all probabilities
        st.write("Class Probabilities:")
        for i, prob in enumerate(probabilities):
            st.progress(float(prob), text=f"{CLASSES[i]}: {prob:.2%}")

    # Grad-CAM Explainability
    # Target layer for EfficientNet-B0
    target_layers = [model.features[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx.item())]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare image for visualization
    img_array = np.array(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    with col3:
        st.subheader("Grad-CAM Explainability")
        st.image(cam_image, use_container_width=True, caption="Heatmap indicating focus areas")
        st.markdown("*Red areas indicate higher importance for the model's decision.*")
