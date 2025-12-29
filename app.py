import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
import os
import torch
from utils.inference import CovidClassifier
# Grad-CAM is disabled for ViT in this version to ensure stability
# from utils.visualization import generate_gradcam 

# Page Config
st.set_page_config(
    page_title="COVID-19 Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/480/coronavirus.png", width=100)
    st.title("Settings")
    
    st.info("Now powered by Google's Vision Transformer (ViT) architecture.")
    
    upload_file = st.file_uploader("Upload Chest X-Ray or CT", type=['jpg', 'png', 'jpeg'])
    
    use_sample = st.checkbox("Use Sample Image")
    sample_choice = st.radio("Select Sample:", ["COVID-19", "Normal"]) if use_sample else None

    # st.caption("Visualizations disabled for ViT backbone")

# Main UI
st.title("🩺 COVID-19 Detection System")
st.markdown("""
> **Disclaimer:** This tool is for **research purposes only** and not for clinical diagnosis. 
> Always consult a medical professional.
""")

# Load Model
@st.cache_resource
def get_classifier():
    classifier = CovidClassifier()
    success = classifier.load_model()
    if not success:
        st.error("Failed to load the model. Please check the logs in the terminal.")
        return None
    return classifier

classifier = get_classifier()

# Handle Input
image = None
if use_sample and sample_choice:
    sample_path = os.path.join("sample_images", "covid.jpg" if sample_choice == "COVID-19" else "normal.jpg")
    if os.path.exists(sample_path):
        image = Image.open(sample_path).convert("RGB")
    else:
        st.warning(f"Sample images not found in {sample_path}")
elif upload_file:
    image = Image.open(upload_file).convert("RGB")

if image:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input X-Ray")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        
        if classifier:
            with st.spinner("Analyzing with Vision Transformer..."):
                time.sleep(1) # UX transition
                
                try:
                    results = classifier.predict(image)
                    
                    # Display Prediction
                    pred_class = results['class']
                    confidence = results['confidence'] * 100
                    
                    # Logic to interpret "Pneumonia" as potential COVID risk
                    display_label = pred_class
                    if pred_class.lower() == "pneumonia":
                        display_label = "PNEUMONIA / ABNORMAL (Potential COVID-19)"
                        st.error(f"## {display_label}")
                    elif pred_class.lower() == "normal":
                        st.success(f"## {display_label}")
                    else:
                        st.warning(f"## {display_label}")
                        
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                    
                    st.write("### Class Probabilities:")
                    for label, prob in results['probabilities'].items():
                        st.progress(prob, text=f"{label}: {prob*100:.1f}%")
                    
                    st.info("Model: **ViT (Vision Transformer)** fine-tuned on Chest X-Rays.")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

else:
    st.info("Please upload an X-Ray image or select a sample to begin.")

st.divider()
st.markdown("### 🔬 How it Works")
st.markdown("""
1.  **Preprocessing**: Image is scaled to 224x224 patches.
2.  **Inference**: A state-of-the-art **Vision Transformer (ViT)** model analyzes the image using self-attention mechanisms.
3.  **Classification**: The model predicts if the lungs show signs of Normalcy or Pneumonia (which is the primary indicator for COVID-19 in X-rays).
""")
