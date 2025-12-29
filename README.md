# 🩺 COVID-19 Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/GUI-Streamlit-red)
![AI](https://img.shields.io/badge/AI-Vision_Transformer-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

A professional, lightweight, and ready-to-use AI system for detecting **COVID-19** and **Pneumonia** anomalies from Chest X-Ray images. This project leverages state-of-the-art **Vision Transformer (ViT)** architecture to provide accurate and rapid diagnostic insights, optimized for CPU inference.

## 📌 Project Overview

The rapid diagnosis of COVID-19 and related pulmonary conditions is critical for effective patient management. This project presents an automated diagnostic tool that uses a fine-tuned **Google Vision Transformer (ViT)** model to classify chest radiographs. Unlike traditional CNNs, ViT leverages self-attention mechanisms to capture global dependencies in the image, offering robust performance even on standard hardware.

**Key Capabilities:**
* **Disease Detection**: Classifies X-rays into **Normal** or **Pneumonia** (a primary indicator for COVID-19).
* **High Efficiency**: Optimized for non-GPU environments, making it suitable for portable deployment.
* **User-Friendly Interface**: Features a clean, interactive web dashboard built with Streamlit.

## 🚀 Features

* **⚡ Lightweight Inference**: Runs efficiently on standard CPUs (Laptop/Desktop).
* **🧠 Advanced AI**: Powered by `nickmuchi/vit-finetuned-chest-xray-pneumonia` (Hugging Face).
* **📂 Automatic Setup**: Auto-downloads models and configurations on first run.
* **📊 Real-time Analysis**: Provides instant classification with confidence probability scores.
* **🖼️ Sample Data**: Includes synthetic sample generator for immediate testing.

## 📂 Project Structure

```
covid_xray_detection/
├── app.py                 # Main Streamlit application entry point
├── requirements.txt       # Python dependencies
├── run_app.bat           # One-click Windows launcher
├── utils/
│   ├── inference.py      # Model loading and prediction logic (ViT)
│   └── preprocessing.py  # Image transformation pipeline
└── sample_images/        # Sample X-rays for testing
```

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8 or higher installed.

### Quick Start (Windows)
1. **Clone the repository**:
```bash
git clone https://github.com/Touseeq20/covid-xray-detection.git
cd covid-xray-detection
```

2. **Run the Launcher**: Double-click **`run_app.bat`**.
*This script will automatically install dependencies and launch the app.*

### Manual Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## 🧠 Model Details

* **Architecture**: Vision Transformer (ViT-base-patch16-224)
* **Source Model**: [nickmuchi/vit-finetuned-chest-xray-pneumonia](https://huggingface.co/nickmuchi/vit-finetuned-chest-xray-pneumonia)
* **Input Resolution**: 224x224 pixels
* **Framework**: PyTorch & Hugging Face Transformers

## 🔬 Research Abstract

> "This project implements a diagnostic tool utilizing a fine-tuned Vision Transformer (ViT) architecture for classifying pulmonary abnormalities in chest radiographs. By leveraging self-attention mechanisms, the model effectively distinguishes between normal and pneumonic lung patterns. This system serves as a rapid, accessible screening aid for COVID-19 related complications, particularly valuable in resource-constrained environments where access to radiologists or advanced GPU hardware is limited."

## 👨‍💻 Author

**Touseeq**
* 📧 **Email**: [mtouseeq20@gmail.com](mailto:mtouseeq20@gmail.com)
* 🐙 **GitHub**: [github.com/Touseeq20](https://github.com/Touseeq20)

---

*Created for Research & Educational Use. Not for clinical diagnosis.*
