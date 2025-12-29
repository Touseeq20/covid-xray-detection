import torch
from torchvision import models
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import os
import torch.nn.functional as F

# Using a working Vision Transformer (ViT) model from Hugging Face
# This model is fine-tuned for X-Ray anomalies (Pneumonia/Normal)
MODEL_ID = "nickmuchi/vit-finetuned-chest-xray-pneumonia"

class CovidClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = None
        self.extractor = None
        self.labels = ['Normal', 'Pneumonia'] # Fallback labels

    def load_model(self):
        try:
            print(f"Loading model from {MODEL_ID}...")
            self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            # Load labels from config
            if hasattr(self.model.config, 'id2label'):
                self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            
            print(f"Model loaded. Labels: {self.labels}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, image_input):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        inputs = self.extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
        conf, pred_idx = torch.max(probs, 1)
        pred_label = self.labels[pred_idx.item()]
        
        prob_dict = {
            self.labels[i]: float(probs[0][i]) for i in range(len(self.labels))
        }
        
        # For ViT, we need to handle GradCAM differently or disable it if complex.
        # ViT attention maps are different from CNN GradCAM.
        # We will return the attention weights if possible or just the inputs.
        
        return {
            'class': pred_label,
            'confidence': float(conf),
            'probabilities': prob_dict,
            'tensor': inputs['pixel_values'],
            'model': self.model
        }

