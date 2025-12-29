import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam(model, target_layer, input_tensor, original_image, target_category_int=None):
    """
    Generates a Grad-CAM heatmap overlay.
    
    Args:
        model: PyTorch model
        target_layer: The target layer to visualize (usually the last conv layer)
        input_tensor: Preprocessed tensor (1, C, H, W)
        original_image: Original PIL image (for overlay)
        target_category_int: The class index to visualize (None = highest predicted)
        
    Returns:
        visualization: Image with heatmap overlay
    """
    # Create GradCAM object
    # We need to ensure the model is in eval mode but gradients are enabled for CAM
    # Note: grad-cam library handles this usually
    
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Default to highest scoring category if None
    targets = [ClassifierOutputTarget(target_category_int)] if target_category_int is not None else None

    # Compute CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Prepare original image for overlay (resize to 224x224 and normalize 0-1)
    img_resized = original_image.resize((224, 224))
    rgb_img = np.float32(img_resized) / 255.0
    
    # Create visualization
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization
