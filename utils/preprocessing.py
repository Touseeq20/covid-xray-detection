import torch
from torchvision import transforms
from PIL import Image

def get_transforms():
    """
    Returns the standard transform pipeline for X-Ray images.
    Resize to 224x224 (standard for ResNet/ViT), Generic normalization.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet normalization is usually a safe default for pre-trained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image):
    """
    Args:
        image: PIL Image or path to image
    Returns:
        Tensor: Preprocessed image tensor (1, C, H, W)
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    
    transform = get_transforms()
    input_tensor = transform(image)
    return input_tensor.unsqueeze(0) # Add batch dimension
