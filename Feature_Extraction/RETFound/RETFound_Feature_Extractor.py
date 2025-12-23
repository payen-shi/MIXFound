"""
RETFound Feature Extraction Script

This script extracts visual features from images using the RETFound Vision Transformer model.
RETFound is a pre-trained Vision Transformer model for retinal fundus image analysis.

Author: [Your Name]
Date: [Date]
"""

import os
import torch
import numpy as np
import pickle
from PIL import Image
from typing import List, Dict, Optional
import models_vit

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def prepare_model(chkpt_dir: str, arch: str = 'vit_large_patch16', num_classes: int = 5) -> torch.nn.Module:
    """
    Load and prepare the RETFound Vision Transformer model from checkpoint.
    
    Args:
        chkpt_dir (str): Path to the model checkpoint file (.pth)
        arch (str): Model architecture name, default: 'vit_large_patch16'
        num_classes (int): Number of classes for the model, default: 5
    
    Returns:
        torch.nn.Module: Loaded RETFound model
    
    Note:
        The model is loaded in CPU mode. You need to move it to GPU manually
        using model.cuda() if GPU is available.
    """
    # Build model with specified architecture
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=num_classes,
        drop_path_rate=0,
        global_pool=True,
    )
    
    # Load checkpoint
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    
    return model


def load_image_paths_from_directory(folder_path: str, extension: str = '.jpg') -> List[str]:
    """
    Recursively load all image paths from a directory.
    
    Args:
        folder_path (str): Root directory to search for images
        extension (str): File extension to filter (default: '.jpg')
    
    Returns:
        List[str]: List of image file paths found in the directory
    """
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                image_paths.append(os.path.join(root, file))
    return image_paths


def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    Preprocess an image for RETFound model input.
    
    The preprocessing pipeline includes:
    1. Load image from file
    2. Resize to target size (224x224)
    3. Normalize pixel values to [0, 1]
    4. Apply ImageNet normalization (subtract mean, divide by std)
    5. Convert to tensor and rearrange dimensions (NHWC -> NCHW)
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image size (width, height), default: (224, 224)
    
    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, H, W)
            ready for model input
    
    Raises:
        AssertionError: If the resized image doesn't have the expected shape (224, 224, 3)
    """
    # Load image
    image = Image.open(image_path)
    
    # Resize to target size
    img = image.resize(target_size)
    
    # Convert to numpy array and normalize to [0, 1]
    img = np.array(img) / 255.0
    
    # Verify image shape
    assert img.shape == (target_size[1], target_size[0], 3), \
        f"Expected shape ({target_size[1]}, {target_size[0]}, 3), got {img.shape}"
    
    # Apply ImageNet normalization
    img = img - IMAGENET_MEAN
    img = img / IMAGENET_STD
    
    # Convert to tensor and add batch dimension
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)  # Add batch dimension: (H, W, C) -> (1, H, W, C)
    
    # Rearrange dimensions: NHWC -> NCHW (batch, channels, height, width)
    x = torch.einsum('nhwc->nchw', x)
    
    return x


def extract_features(model: torch.nn.Module, image_path: str, device: str = 'cuda') -> Dict:
    """
    Extract features from an image using the RETFound model.
    
    Args:
        model (torch.nn.Module): Pre-loaded RETFound model
        image_path (str): Path to the image file
        device (str): Device to run inference on ('cuda' or 'cpu'), default: 'cuda'
    
    Returns:
        Dict: Dictionary containing extracted features and image path
            - 'feats': numpy array of extracted features
            - 'img_path': original image path
    
    Note:
        The model should be in eval mode (model.eval()) before calling this function.
    """
    # Preprocess image
    x = preprocess_image(image_path)
    
    # Move tensor to specified device
    x = x.to(device)
    
    # Forward pass through model to extract features
    with torch.no_grad():  # Disable gradient computation for inference
        latent = model.forward_features(x.float())
        feature = torch.squeeze(latent)  # Remove batch dimension
    
    # Store features and metadata
    feats_dict = {
        'feats': feature.detach().cpu().numpy(),
        'img_path': image_path
    }
    
    return feats_dict


def extract_features_batch(
    model: torch.nn.Module,
    image_paths: List[str],
    device: str = 'cuda',
    verbose: bool = True
) -> List[Dict]:
    """
    Extract features from a batch of images.
    
    Args:
        model (torch.nn.Module): Pre-loaded RETFound model
        image_paths (List[str]): List of image file paths to process
        device (str): Device to run inference on ('cuda' or 'cpu'), default: 'cuda'
        verbose (bool): Whether to print progress information, default: True
    
    Returns:
        List[Dict]: List of feature dictionaries, one per image
    """
    all_features = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            if verbose and (idx + 1) % 10 == 0:
                print(f"Processing image {idx + 1}/{len(image_paths)}...")
            
            feats_dict = extract_features(model, img_path, device=device)
            all_features.append(feats_dict)
            
            if verbose and idx == 0:
                print(f"Feature shape: {feats_dict['feats'].shape}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return all_features


def save_features(features: List[Dict], output_path: str) -> None:
    """
    Save extracted features to a pickle file.
    
    Args:
        features (List[Dict]): List of feature dictionaries to save
        output_path (str): Path to the output pickle file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save features to pickle file
    with open(output_path, 'wb') as file:
        pickle.dump(features, file)
    
    print(f"Successfully saved {len(features)} feature vectors to {output_path}")


def main():
    """
    Main function to orchestrate the feature extraction process.
    
    Configure the following parameters according to your needs:
    - folder_path: Directory containing images to process
    - checkpoint_path: Path to RETFound model checkpoint
    - output_path: Path where features will be saved
    - device: Device to use for inference ('cuda' or 'cpu')
    """
    # ========== Configuration ==========
    # Directory containing images to process
    folder_path = 'dataset/fundus/E'
    
    # Path to RETFound model checkpoint
    checkpoint_path = 'RETFound_cfp_weights.pth'
    
    # Output path for saved features
    output_path = 'Final_feature/RETFound_feature_E.pickle'
    
    # Model architecture
    model_arch = 'vit_large_patch16'
    
    # Number of classes (adjust based on your model)
    num_classes = 5
    
    # Device to use for inference ('cuda' or 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Image file extension to search for
    image_extension = '.jpg'
    
    # ========== Feature Extraction Pipeline ==========
    # Load model
    print(f"Loading RETFound model from {checkpoint_path}...")
    model = prepare_model(checkpoint_path, arch=model_arch, num_classes=num_classes)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}.")
    
    # Load image paths
    print(f"Loading image paths from {folder_path}...")
    image_paths = load_image_paths_from_directory(folder_path, extension=image_extension)
    print(f"Found {len(image_paths)} images to process.")
    
    if len(image_paths) == 0:
        print("No images found. Please check the folder_path configuration.")
        return
    
    # Extract features
    print("Extracting features...")
    all_features = extract_features_batch(
        model=model,
        image_paths=image_paths,
        device=device,
        verbose=True
    )
    
    # Save features
    print(f"Total features extracted: {len(all_features)}")
    save_features(all_features, output_path)
    
    print("Feature extraction completed!")


if __name__ == "__main__":
    main()

