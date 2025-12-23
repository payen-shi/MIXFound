"""
FLAIR Feature Extraction Script

This script extracts visual features from images using the FLAIR model.
It supports multiple data augmentation strategies and can process images
from a text file containing image paths.

Author: [Your Name]
Date: [Date]
"""

import os
from PIL import Image
import numpy as np
import pickle
from flair import FLAIRModel
from torchvision import transforms
from typing import List, Dict, Optional


def load_image_paths_from_txt(txt_path: str, delimiter: str = ';') -> List[str]:
    """
    Load image paths from a text file.
    
    The text file should contain one image path per line. If each line contains
    multiple fields separated by a delimiter, only the first field (image path)
    will be extracted.
    
    Args:
        txt_path (str): Path to the text file containing image paths
        delimiter (str): Delimiter used to separate fields in each line (default: ';')
    
    Returns:
        List[str]: List of image file paths
    
    Example:
        If the text file contains:
            /path/to/image1.jpg;label1
            /path/to/image2.jpg;label2
        
        This function returns:
            ['/path/to/image1.jpg', '/path/to/image2.jpg']
    """
    image_paths = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        # Extract image path (first field before delimiter) and remove whitespace
        image_paths = [line.strip().split(delimiter)[0] for line in lines]
    return image_paths


def load_image_paths_from_directory(folder_path: str, extension: str = '.jpg') -> List[str]:
    """
    Load all image paths from a directory recursively.
    
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


def extract_features_no_augmentation(
    model: FLAIRModel,
    image_path: str,
    text_prompts: List[str]
) -> Dict:
    """
    Extract features from an image without any data augmentation.
    
    Args:
        model (FLAIRModel): Initialized FLAIR model
        image_path (str): Path to the image file
        text_prompts (List[str]): List of text prompts for feature extraction
    
    Returns:
        Dict: Dictionary containing extracted features and image path
            - 'feats': numpy array of extracted features
            - 'img_path': original image path
    """
    # Load and convert image to numpy array
    image = np.array(Image.open(image_path))
    
    # Forward pass through FLAIR model
    feature, _, _ = model(image, text_prompts)
    
    # Store features and metadata
    feats_dict = {
        'feats': feature.cpu().numpy(),
        'img_path': image_path
    }
    return feats_dict


def extract_features_with_random_crop(
    model: FLAIRModel,
    image_path: str,
    text_prompts: List[str],
    crop_size: tuple = (224, 224)
) -> Dict:
    """
    Extract features from an image with random crop augmentation.
    
    Args:
        model (FLAIRModel): Initialized FLAIR model
        image_path (str): Path to the image file
        text_prompts (List[str]): List of text prompts for feature extraction
        crop_size (tuple): Size of the random crop (width, height), default: (224, 224)
    
    Returns:
        Dict: Dictionary containing extracted features and image path
            - 'feats': numpy array of extracted features
            - 'img_path': original image path
    """
    # Load image
    image = np.array(Image.open(image_path))
    pil_image = Image.fromarray(image)
    
    # Apply random crop augmentation
    random_crop = transforms.RandomCrop(size=crop_size)
    cropped_image = random_crop(pil_image)
    
    # Convert back to numpy array
    image = np.array(cropped_image)
    
    # Forward pass through FLAIR model
    feature, _, _ = model(image, text_prompts)
    
    # Store features and metadata
    feats_dict = {
        'feats': feature.cpu().numpy(),
        'img_path': image_path
    }
    return feats_dict


def extract_features_with_random_augmentation(
    model: FLAIRModel,
    image_path: str,
    text_prompts: List[str],
    apply_augmentation: bool = False
) -> Dict:
    """
    Extract features from an image with optional random augmentation.
    
    The augmentation includes:
    - Random horizontal flip (50% probability)
    - Random vertical flip (50% probability)
    
    Args:
        model (FLAIRModel): Initialized FLAIR model
        image_path (str): Path to the image file
        text_prompts (List[str]): List of text prompts for feature extraction
        apply_augmentation (bool): Whether to apply random augmentation (default: False)
    
    Returns:
        Dict: Dictionary containing extracted features and image path
            - 'feats': numpy array of extracted features
            - 'img_path': original image path
    """
    # Load image
    image = Image.open(image_path)
    
    # Define random augmentation transforms
    random_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
        transforms.RandomVerticalFlip(p=0.5),     # Random vertical flip with 50% probability
    ])
    
    # Apply augmentation if requested
    if apply_augmentation:
        image = random_augmentation(image)
    
    # Convert to numpy array
    image = np.array(image)
    
    # Forward pass through FLAIR model
    feature, _, _ = model(image, text_prompts)
    
    # Store features and metadata
    feats_dict = {
        'feats': feature.cpu().numpy(),
        'img_path': image_path
    }
    return feats_dict


def extract_features_batch(
    model: FLAIRModel,
    image_paths: List[str],
    text_prompts: List[str],
    augmentation_type: str = 'none',
    num_iterations: int = 1
) -> List[Dict]:
    """
    Extract features from a batch of images with optional data augmentation.
    
    Args:
        model (FLAIRModel): Initialized FLAIR model
        image_paths (List[str]): List of image file paths to process
        text_prompts (List[str]): List of text prompts for feature extraction
        augmentation_type (str): Type of augmentation to apply
            - 'none': No augmentation
            - 'random_crop': Random crop augmentation
            - 'random_flip': Random horizontal/vertical flip augmentation
        num_iterations (int): Number of times to process each image (for data augmentation)
    
    Returns:
        List[Dict]: List of feature dictionaries, one per image per iteration
    """
    all_features = []
    
    for iteration in range(num_iterations):
        print(f"Processing iteration {iteration + 1}/{num_iterations}...")
        batch_features = []
        
        for img_path in image_paths:
            try:
                if augmentation_type == 'none':
                    feats_dict = extract_features_no_augmentation(model, img_path, text_prompts)
                elif augmentation_type == 'random_crop':
                    feats_dict = extract_features_with_random_crop(model, img_path, text_prompts)
                elif augmentation_type == 'random_flip':
                    # Apply augmentation only if iteration > 0 (first iteration uses original)
                    apply_aug = (iteration > 0)
                    feats_dict = extract_features_with_random_augmentation(
                        model, img_path, text_prompts, apply_augmentation=apply_aug
                    )
                else:
                    raise ValueError(f"Unknown augmentation type: {augmentation_type}")
                
                batch_features.append(feats_dict)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"Extracted features for {len(batch_features)} images in iteration {iteration + 1}")
        all_features.extend(batch_features)
    
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
    - txt_path: Path to text file containing image paths
    - output_path: Path where features will be saved
    - text_prompts: Text prompts for FLAIR model
    - augmentation_type: Type of data augmentation ('none', 'random_crop', 'random_flip')
    - num_iterations: Number of times to process each image
    """
    # ========== Configuration ==========
    # Path to text file containing image paths (one per line, optionally with labels separated by ';')
    txt_path = 'dataset/fundus/E/subtest_labels.txt'
    
    # Output path for saved features
    output_path = 'Final_feature/FLAIR_feature_E.pickle'
    
    # Text prompts for FLAIR model
    text_prompts = ["high myopia", "glaucoma"]
    
    # Augmentation strategy: 'none', 'random_crop', or 'random_flip'
    augmentation_type = 'none'
    
    # Number of iterations (useful for data augmentation)
    num_iterations = 1
    
    # ========== Feature Extraction Pipeline ==========
    # Initialize FLAIR model
    print("Loading FLAIR model...")
    model = FLAIRModel(from_checkpoint=True)
    print("Model loaded successfully.")
    
    # Load image paths
    print(f"Loading image paths from {txt_path}...")
    image_paths = load_image_paths_from_txt(txt_path)
    print(f"Found {len(image_paths)} images to process.")
    
    # Extract features
    print(f"Extracting features with augmentation type: {augmentation_type}...")
    all_features = extract_features_batch(
        model=model,
        image_paths=image_paths,
        text_prompts=text_prompts,
        augmentation_type=augmentation_type,
        num_iterations=num_iterations
    )
    
    # Save features
    print(f"Total features extracted: {len(all_features)}")
    save_features(all_features, output_path)
    
    print("Feature extraction completed!")


if __name__ == "__main__":
    main()

