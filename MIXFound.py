"""
Ensemble_Mixture.py
--------------------------------------------------
Ensemble script for combining predictions from multiple models 
(VisionFM, RETFound, FLAIR, CLIP) using an AUC-weighted fusion strategy.

Key Features:
- Loads prediction pickle files from different models.
- Aligns predictions based on image paths and ground truth labels.
- Computes ensemble weights based on per-class AUC performance.
- Evaluates the fused predictions with Bootstrap Confidence Intervals.

Usage:
    python Ensemble_Mixture.py --data_path ./dataset/fundus --output_dir ./results
"""

import sys
import os
import argparse
import json
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict

# Scikit-learn
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.utils import resample
from scipy import interp

import utils as utils_mixfound

# -----------------------------------------------------------------------------
# Helper Classes & Functions
# -----------------------------------------------------------------------------

class DualOutput:
    """Helper class to duplicate stdout to both console and a log file."""
    def __init__(self, file, console):
        self.file = file
        self.console = console
    
    def write(self, message):
        self.file.write(message)
        self.console.write(message)
    
    def flush(self):
        self.file.flush()
        self.console.flush()


def compute_roc_auc_multiclass(y_pred: torch.Tensor, y: torch.Tensor, average: str = "macro") -> Union[float, List[float]]:
    """
    Computes ROC AUC for multiclass classification.
    """
    y_pred_np = y_pred.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # Handle binary case disguised as multi-dim
    if y_pred.shape[1] == 1:
        return roc_auc_score(y_np, y_pred_np)

    # One-vs-Rest AUC
    try:
        if average == "none":
            # Return list of AUCs per class
            n_classes = y_pred.shape[1]
            y_onehot = np.eye(n_classes)[y_np]
            return [roc_auc_score(y_onehot[:, i], y_pred_np[:, i]) for i in range(n_classes)]
        else:
            y_onehot = np.eye(y_pred.shape[1])[y_np]
            return roc_auc_score(y_onehot, y_pred_np, average=average, multi_class='ovr')
    except ValueError:
        return 0.0


def bootstrap_auc_ci(y_pred: torch.Tensor, y: torch.Tensor, average: str = "macro", n_bootstraps: int = 1000) -> tuple:
    """
    Computes Mean AUC and 95% Confidence Interval using Bootstrap resampling.
    """
    y_pred_np = y_pred.cpu().numpy()
    y_np = y.cpu().numpy()
    auc_values = []

    for _ in range(n_bootstraps):
        y_pred_resampled, y_resampled = resample(y_pred_np, y_np)
        try:
            # We use a simplified call here for speed in bootstrap
            y_onehot = np.eye(y_pred.shape[1])[y_resampled]
            val = roc_auc_score(y_onehot, y_pred_resampled, average=average, multi_class='ovr')
            auc_values.append(val)
        except ValueError:
            continue

    if not auc_values:
        return 0.0, (0.0, 0.0)

    lower_bound = np.percentile(auc_values, 2.5)
    upper_bound = np.percentile(auc_values, 97.5)
    mean_auc = np.mean(auc_values)

    return mean_auc, (lower_bound, upper_bound)


def evaluate_metrics(preds_all, targets_all, n_bootstrap=1000):
    """
    Comprehensive evaluation including Accuracy, AUC, Sensitivity, Specificity, 
    Precision, Recall, F1 with Bootstrap Confidence Intervals.
    """
    device = preds_all.device
    num_classes = preds_all.shape[1]
    targets_all = targets_all.cpu()
    preds_all = preds_all.cpu()
    
    preds_np = preds_all.numpy()
    targets_np = targets_all.numpy()
    predictions = torch.argmax(preds_all, dim=1)

    # Structure to hold results
    results = {
        'per_class': {},
        'overall': {}
    }

    # --- Basic Metrics ---
    # 1. Confusion Matrix
    cm = confusion_matrix(targets_np, predictions.numpy(), labels=list(range(num_classes)))
    
    sensitivities = []
    specificities = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        
        sens = TP / (TP + FN + 1e-6)
        spec = TN / (TN + FP + 1e-6)
        sensitivities.append(sens)
        specificities.append(spec)
    
    results['per_class']['Sensitivity'] = sensitivities
    results['per_class']['Specificity'] = specificities

    # 2. Precision/Recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predictions.numpy(), average=None, zero_division=0
    )
    results['per_class']['Precision'] = precision.tolist()
    results['per_class']['Recall'] = recall.tolist()
    results['per_class']['F1'] = f1.tolist()

    # 3. Overall Accuracy
    overall_acc = accuracy_score(targets_np, predictions.numpy())
    results['overall']['Accuracy'] = {'value': overall_acc}

    # --- Bootstrap for CI ---
    bootstrap_metrics = defaultdict(list)

    for _ in range(n_bootstrap):
        indices = resample(np.arange(len(targets_np)))
        pred_sample = preds_np[indices]
        target_sample = targets_np[indices]
        pred_labels = np.argmax(pred_sample, axis=1)

        # Overall Accuracy
        bootstrap_metrics['acc'].append(accuracy_score(target_sample, pred_labels))

        # Macro AUC
        try:
            y_onehot = np.eye(num_classes)[target_sample]
            auc_val = roc_auc_score(y_onehot, pred_sample, average='macro', multi_class='ovr')
            bootstrap_metrics['auc'].append(auc_val)
        except ValueError:
            pass
        
        # Macro F1
        f1_val = precision_recall_fscore_support(target_sample, pred_labels, average='macro', zero_division=0)[2]
        bootstrap_metrics['f1'].append(f1_val)

    # Helper for CI
    def get_ci(values):
        if not values: return (0, 0)
        return (np.percentile(values, 2.5), np.percentile(values, 97.5))

    results['overall']['Accuracy']['ci'] = get_ci(bootstrap_metrics['acc'])
    results['overall']['AUC'] = {
        'value': np.mean(bootstrap_metrics['auc']) if bootstrap_metrics['auc'] else 0,
        'ci': get_ci(bootstrap_metrics['auc'])
    }
    results['overall']['F1'] = {
        'value': np.mean(bootstrap_metrics['f1']) if bootstrap_metrics['f1'] else 0,
        'ci': get_ci(bootstrap_metrics['f1'])
    }
    
    return results


def plot_combined_roc(models_dict, y_true, class_names, output_path=None):
    """Plots ROC curves for multiple models and the ensemble."""
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    colors = ['#219EBC', '#023047', '#FFB703', '#FB8402', '#d62728'] 
    n_classes = len(class_names)
    y_true_onehot = np.eye(n_classes)[y_true]
    
    color_idx = 0
    for model_name, y_pred in models_dict.items():
        # Compute macro-average ROC
        fpr = dict()
        tpr = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        
        ax.plot(all_fpr, mean_tpr, color=colors[color_idx % len(colors)], 
                label=f'{model_name} (Avg AUC = {macro_auc:.4f})', lw=2)
        color_idx += 1
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc='lower right', fontsize=16)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'ROC curves saved to {output_path}')
    else:
        plt.show()
    plt.close()


def load_and_align_data(prediction_files: Dict[str, str], label_file_path: str):
    """
    Loads prediction pickles and aligns them all to the order of the label file.
    
    Args:
        prediction_files: Dict {ModelName: PathToPickle}
        label_file_path: Path to the text file containing the ground truth order.
    
    Returns:
        aligned_preds: Dict {ModelName: numpy array [N, C]}
        aligned_labels: numpy array [N]
        image_paths: List of image paths
    """
    # 1. Load Ground Truth Labels and Order
    with open(label_file_path, 'r') as f:
        # Assuming format: path;label or just path
        target_paths = [line.strip().split(';')[0] for line in f.readlines()]
    
    # 2. Process each model
    aligned_preds = {}
    reference_labels = None

    for model_name, pkl_path in prediction_files.items():
        print(f"Loading {model_name} from {pkl_path}...")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"{pkl_path} not found.")
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data from list of dicts
        # Expected dict keys: 'preds', 'img_path', 'labels'
        curr_preds = [item['preds'] for item in data]
        curr_paths = [item['img_path'] for item in data]
        curr_labels = [item['labels'] for item in data]
        
        # Create mapping for O(1) access
        path_to_idx = {p: i for i, p in enumerate(curr_paths)}
        
        # Align
        aligned_model_preds = []
        aligned_model_labels = []
        
        for target_p in target_paths:
            # Handle potential whitespace or path differences if necessary
            target_p = target_p.strip()
            if target_p in path_to_idx:
                idx = path_to_idx[target_p]
                aligned_model_preds.append(curr_preds[idx])
                aligned_model_labels.append(curr_labels[idx])
            else:
                print(f"Warning: {target_p} not found in {model_name} predictions.")
                # Insert dummy or handle error (here we assume strict alignment)
        
        aligned_preds[model_name] = np.array(aligned_model_preds).reshape(len(aligned_model_preds), -1)
        
        # Sanity Check on labels
        current_labels_np = np.array(aligned_model_labels)
        if reference_labels is None:
            reference_labels = current_labels_np
        else:
            # Ensure labels match across different model files for the same image
            if not np.array_equal(reference_labels, current_labels_np):
                print(f"Warning: Label mismatch detected in {model_name}!")
    
    return aligned_preds, reference_labels, target_paths


def main(args):
    # Setup Output
    utils_mixfound.init_distributed_mode(args) # Only if using DDP utilities, else safe to remove
    args.output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, f"Mixture_Log_{args.Task}.log")
    
    # Setup Logging
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n\n{'='*40}\n")
        f.write(f"Execution Time: {timestamp}\n")
        
        original_stdout = sys.stdout
        sys.stdout = DualOutput(f, original_stdout)
        
        try:
            # --- 1. Define File Paths ---
            # NOTE: These paths should ideally be passed via args or config file
            pred_base_dir = args.prediction_dir
            task_label = args.Task # e.g., 'G'
            
            # Map model names to their pickle files
            model_files = {
                'VisionFM': os.path.join(pred_base_dir, 'Final_prediction', f'VisionFM_pred_{task_label}_256.pickle'),
                'RETFound': os.path.join(pred_base_dir, 'Final_prediction', f'RETFound_pred_{task_label}_32.pickle'),
                'FLAIR':    os.path.join(pred_base_dir, 'Final_prediction', f'FLAIR_pred_{task_label}_64.pickle'),
                'CLIP':     os.path.join(pred_base_dir, 'Final_prediction', f'CLIP_pred_{task_label}_64.pickle')
            }
            
            label_file = os.path.join(args.data_path, task_label, 'subtest_labels.txt')
            
            # --- 2. Load and Align Data ---
            aligned_preds, val_labels, img_paths = load_and_align_data(model_files, label_file)
            
            # Convert labels to one-hot for AUC calculation with evaluation dataset
            num_classes = args.num_labels
            y_true_onehot = np.eye(num_classes)[val_labels]
            
            # --- 3. Compute Ensemble Weights (AUC-based) ---
            print("\nComputing Ensemble Weights...")
            model_names = ['VisionFM', 'RETFound', 'FLAIR', 'CLIP']
            auc_matrix = []
            
            for name in model_names:
                preds = aligned_preds[name]
                # Calculate per-class AUC with evaluation dataset
                aucs = [roc_auc_score(y_true_onehot[:, c], preds[:, c]) for c in range(num_classes)]
                auc_matrix.append(aucs)
            
            auc_matrix = np.array(auc_matrix) # Shape: [4, 4]
            
            # Apply power weighting heuristic
            # Power 12 emphasizes the best models significantly
            auc_power = np.power(auc_matrix, 12)
            auc_weights = auc_power / np.sum(auc_power, axis=0, keepdims=True)
            
            print("AUC Weights Matrix (Rows=Models, Cols=Classes):")
            print(auc_weights)
            
            # --- 4. Perform Fusion ---
            fused_preds = np.zeros_like(aligned_preds['VisionFM'])
            
            for i, name in enumerate(model_names):
                # Weight shape [1, C] * Preds shape [N, C] -> Broadcasts correctly
                weight = auc_weights[i][None, :]
                fused_preds += weight * aligned_preds[name]
            
            # Define final predictions (Use fused_preds for ensemble)
            final_predictions = fused_preds 
            # Note: In original code, it was overridden by VisionFM. 
            # If you want single model, change this line.
            
            targets_tensor = torch.tensor(val_labels)
            preds_tensor = torch.tensor(final_predictions)
            preds_label = torch.argmax(preds_tensor, dim=1)
            
            # --- 5. Save Detailed Predictions ---
            result_save_dir = os.path.join(args.output_dir, "analysis")
            os.makedirs(result_save_dir, exist_ok=True)
            result_path = os.path.join(result_save_dir, "predictions_with_labels.txt")
            
            with open(result_path, 'w', encoding='utf-8') as f_res:
                header = f"{'Image Path':<50} | {'Pred':<5} | {'True':<5} | {'Correct':<5}"
                f_res.write(header + '\n' + '-'*len(header) + '\n')
                for path, p, t in zip(img_paths, preds_label.numpy(), val_labels):
                    mark = "✔" if p == t else "✗"
                    f_res.write(f"{path:<50} | {p:<5} | {t:<5} | {mark:<5}\n")
            print(f"Predictions saved to {result_path}")

            # --- 6. Plot ROC ---
            plot_dict = aligned_preds.copy()
            plot_dict['Ensemble'] = final_predictions
            plot_combined_roc(
                plot_dict, 
                val_labels, 
                class_names=["Glaucoma", "Myopia", "Normal", "AMD"],
                output_path=os.path.join(result_save_dir, f"ROC_Curves_{args.Task}.png")
            )

            # --- 7. Evaluate Metrics ---
            print("\n-------- Evaluation Metrics --------")
            metrics = evaluate_metrics(preds_tensor, targets_tensor, n_bootstrap=1000)
            
            # Print Formatted Results
            print(f"Overall Accuracy: {metrics['overall']['Accuracy']['value']:.4f} "
                  f"(CI: {metrics['overall']['Accuracy']['ci'][0]:.4f}-{metrics['overall']['Accuracy']['ci'][1]:.4f})")
            print(f"Macro AUC:        {metrics['overall']['AUC']['value']:.4f}")
            print(f"Macro F1:         {metrics['overall']['F1']['value']:.4f}")
            
            print("\nPer-Class Sensitivity:")
            for i, sens in enumerate(metrics['per_class']['Sensitivity']):
                print(f"  Class {i}: {sens:.4f}")
                
            print("\nPer-Class Specificity:")
            for i, spec in enumerate(metrics['per_class']['Specificity']):
                print(f"  Class {i}: {spec:.4f}")

        finally:
            sys.stdout = original_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ensemble Mixture Evaluation')
    
    # Paths
    parser.add_argument('--name', type=str, default='ensemble_results', help='Experiment name')
    parser.add_argument('--data_path', default='dataset/fundus', type=str, help='Root dataset path containing labels')
    parser.add_argument('--prediction_dir', default='/', type=str, help='Root directory where pickles are stored')
    parser.add_argument('--output_dir', default="./results", help='Directory to save results')
    
    # Task Config
    parser.add_argument('--Task', default='G', type=str, help='Task Identifier (e.g., G)')
    parser.add_argument('--num_labels', default=4, type=int, help='Number of classes')
    parser.add_argument('--seed', default=0, type=int)
    
    # Dummy args to satisfy init_distributed_mode if imported from utils
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)