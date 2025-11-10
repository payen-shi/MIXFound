import sys
sys.path.append('./')
import os
import argparse
import json
import copy
import torch
import pickle
import torch.backends.cudnn as cudnn
import utils
import models
import numpy as np
from models.head import ClsHead
from collections import defaultdict
from pathlib import Path
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision import datasets
import transforms as self_transforms
from dataset import ClsImgs, ImageFolderDataset
from sklearn.metrics import precision_recall_fscore_support
from monai.metrics import compute_roc_auc
from evaluation_funcs import performance_single_cls
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import entropy
import torch.nn.functional as F
from scipy.spatial import distance
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch
import numpy as np
from typing import Union, List
from sklearn.utils import resample
import cv2
import os
import contextlib
from datetime import datetime
import time
import numpy as np
from typing import Union, List
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

def evaluate_metrics(preds_all, targets_all, n_bootstrap=1000):
    """
    多分类评估函数 (含准确率及置信区间)
    
    参数:
    preds_all (Tensor): 预测概率矩阵 [N, C]
    targets_all (Tensor): 真实标签 [N]
    n_bootstrap (int): Bootstrap采样次数
    
    返回:
    dict: 包含分层指标的嵌套字典结构
    """
    device = preds_all.device
    num_classes = preds_all.shape[1]
    targets_all = targets_all.cpu()
    preds_all = preds_all.cpu()
    
    preds_np = preds_all.numpy()
    targets_np = targets_all.numpy()

    results = {
        'per_class': {
            'AUC': [],
            'Accuracy': [],
            'Sensitivity': [],
            'Specificity': [],
            'Precision': [],
            'Recall': [],
            'F1': []
        },
        'overall': {
            'Accuracy': {'value': 0, 'ci': (0, 0)},
            'AUC': {'value': 0, 'ci': (0, 0)},
            'Sensitivity': {'value': 0, 'ci': (0, 0)},
            'Specificity': {'value': 0, 'ci': (0, 0)},
            'Precision': {'value': 0, 'ci': (0, 0)},
            'Recall': {'value': 0, 'ci': (0, 0)},
            'F1': {'value': 0, 'ci': (0, 0)}
        }
    }

    predictions = torch.argmax(preds_all, dim=1)
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    for label, pred in zip(targets_all, predictions):
        class_total[label] += 1
        class_correct[label] += (pred == label).item()
    class_acc = (class_correct / class_total.clamp(min=1e-6)).tolist()
    results['per_class']['Accuracy'] = class_acc

    overall_acc = accuracy_score(targets_np, predictions.numpy())
    results['overall']['Accuracy']['value'] = overall_acc

    target_onehot = torch.nn.functional.one_hot(targets_all, num_classes).float()
    auc_per_class = []
    for i in range(num_classes):
        auc = roc_auc_score((targets_np == i).astype(int), preds_np[:, i])
        auc_per_class.append(auc)
    results['per_class']['AUC'] = auc_per_class

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets_all, predictions):
        cm[t, p] += 1

    sensitivity = []
    specificity = []
    for i in range(num_classes):
        TP = cm[i, i].item()
        FN = cm[i, :].sum().item() - TP
        TN = cm.sum().item() - cm[i, :].sum() - cm[:, i].sum() + TP
        FP = cm[:, i].sum().item() - TP
        
        sensitivity.append(TP / (TP + FN + 1e-6))
        specificity.append(TN / (TN + FP + 1e-6))
    
    results['per_class']['Sensitivity'] = sensitivity
    results['per_class']['Specificity'] = specificity

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predictions.numpy(), average=None
    )
    results['per_class']['Precision'] = precision.tolist()
    results['per_class']['Recall'] = recall.tolist()
    results['per_class']['F1'] = f1.tolist()

    def bootstrap_sample():
        indices = np.random.choice(len(targets_np), len(targets_np), replace=True)
        return indices

    bootstrap_metrics = {
        'overall_acc': [],
        'class_acc': [[] for _ in range(num_classes)],
        'macro_auc': [],
        'macro_sensitivity': [],
        'macro_specificity': [],
        'macro_precision': [],
        'macro_recall': [],
        'macro_f1': []
    }

    for _ in range(n_bootstrap):
        indices = bootstrap_sample()
        pred_sample = preds_all[indices]
        target_sample = targets_all[indices]
        pred_labels = torch.argmax(pred_sample, dim=1)
        
        acc = (pred_labels == target_sample).float().mean().item()
        bootstrap_metrics['overall_acc'].append(acc)
        
        for c in range(num_classes):
            mask = (target_sample == c)
            if mask.sum() == 0:
                bootstrap_metrics['class_acc'][c].append(0.0)
            else:
                class_acc = (pred_labels[mask] == c).float().mean().item()
                bootstrap_metrics['class_acc'][c].append(class_acc)
        
        auc_macro = roc_auc_score(
            torch.nn.functional.one_hot(target_sample, num_classes).numpy(),
            pred_sample.numpy(),
            multi_class='ovr',
            average='macro'
        )
        bootstrap_metrics['macro_auc'].append(auc_macro)
        
        cm_sample = torch.zeros((num_classes, num_classes))
        for t, p in zip(target_sample, pred_labels):
            cm_sample[t, p] += 1
        
        sens = []
        spec = []
        for c in range(num_classes):
            TP = cm_sample[c, c].item()
            FN = cm_sample[c, :].sum().item() - TP
            TN = cm_sample.sum().item() - cm_sample[c, :].sum() - cm_sample[:, c].sum() + TP
            FP = cm_sample[:, c].sum().item() - TP
            
            sens.append(TP / (TP + FN + 1e-6))
            spec.append(TN / (TN + FP + 1e-6))
        
        bootstrap_metrics['macro_sensitivity'].append(np.mean(sens))
        bootstrap_metrics['macro_specificity'].append(np.mean(spec))
        
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            target_sample.numpy(), pred_labels.numpy(), average='macro'
        )
        bootstrap_metrics['macro_precision'].append(prec_macro)
        bootstrap_metrics['macro_recall'].append(rec_macro)
        bootstrap_metrics['macro_f1'].append(f1_macro)

    def calculate_ci(data, is_per_class=False):
        alpha = (1 - 0.95) / 2
        if is_per_class:
            return [(np.percentile(class_data, 100*alpha), 
                    np.percentile(class_data, 100*(1-alpha))) 
                   for class_data in data]
        else:
            return (np.percentile(data, 100*alpha), np.percentile(data, 100*(1-alpha)))

    class_acc_ci = calculate_ci(bootstrap_metrics['class_acc'], is_per_class=True)
    for c in range(num_classes):
        results['per_class']['Accuracy'][c] = {
            'value': results['per_class']['Accuracy'][c],
            'ci': class_acc_ci[c]
        }
    
    results['overall']['Accuracy']['ci'] = calculate_ci(bootstrap_metrics['overall_acc'])
    results['overall']['AUC'] = {
        'value': np.mean(bootstrap_metrics['macro_auc']),
        'ci': calculate_ci(bootstrap_metrics['macro_auc'])
    }
    results['overall']['Sensitivity'] = {
        'value': np.mean(bootstrap_metrics['macro_sensitivity']),
        'ci': calculate_ci(bootstrap_metrics['macro_sensitivity'])
    }
    results['overall']['Specificity'] = {
        'value': np.mean(bootstrap_metrics['macro_specificity']),
        'ci': calculate_ci(bootstrap_metrics['macro_specificity'])
    }
    results['overall']['Precision'] = {
        'value': np.mean(bootstrap_metrics['macro_precision']),
        'ci': calculate_ci(bootstrap_metrics['macro_precision'])
    }
    results['overall']['Recall'] = {
        'value': np.mean(bootstrap_metrics['macro_recall']),
        'ci': calculate_ci(bootstrap_metrics['macro_recall'])
    }
    results['overall']['F1'] = {
        'value': np.mean(bootstrap_metrics['macro_f1']),
        'ci': calculate_ci(bootstrap_metrics['macro_f1'])
    }

    return results
    
    
    
    
            
            

    
    
    

def plot_combined_roc(models_dict, y_true, class_names, output_path=None):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    colors = ['#219EBC', '#023047', '#FFB703', '#FB8402','#d62728'] 
    line_styles = ['-', '-', '-', '-', '-']
    
    n_classes = len(class_names)
    y_true_onehot = np.eye(n_classes)[y_true]
    
    text_elements = []

    for (model_name, y_pred), color, ls in zip(models_dict.items(), colors, line_styles):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        
        ax.plot(all_fpr, mean_tpr, color=color, linestyle=ls,
                label=f'{model_name} (Avg AUC = {macro_auc:.4f})', lw=2)
        
        if model_name == 'MOFM':
            fusion_class_auc = [roc_auc[i] for i in range(n_classes)]
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)  # 新增刻度字体设置
    
    ax.legend(loc='lower right', fontsize=20)
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f'ROC curves saved to {output_path}')
    else:
        plt.show()
    plt.close()


def compute_roc_auc2(
    y_pred: torch.Tensor, y: torch.Tensor, average: str = "macro"
) -> Union[float, List[float]]:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    y_pred_ndim = y_pred.ndimension()
    y_ndim = y.ndimension()
    if y_pred_ndim not in (1, 2):
        raise ValueError(
            f"Predictions should be of shape (batch_size, num_classes) or (batch_size, ), got {y_pred.shape}."
        )
    if y_ndim not in (1, 2):
        raise ValueError(f"Targets should be of shape (batch_size, num_classes) or (batch_size, ), got {y.shape}.")
    if y_pred_ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(dim=-1)
        y_pred_ndim = 1
    if y_ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(dim=-1)

    if y_pred_ndim == 1:
        return _calculate(y_pred, y)

    if y.shape != y_pred.shape:
        raise ValueError(f"data shapes of y_pred and y do not match, got {y_pred.shape} and {y.shape}.")

    if average == "micro":
        return _calculate(y_pred.flatten(), y.flatten())
    y, y_pred = y.transpose(0, 1), y_pred.transpose(0, 1)
    auc_values = [_calculate(y_pred_, y_) for y_pred_, y_ in zip(y_pred, y)]
    if average == "none":
        return auc_values
    if average == "macro":
        return np.mean(auc_values)
    if average == "weighted":
        weights = [sum(y_) for y_ in y]
        return np.average(auc_values, weights=weights)
    raise ValueError(f'Unsupported average: {average}, available options are ["macro", "weighted", "micro", "none"].')

def _calculate(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    """Helper function to calculate ROC AUC."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y.cpu().numpy(), y_pred.cpu().numpy())

def bootstrap_auc_ci(
    y_pred: torch.Tensor, y: torch.Tensor, average: str = "macro", n_bootstraps: int = 1000, confidence_level: float = 0.95
) -> Union[float, List[float], tuple]:
    """Computes ROC AUC and its confidence interval using Bootstrap method."""
    y_pred_np = y_pred.cpu().numpy()
    y_np = y.cpu().numpy()
    auc_values = []

    for _ in range(n_bootstraps):
        y_pred_resampled, y_resampled = resample(y_pred_np, y_np)
        y_pred_resampled = torch.tensor(y_pred_resampled)
        y_resampled = torch.tensor(y_resampled)
        auc = compute_roc_auc2(y_pred_resampled, y_resampled, average=average)
        auc_values.append(auc)

    lower_bound = np.percentile(auc_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auc_values, (1 + confidence_level) / 2 * 100)
    mean_auc = np.mean(auc_values)

    return mean_auc, (lower_bound, upper_bound)
class DualOutput:
    """同时输出到文件和屏幕的类"""
    def __init__(self, file, console):
        self.file = file
        self.console = console
    
    def write(self, message):
        self.file.write(message)
        self.console.write(message)
    
    def flush(self):
        self.file.flush()
        self.console.flush()
def binary2multi(input_tensor):
    if len(input_tensor.shape) == 2 and input_tensor.shape[1] == 1:
        return torch.cat([1.0 - input_tensor, input_tensor], dim=1)
    elif len(input_tensor.shape) == 1:
        return torch.cat([1.0 - input_tensor.unsqueeze(dim=1), input_tensor.unsqueeze(dim=1)], dim=1)
    else:
        raise NotImplementedError

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 稳定化 softmax，避免溢出
    return e_x / np.sum(e_x, axis=1, keepdims=True)
def main(args):
    VisionFM_val_file_path = '../VisionFM/Final_prediction_Selection/VisionFM_pred_D_128.pickle'
    with open(VisionFM_val_file_path, 'rb') as file:
        VisionFM_val_feats = pickle.load(file)
    val_path = '../VisionFM/dataset/fundus/D/subtest_labels.txt'
    with open(val_path, 'r') as file:
        txt_img_paths = file.readlines()
        txt_img_paths = [path.strip().split(';')[0] for path in txt_img_paths]
    VisionFM_val_feats_data = [feat['preds'] for feat in VisionFM_val_feats]
    VisionFM_val_feats_path= [feat['img_path'] for feat in VisionFM_val_feats]
    VisionFM_val_labels_data = [feat['labels'] for feat in VisionFM_val_feats]
    path_to_index = {path: idx for idx, path in enumerate(VisionFM_val_feats_path)}
    sorted_feats_path = [path.strip() for path in txt_img_paths]  # 确保 txt_img_paths 中路径与 VisionFM_val_feats_path 的一致
    sorted_index = [path_to_index[path] for path in sorted_feats_path]

    sorted_feats_data = [VisionFM_val_feats_data[idx] for idx in sorted_index]
    sorted_labels_data = [VisionFM_val_labels_data[idx] for idx in sorted_index]
    
    VisionFM_val_fea = np.array(sorted_feats_data)
    VisionFM_val_fea = VisionFM_val_fea.reshape(VisionFM_val_fea.shape[0], -1)
    VisionFM_val_label = np.array(sorted_labels_data)
    
    RETFound_val_file_path = '../VisionFM/Final_prediction_Selection/RETFound_pred_D_0.pickle'
    with open(RETFound_val_file_path, 'rb') as file:
        RETFound_val_feats = pickle.load(file)
    RETFound_val_feats_data = [feat['preds'] for feat in RETFound_val_feats]
    RETFound_val_labels_data = [feat['labels'] for feat in RETFound_val_feats]
    RETFound_val_fea = np.array(RETFound_val_feats_data)
    RETFound_val_label = np.array(RETFound_val_labels_data)
    RETFound_val_fea = RETFound_val_fea.reshape(RETFound_val_fea.shape[0], -1)

    FLAIR_val_file_path ='../VisionFM/Final_prediction_Selection/FLAIR_pred_D_64.pickle'
    with open(FLAIR_val_file_path, 'rb') as file:
        FLAIR_val_feats = pickle.load(file)
    FLAIR_val_feats_data = [feat['preds'] for feat in FLAIR_val_feats]
    FLAIR_val_labels_data = [feat['labels'] for feat in FLAIR_val_feats]
    FLAIR_val_fea = np.array(FLAIR_val_feats_data)
    FLAIR_val_label = np.array(FLAIR_val_labels_data)
    FLAIR_val_fea = FLAIR_val_fea.reshape(FLAIR_val_fea.shape[0], -1)
  
    CLIP_val_file_path = '../VisionFM/Final_prediction_Selection/CLIP_pred_D_64.pickle'
    with open(CLIP_val_file_path, 'rb') as file:
        CLIP_val_feats = pickle.load(file)
    with open(val_path, 'r') as file:
        txt_img_paths = file.readlines()
        txt_img_paths = [path.strip().split(';')[0] for path in txt_img_paths]
    CLIP_val_feats_data = [feat['preds'] for feat in CLIP_val_feats]
    CLIP_val_feats_path= [feat['img_path'] for feat in CLIP_val_feats]
    CLIP_val_labels_data = [feat['labels'] for feat in CLIP_val_feats]
    path_to_index = {path: idx for idx, path in enumerate(CLIP_val_feats_path)}
    sorted_feats_path = [path.strip() for path in txt_img_paths]  # 确保 txt_img_paths 中路径与 CLIP_val_feats_path 的一致
    sorted_index = [path_to_index[path] for path in sorted_feats_path]

    sorted_feats_data = [CLIP_val_feats_data[idx] for idx in sorted_index]
    sorted_labels_data = [CLIP_val_labels_data[idx] for idx in sorted_index]
    
    CLIP_val_fea = np.array(sorted_feats_data)
    CLIP_val_fea = CLIP_val_fea.reshape(CLIP_val_fea.shape[0], -1)
    CLIP_val_label = np.array(sorted_labels_data)


    val_label = VisionFM_val_label
    predictions = []
    
    Prediction_VisionFM =  np.argmax(VisionFM_val_fea, axis=1)
    Prediction_RETFound =  np.argmax(RETFound_val_fea, axis=1)
    Prediction_FLAIR =  np.argmax(FLAIR_val_fea, axis=1)
    Prediction_CLIP =  np.argmax(CLIP_val_fea, axis=1)
    


    num_class=4
    
    
    y_true_onehot = np.eye(np.max(val_label) + 1)[val_label]

    auc_VisionFM = [roc_auc_score(y_true_onehot[:, c], VisionFM_val_fea[:, c]) for c in range(y_true_onehot.shape[1])]
    auc_RETFound = [roc_auc_score(y_true_onehot[:, c], RETFound_val_fea[:, c]) for c in range(y_true_onehot.shape[1])]
    auc_FLAIR = [roc_auc_score(y_true_onehot[:, c], FLAIR_val_fea[:, c]) for c in range(y_true_onehot.shape[1])]
    auc_CLIP = [roc_auc_score(y_true_onehot[:, c], CLIP_val_fea[:, c]) for c in range(y_true_onehot.shape[1])]
    
    auc_matrix = np.array([auc_VisionFM, auc_RETFound, auc_FLAIR, auc_CLIP])  # 形状为 (4, n_classes)
    auc_matrix = np.power(auc_matrix, 12)   #A,B,C,D;4
    auc_weights = auc_matrix / np.sum(auc_matrix, axis=0, keepdims=True)  # 形状为 (4, n_classes)

    final_weights_VisionFM = auc_weights[0][None, :] 
    final_weights_RETFound = auc_weights[1][None, :]
    final_weights_FLAIR = auc_weights[2][None, :]
    final_weights_CLIP = auc_weights[3][None, :] 

    fused_preds = (
        final_weights_VisionFM * VisionFM_val_fea +
        final_weights_RETFound * RETFound_val_fea +
        final_weights_FLAIR * FLAIR_val_fea +
        final_weights_CLIP * CLIP_val_fea
    )
    final_predictions = CLIP_val_fea    
    log_file = "../VisionFM/all_score_classification_logs/CLIP_Mixture_D.log"
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n\n{'='*40}\n")
        f.write(f"执行时间: {timestamp}\n")
        
        original_stdout = sys.stdout
        sys.stdout = DualOutput(f, original_stdout)
        try:
            targets_all = torch.tensor(val_label)
            preds_all = torch.tensor(final_predictions)
            print("--------scores_all--------")
            result_all = evaluate_metrics(preds_all, targets_all, num_class)
            print("result_all:", result_all)
            acc1, = utils.accuracy(preds_all, targets_all, topk=(1,))
            acc1 = acc1.item()
            print("mean_acc:", acc1)
            target_onehot = torch.nn.functional.one_hot(targets_all, num_class).float()
            auc_dr_grading_list = compute_roc_auc(preds_all, target_onehot, average="none")
            print(auc_dr_grading_list)
            auc_dr_grading_list2 = compute_roc_auc2(preds_all, target_onehot, average="none")
            print(np.array(auc_dr_grading_list2).mean())
            mean_auc, ci = bootstrap_auc_ci(preds_all, target_onehot, average="macro")
            print(f"Mean AUC: {mean_auc}")
            print(f"95% Confidence Interval: {ci}")
            auc = np.array(auc_dr_grading_list).mean()
            print(auc)
            pcf = precision_recall_fscore_support(targets_all.numpy(), preds_all.argmax(dim=1).numpy(), average=None)
            precision = pcf[0].mean()
            recall = pcf[1].mean()
            f1 = pcf[2].mean()
            class_correct = list(0. for i in range(num_class))
            class_total = list(0. for i in range(num_class))
            predictions = torch.argmax(preds_all, dim=1)
            correct = (predictions == targets_all).squeeze()
            for i in range(len(targets_all)):
                label = targets_all[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            class_accuracy = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(num_class)]
            num_classes = num_class  # 从之前的代码获取类别数
            confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

            for t, p in zip(targets_all, predictions):
                confusion_matrix[t.long(), p.long()] += 1

            sensitivities = []
            specificities = []

            for i in range(num_classes):
                TP = confusion_matrix[i, i].item()
                
                FN = confusion_matrix[i, :].sum().item() - TP
                
                TN = confusion_matrix.sum().item() - confusion_matrix[i, :].sum().item() - confusion_matrix[:, i].sum().item() + TP
                
                FP = confusion_matrix[:, i].sum().item() - TP
                
                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                sensitivities.append(sensitivity)
                
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                specificities.append(specificity)

            print("\n【每个类别的性能指标】")
            for i in range(num_classes):
                print(f"类别 {i}:")
                print(f"├─ Sensitivity: {sensitivities[i]:.4f}")
                print(f"└─ Specificity: {specificities[i]:.4f}")

            macro_sensitivity = np.mean(sensitivities)
            macro_specificity = np.mean(specificities)
            print(f"\n* 宏平均 Sensitivity: {macro_sensitivity:.4f}")
            print(f"* 宏平均 Specificity: {macro_specificity:.4f}")
            incorrect_indices = []
            for i in range(len(targets_all)):
                if targets_all[i] == 2 and predictions[i] != targets_all[i]:
                    incorrect_indices.append(i+1)
            print("每个类别的准确率:", class_accuracy)
            print("AUC:", auc)
            print("F1:",f1)
            print("recall:", recall)
            print("precision:",precision)
        finally:
                sys.stdout = original_stdout
if __name__ == '__main__':
    parser = argparse.ArgumentParser('training a classification decoder on pretrained decoder')
    parser.add_argument('--name', type=str, default='single_cls_debug', help='the trial name')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. """)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int)
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
        'vit_large'], help='Architecture.')
    parser.add_argument('--input_size', type=int, default=224, help='input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='../VisionFM/VFM_Fundus_weights.pth', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--modality', default='Fundus', type=str)
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the beginning of training""")
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Per-GPU batch-size')

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=1, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='../VisionFM/dataset/fundus', type=str, help='Please specify path to the dataset.')
    parser.add_argument('--dataset_format', default='vfm',choices=['vfm', 'ImageNet'], type=str, help='Please specify path to the dataset.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./results", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=4, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument('--test', action='store_true', help='Whether to run inference only')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        main(args_copy)