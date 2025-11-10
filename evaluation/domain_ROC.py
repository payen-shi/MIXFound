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
    
    
    
    
            
            

    
    
    

def plot_combined_roc(datasets_dict, class_names, output_path=None):
    """绘制包含多个数据集宏平均ROC曲线的对比图"""
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    colors = ['#98DE87', '#2CA02C', '#FFB14A', '#D35400']#['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']
    line_styles = ['-', '-', '-', '-']
    
    for idx, (ds_name, (y_pred, y_true)) in enumerate(datasets_dict.items()):
        if not np.allclose(np.sum(y_pred, axis=1), 1):
            y_pred = softmax(y_pred, axis=1)
        
        n_classes = len(class_names)
        y_true_onehot = np.eye(n_classes)[y_true]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        
        ax.plot(all_fpr, mean_tpr, 
                color=colors[idx % len(colors)],
                linestyle=line_styles[idx % len(line_styles)],
                lw=2,
                label=f'{ds_name} ')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=26)
    ax.set_ylabel('True Positive Rate', fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    
    ax.legend(loc='lower right', fontsize=22, frameon=True, framealpha=0.8)
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f'>> ROC曲线已保存至 {output_path}')
    else:
        plt.show()
    plt.close()

def confusion_matrix_figure(prediction,target):
    pred_labels = torch.argmax(prediction,1).cpu().numpy()
    true_labels = target.cpu().numpy()
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Glaucoma", "Myopia", "Normal", "AMD"]  # 替换为你的实际类别名称
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(
        ax=ax,
        cmap=plt.cm.Blues,
        text_kw={'fontsize': 16}, # 调整矩阵内部数字大小，ßß
        colorbar=False  
    )
    ax.set_xlabel('Predicted label', fontsize=16)  # X轴标签ß
    ax.set_ylabel('True label', fontsize=16)       # Y轴标签
    ax.tick_params(axis='both', which='major', labelsize=14)  # 刻度标签
    
    plt.savefig("../VisionFM/class_confusion_matrix/VisionFM_Noaug_A", dpi=600, bbox_inches='tight')
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
    
    VisionFM_val_file_path = '../VisionFM/Final_prediction_Selection/VisionFM_pred_A_64.pickle'
    with open(VisionFM_val_file_path, 'rb') as file:
        VisionFM_val_feats = pickle.load(file)
    val_path = '../VisionFM/dataset/fundus/A/subtest_labels.txt'
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
    VisionFM_val_fea_A = VisionFM_val_fea.reshape(VisionFM_val_fea.shape[0], -1)
    VisionFM_val_label_A = np.array(sorted_labels_data)
    
    VisionFM_val_file_path = '../VisionFM/Final_prediction_Selection/VisionFM_pred_B_0.pickle'
    with open(VisionFM_val_file_path, 'rb') as file:
        VisionFM_val_feats = pickle.load(file)
    val_path = '../VisionFM/dataset/fundus/B/subtest_labels.txt'
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
    VisionFM_val_fea_B = VisionFM_val_fea.reshape(VisionFM_val_fea.shape[0], -1)
    VisionFM_val_label_B = np.array(sorted_labels_data)
    
    VisionFM_val_file_path = '../VisionFM/Final_prediction_Selection/VisionFM_pred_C_32.pickle'
    with open(VisionFM_val_file_path, 'rb') as file:
        VisionFM_val_feats = pickle.load(file)
    val_path = '../VisionFM/dataset/fundus/C/subtest_labels.txt'
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
    VisionFM_val_fea_C = VisionFM_val_fea.reshape(VisionFM_val_fea.shape[0], -1)
    VisionFM_val_label_C = np.array(sorted_labels_data)
    
    
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
    VisionFM_val_fea_D = VisionFM_val_fea.reshape(VisionFM_val_fea.shape[0], -1)
    VisionFM_val_label_D = np.array(sorted_labels_data)
    datasets = {
    'Plain 1': (VisionFM_val_fea_A, VisionFM_val_label_A),
    'Plain 2': (VisionFM_val_fea_B, VisionFM_val_label_B),
    'High Altitude Plateau': (VisionFM_val_fea_C, VisionFM_val_label_C),
    'Very High Altitude Plateau': (VisionFM_val_fea_D, VisionFM_val_label_D)
}

    plot_combined_roc(
        datasets_dict=datasets,
        class_names=["Glaucoma", "Myopia", "Normal", "AMD"],  # 根据实际类别修改
        output_path="combined_roc_curves3.png"
    )
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