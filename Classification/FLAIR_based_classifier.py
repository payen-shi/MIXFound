"""
FLAIR.py
--------------------------------------------------
Training and Validation script for the FLAIR Classification Decoder.
This script loads pre-extracted features (from pickle files), trains a 
linear classification head, and evaluates performance using metrics 
like AUC, F1-score, and Confusion Matrices.

Key Changes from original:
1. Removed hardcoded paths and personal info.
2. Added English docstrings and comments.
3. Centralized configuration via argparse.
4. Optimized import structure.

Usage:
    python FLAIR2.py --name flair_exp --feature_path ./features --data_path ./dataset --epochs 20
"""

import sys
import os
import argparse
import json
import copy
import time
import pickle
import random
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# PyTorch & Torchvision
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.transforms import InterpolationMode

# Scikit-learn & Metrics
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.utils import resample

# Monai
from monai.metrics import compute_roc_auc

# Local modules
sys.path.append('./')
import utils
import models
from models.head import ClsHead
# from dataset import ClsImgs, ImageFolderDataset # Uncomment if needed


def confusion_matrix_figure(prediction, target, task, seed, output_dir, class_names=None):
    """
    Generates and saves the confusion matrix figure.
    """
    pred_labels = torch.argmax(prediction, 1).cpu().numpy()
    true_labels = target.cpu().numpy()
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    if class_names is None:
        class_names = ["Glaucoma", "Myopia", "Normal", "AMD"]
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(
        ax=ax,
        cmap=plt.cm.Blues,
        text_kw={'fontsize': 16},
        colorbar=False  
    )
    
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    save_path = os.path.join(output_dir, "class_confusion_matrix")
    os.makedirs(save_path, exist_ok=True)
    
    filename = f"FLAIR_Noaug_{task}_{seed}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=600, bbox_inches='tight')
    plt.close()


def compute_roc_auc2(
    y_pred: torch.Tensor, y: torch.Tensor, average: str = "macro"
) -> Union[float, List[float]]:
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Handles multiclass and multilabel inputs.
    """
    y_pred_ndim = y_pred.ndimension()
    y_ndim = y.ndimension()
    
    if y_pred_ndim not in (1, 2):
        raise ValueError(f"Predictions shape error: {y_pred.shape}")
    if y_ndim not in (1, 2):
        raise ValueError(f"Targets shape error: {y.shape}")
        
    if y_pred_ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(dim=-1)
        y_pred_ndim = 1
    if y_ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(dim=-1)

    if y_pred_ndim == 1:
        return _calculate_auc(y_pred, y)

    if y.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y {y.shape}.")

    if average == "micro":
        return _calculate_auc(y_pred.flatten(), y.flatten())
    
    y, y_pred = y.transpose(0, 1), y_pred.transpose(0, 1)
    auc_values = [_calculate_auc(y_pred_, y_) for y_pred_, y_ in zip(y_pred, y)]
    
    if average == "none":
        return auc_values
    if average == "macro":
        return np.mean(auc_values)
    if average == "weighted":
        weights = [sum(y_) for y_ in y]
        return np.average(auc_values, weights=weights)
        
    raise ValueError(f'Unsupported average: {average}')


def _calculate_auc(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    """Helper for sklearn roc_auc_score."""
    return roc_auc_score(y.cpu().numpy(), y_pred.cpu().numpy())


def bootstrap_auc_ci(
    y_pred: torch.Tensor, y: torch.Tensor, average: str = "macro", 
    n_bootstraps: int = 1000, confidence_level: float = 0.95
) -> tuple:
    """
    Computes ROC AUC and its confidence interval using Bootstrap resampling.
    """
    y_pred_np = y_pred.cpu().numpy()
    y_np = y.cpu().numpy()
    auc_values = []

    for _ in range(n_bootstraps):
        y_pred_resampled, y_resampled = resample(y_pred_np, y_np)
        y_pred_resampled = torch.tensor(y_pred_resampled)
        y_resampled = torch.tensor(y_resampled)
        try:
            auc = compute_roc_auc2(y_pred_resampled, y_resampled, average=average)
            auc_values.append(auc)
        except ValueError:
            continue

    if not auc_values:
        return 0.0, (0.0, 0.0)

    lower_bound = np.percentile(auc_values, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(auc_values, (1 + confidence_level) / 2 * 100)
    mean_auc = np.mean(auc_values)

    return mean_auc, (lower_bound, upper_bound)


class DualOutput:
    """Helper to write to both file and console."""
    def __init__(self, file, console):
        self.file = file
        self.console = console
    
    def write(self, message):
        self.file.write(message)
        self.console.write(message)
    
    def flush(self):
        self.file.flush()
        self.console.flush()


def train(linear_classifier, optimizer, loader, epoch):
    """
    Training loop for one epoch.
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    criterion = nn.CrossEntropyLoss()

    for (inp, target) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        # Flatten if necessary, assuming feature input
        inp = inp.view(inp.shape[0], -1)
        
        target = target.cuda(non_blocking=True).long()
        if len(target.shape) == 2:
            target = target.squeeze()
            
        output = linear_classifier(inp)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_network(loader, epoch, linear_classifier, args):
    """
    Validation loop with metrics calculation and logging.
    """
    seed = args.seed
    task = args.Task
    
    # Setup log directory
    log_dir = os.path.join(args.output_dir, "classification_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"FLAIR_{task}_{seed}.log")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n\n{'='*40}\n")
        f.write(f"Execution Time: {timestamp}\n")
        
        original_stdout = sys.stdout
        sys.stdout = DualOutput(f, original_stdout)
        
        try:
            print(f"\n{'='*30} Epoch {epoch} {'='*30}")
            linear_classifier.eval()
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Test:'
            
            targets, preds = [], []
            all_preds_dicts = []
            
            # --- Inference Loop ---
            for inp, target in metric_logger.log_every(loader, 20, header):
                inp = inp.cuda(non_blocking=True)
                batch_size = inp.shape[0]
                inp = inp.view(inp.shape[0], -1)
                
                target = target.cuda(non_blocking=True).long()
                if len(target.shape) == 2:
                    target = target.squeeze()
                
                with torch.no_grad():
                    output = linear_classifier(inp)
                
                num_class = output.shape[1]
                
                if num_class > 1:
                    loss = nn.CrossEntropyLoss()(output, target)
                    preds.append(output.softmax(dim=1).detach().cpu())
                    targets.append(target.detach().cpu())
                else:
                    loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())
                    targets.append(target.detach().cpu())
                    preds.append(output.detach().cpu().sigmoid())

                metric_logger.update(loss=loss.item())
                
                # Store raw predictions
                output_softmax = torch.nn.functional.softmax(output, dim=1)
                for batch_idx in range(batch_size):
                    preds_dict = {
                        'preds': [output_softmax[batch_idx].detach().cpu().numpy()],
                        'labels': target[batch_idx].cpu().numpy()
                    }
                    all_preds_dicts.append(preds_dict)
            
            # --- Save Pickle ---
            pred_dir = os.path.join(args.output_dir, "Final_prediction")
            os.makedirs(pred_dir, exist_ok=True)
            final_pickle_path = os.path.join(pred_dir, f'FLAIR_pred_{task}_{seed}_new.pickle')
            with open(final_pickle_path, 'wb') as file:
                pickle.dump(all_preds_dicts, file)
            
            # --- Compute Metrics ---
            preds_all = torch.cat(preds, dim=0)
            targets_all = torch.cat(targets, dim=0)
            
            # Confusion Matrix
            confusion_matrix_figure(preds_all, targets_all, task, seed, args.output_dir)
            
            num_class = preds_all.shape[1]
            acc1, = utils.accuracy(preds_all, targets_all, topk=(1,))
            acc1 = acc1.item()
            
            # AUC
            target_onehot = torch.nn.functional.one_hot(targets_all, num_class).float()
            auc_list = compute_roc_auc2(preds_all, target_onehot, average="none")
            print(f"AUC per class: {auc_list}")
            
            mean_auc, ci = bootstrap_auc_ci(preds_all, target_onehot, average="macro")
            print(f"Mean AUC: {mean_auc:.4f}")
            print(f"95% Confidence Interval: {ci}")
            
            # Precision, Recall, F1
            pcf = precision_recall_fscore_support(targets_all.numpy(), preds_all.argmax(dim=1).numpy(), average=None)
            precision = pcf[0][1:].mean() # Assuming averaging over classes 1-N (ignoring background/0)
            recall = pcf[1][1:].mean()
            f1 = pcf[2][1:].mean()
            
            # Per-class Stats
            class_correct = [0.] * num_class
            class_total = [0.] * num_class
            predictions = torch.argmax(preds_all, dim=1)
            correct = (predictions == targets_all).squeeze()
            
            for i in range(len(targets_all)):
                label = targets_all[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            
            class_accuracy = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(num_class)]
            print("Per-class Accuracy:", class_accuracy)
            
            # Sensitivity & Specificity
            confusion_matrix_val = torch.zeros((num_class, num_class), dtype=torch.int64)
            for t, p in zip(targets_all, predictions):
                confusion_matrix_val[t.long(), p.long()] += 1

            sensitivities = []
            specificities = []

            print("\n【Per-class Performance】")
            for i in range(num_class):
                TP = confusion_matrix_val[i, i].item()
                FN = confusion_matrix_val[i, :].sum().item() - TP
                FP = confusion_matrix_val[:, i].sum().item() - TP
                TN = confusion_matrix_val.sum().item() - (TP + FN + FP)
                
                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                
                sensitivities.append(sensitivity)
                specificities.append(specificity)
                print(f"Class {i}: Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

            print(f"* Macro Avg Sensitivity: {np.mean(sensitivities):.4f}")
            print(f"* Macro Avg Specificity: {np.mean(specificities):.4f}")

            # Update Logger
            batch_size = 1
            metric_logger.meters['acc1'].update(acc1, n=batch_size)
            metric_logger.meters['auc'].update(mean_auc, n=batch_size)
            metric_logger.meters['precision'].update(precision, n=batch_size)
            metric_logger.meters['recall'].update(recall, n=batch_size)
            metric_logger.meters['f1'].update(f1, n=batch_size)

            print(
                '* Acc@1 {top1.global_avg:.4f} loss {losses.global_avg:.4f} AUC {auc.global_avg:.4f} '
                'Pre {precision.global_avg:.4f} Recall {recall.global_avg:.4f} F1 {f1.global_avg:.4f}'
                .format(top1=metric_logger.acc1, losses=metric_logger.loss, auc=metric_logger.auc,
                        precision=metric_logger.precision, recall=metric_logger.recall, f1=metric_logger.f1))
                        
        finally:
            sys.stdout = original_stdout
            
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = True

    # Setup output dir
    args.output_dir = os.path.join(args.output_dir, args.name)
    if not os.path.exists(args.output_dir):
        print(f"Creating output_dir: {args.output_dir}")
        os.makedirs(args.output_dir)

    utils.fix_random_seeds(args.seed)
    
    # --- Load Features ---
    # Construct paths using args.feature_path and args.data_path
    feature_dir = args.feature_path
    data_dir = args.data_path
    
    train_feat_path = os.path.join(feature_dir, "FLAIR_feature_A.pickle")
    val_feat_path = os.path.join(feature_dir, f"FLAIR_feature_{args.Task}_new.pickle")
    
    print(f"Loading features from: {train_feat_path}")
    with open(train_feat_path, 'rb') as file:
        train_feats = pickle.load(file)
    with open(val_feat_path, 'rb') as file:
        val_feats = pickle.load(file)

    # Label files
    train_txt_path = os.path.join(data_dir, 'training', 'training_labels.txt')
    test_txt_path = os.path.join(data_dir, args.Task, 'subtest_labels.txt')
    
    # --- Process Train Data ---
    # Logic: Map substrings in filename/label to integers
    train_labels_data = []
    for feat in train_feats:
        label = feat['img_path']
        if 'Glaucoma' in label:
            train_labels_data.append(0)
        elif 'High myopia' in label:
            train_labels_data.append(1)
        elif 'Normal fundus' in label:
            train_labels_data.append(2)
        elif 'Age related macular degeneration (AMD)' in label:
            train_labels_data.append(3)
    
    # Filter based on text file
    with open(train_txt_path, 'r') as file:
        txt_img_paths = [path.strip().split(';')[0] for path in file.readlines()]

    selected_train_feats_data = []
    selected_train_feats_labels = []
    
    for feat, label in zip(train_feats, train_labels_data):
        img_path = feat['img_path']
        if img_path in txt_img_paths:
            selected_train_feats_data.append(feat['feats'])
            selected_train_feats_labels.append(label)
            
    selected_train_feats_data = torch.tensor(selected_train_feats_data)
    selected_train_feats_labels = torch.tensor(selected_train_feats_labels)
    
    train_dataset = torch.utils.data.TensorDataset(selected_train_feats_data, selected_train_feats_labels)
    print(f"Training set size: {selected_train_feats_data.shape[0]}")

    # --- Process Validation Data ---
    val_labels_data = []
    for feat in val_feats:
        label = feat['img_path']
        if 'Glaucoma' in label:
            val_labels_data.append(0)
        elif 'High myopia' in label:
            val_labels_data.append(1)
        elif 'Normal fundus' in label:
            val_labels_data.append(2)
        elif 'Age related macular degeneration (AMD)' in label:
            val_labels_data.append(3)
            
    with open(test_txt_path, 'r') as file:
        txt_img_paths = [path.strip().split(';')[0] for path in file.readlines()]
        
    img_path_to_feat = {feat['img_path']: feat['feats'] for feat in val_feats}
    img_path_to_label = {feat['img_path']: label for feat, label in zip(val_feats, val_labels_data)}

    selected_val_feats_data = []
    selected_val_feats_labels = []

    for img_path in txt_img_paths:
        if img_path in img_path_to_feat:
            selected_val_feats_data.append(img_path_to_feat[img_path])
            selected_val_feats_labels.append(img_path_to_label[img_path])
            
    selected_val_feats_data = torch.tensor(selected_val_feats_data)
    selected_val_feats_labels = torch.tensor(selected_val_feats_labels)
    
    val_dataset = torch.utils.data.TensorDataset(selected_val_feats_data, selected_val_feats_labels)
    print(f"Validation set size: {selected_val_feats_data.shape[0]}")

    # Create DataLoaders
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Build Classifier ---
    embed_dim = 768 # Default embedding dimension
    # Adjust input dimension based on pooling strategy if needed
    if args.avgpool_patchtokens == 0:
        # NOTE: Original code used 512 here specifically for avgpool=0
        input_dim = 512 
        linear_classifier = ClsHead(embed_dim=input_dim, num_classes=args.num_labels, layers=1)
    elif args.avgpool_patchtokens == 1:
        input_dim = embed_dim
        linear_classifier = ClsHead(embed_dim=input_dim, num_classes=args.num_labels, layers=3)

    print(f"Classifier Parameters: {utils.get_parameter_number(linear_classifier)}")
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # Optimizer
    optimizer = torch.optim.AdamW(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        betas=(0.9, 0.999), weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Resume
    to_restore = {"epoch": 0, "best_f1": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(
            args.load_from,
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler)

    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_f1"]

    # --- Main Loop ---
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        linear_classifier.train()
        train_stats = train(linear_classifier, optimizer, train_loader, epoch)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            linear_classifier.eval()
            test_stats = validate_network(val_loader, epoch, linear_classifier, args)

            log_stats = {**{k: v for k, v in log_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process() and (test_stats["f1"] >= best_f1):
                log_path = os.path.join(args.output_dir, "log.txt")
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": test_stats["f1"],
                }
                torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint_{args.checkpoint_key}_linear.pth"))

            best_f1 = max(best_f1, test_stats["f1"])
            print(f'Max F1 so far: {best_f1:.4f}')
            
    print(f"Training completed. Best F1-score: {best_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FLAIR Classification Decoder Training')
    
    # Experiment Config
    parser.add_argument('--name', type=str, default='flair_experiment', help='Trial name')
    parser.add_argument('--output_dir', default="./results", help='Output directory')
    parser.add_argument('--seed', default=0, choices=[0, 32, 64, 128, 256], type=int)
    parser.add_argument('--Task', default='E', choices=['A', 'B','C','D','E','F','G'], type=str, help='Task ID')

    # Data Config
    parser.add_argument('--feature_path', default='Final_feature', type=str, help='Directory containing feature pickle files')
    parser.add_argument('--data_path', default='dataset/fundus', type=str, help='Directory containing label text files')
    parser.add_argument('--dataset_format', default='vfm', choices=['vfm', 'ImageNet'], type=str)
    
    # Model Config
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture name')
    parser.add_argument('--input_size', type=int, default=224, help='Input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size')
    
    # Decoder Config
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int)
    parser.add_argument('--num_labels', default=4, type=int, help='Number of classes')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="Blocks to concatenate (if used)")

    # Training Config
    parser.add_argument('--epochs', default=20, type=int, help='Total epochs')
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Batch size per GPU')
    parser.add_argument('--num_workers', default=10, type=int, help='Workers per GPU')
    parser.add_argument('--val_freq', default=1, type=int, help="Validation frequency")
    parser.add_argument('--pretrained_weights', default=None, type=str, help='Pretrained weights path')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Checkpoint key')
    parser.add_argument('--load_from', default=None, help='Path to resume checkpoint')
    parser.add_argument('--test', action='store_true', help='Inference only')

    # Distributed
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--modality', default='Fundus', type=str)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for checkpoint_key in args.checkpoint_key.split(','):
        print(f"Starting evaluation for {checkpoint_key}.")
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        main(args_copy)