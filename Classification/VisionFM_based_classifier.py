"""
VisionFM2.py
--------------------------------------------------
Training and Validation script for VisionFM Classification Decoder.
This script performs single-modal classification training on top of a 
pre-trained Vision Transformer backbone (VisionFM).

Features:
- Supports Distributed Data Parallel (DDP) training.
- Computes comprehensive metrics: AUC, F1, Accuracy, Sensitivity, Specificity.
- Generates Confusion Matrix and ROC-AUC analysis.

Usage:
    python VisionFM2.py --name visionfm_exp --data_path ./dataset/fundus --epochs 20
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

# Local modules (Ensure these exist in your repository)
sys.path.append('./')
import utils
import models
from models.head import ClsHead
import transforms as self_transforms # Custom transforms
from dataset import ClsImgs, ImageFolderDataset
# from evaluation_funcs import performance_single_cls # Uncomment if needed


def confusion_matrix_figure(prediction, target, task, seed, output_dir, class_names=None):
    """
    Generates and saves the confusion matrix figure.

    Args:
        prediction (torch.Tensor): Model logits or probabilities.
        target (torch.Tensor): Ground truth labels.
        task (str): Task identifier.
        seed (int): Random seed.
        output_dir (str): Directory to save the plot.
        class_names (list): List of class string names.
    """
    pred_labels = torch.argmax(prediction, 1).cpu().numpy()
    true_labels = target.cpu().numpy()
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Default class names if not provided
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
    filename = f"VisionFM_{task}_{seed}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=600, bbox_inches='tight')
    plt.close()


def compute_roc_auc2(
    y_pred: torch.Tensor, y: torch.Tensor, average: str = "macro"
) -> Union[float, List[float]]:
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Supports multiclass and multilabel.
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
    """Helper wrapper for sklearn roc_auc_score."""
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


def save_preds(targets, preds, paths: list, dst_json_path):
    """Saves predictions and ground truth to a JSON file."""
    results = defaultdict()

    targets_th = torch.cat(targets, dim=0) # [N] or [N, C]
    preds_th = torch.cat(preds, dim=0)     # [N, C]
    
    num_class = preds_th.shape[1]
    
    # Ensure targets are one-hot for consistency in JSON
    if num_class > 1 and targets_th.dim() == 1:
        targets_th = torch.nn.functional.one_hot(targets_th, num_class).float()
        
    num = preds_th.shape[0]
    for idx in range(num):
        img_path = paths[idx]
        key = img_path # Use path as key
        
        if key not in results.keys():
            if len(preds_th[idx]) > 1:
                results[key] = {
                    'gt': targets_th[idx].tolist(),
                    'pred': preds_th[idx].tolist()
                }
            else:
                results[key] = {
                    'gt': [targets_th[idx].item()],
                    'pred': [preds_th[idx].item()]
                }

    print(f"Total images processed: {len(results.keys())}")
    with open(dst_json_path, "w") as f:
        f.write(json.dumps(results, indent=4))
    print(f"Predictions saved to {dst_json_path}")


def train(model, linear_classifier, optimizer, loader, epoch, n_blocks, avg_pool):
    """
    Training loop for one epoch.
    
    Args:
        model: Backbone model (VisionFM).
        linear_classifier: Head model.
        optimizer: Optimizer.
        loader: DataLoader.
        epoch: Current epoch number.
        n_blocks: Number of last blocks to concatenate.
        avg_pool: Pooling strategy (0: CLS token, 1: Avg pooling).
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    # Use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    for (inp, target, extras) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).long()
        
        if len(target.shape) == 2:
            target = target.squeeze()

        # Forward Pass
        with torch.no_grad():
            # Extract features from backbone
            intermediate_output = model.get_intermediate_layers(inp, n_blocks)
            
            if avg_pool == 0:
                # Retain CLS tokens only
                output = [x[:, 0] for x in intermediate_output]
            elif avg_pool == 1:
                # Average pool patch tokens (excluding CLS)
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
                
            output = torch.cat(output, dim=-1)
        
        # Classifier Forward
        output = linear_classifier(output)
        num_class = output.shape[1]
        
        if num_class > 1:
            loss = criterion(output, target)
        else:
            loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, epoch, model, linear_classifier, n_blocks, avg_pool, args, dst_json_path=None):
    """
    Validation loop with comprehensive metric calculation and logging.
    """
    seed = args.seed
    task = args.Task
    
    # Setup Logging
    log_dir = os.path.join(args.output_dir, "classification_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"VisionFM_{task}_{seed}.log")
    
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
            
            targets, preds, img_paths = [], [], []
            all_preds_dicts = []
            
            for inp, target, extras in metric_logger.log_every(val_loader, 20, header):
                inp = inp.cuda(non_blocking=True)
                batch_size = inp.shape[0]
                target = target.cuda(non_blocking=True).long()
                if len(target.shape) == 2:
                    target = target.squeeze()

                # Feature Extraction
                intermediate_output = model.get_intermediate_layers(inp, n_blocks)
                if avg_pool == 0:
                    output = [x[:, 0] for x in intermediate_output]
                elif avg_pool == 1:
                    output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
                output = torch.cat(output, dim=-1)
                
                # Classification
                output = linear_classifier(output)
                num_class = output.shape[1]
                
                if num_class > 1:
                    loss = nn.CrossEntropyLoss()(output, target)
                    preds.append(output.softmax(dim=1).detach().cpu())
                    targets.append(target.detach().cpu())
                else:
                    loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())
                    targets.append(target.detach().cpu())
                    preds.append(output.detach().cpu().sigmoid())
                
                img_paths += extras['img_path']
                metric_logger.update(loss=loss.item())
                
                # Store predictions
                output_softmax = torch.nn.functional.softmax(output, dim=1)
                for batch_idx in range(batch_size):
                    preds_dict = {
                        'preds': [output_softmax[batch_idx].detach().cpu().numpy()],
                        'labels': target[batch_idx].cpu().numpy(),
                        'img_path': extras['img_path'][batch_idx]
                    }
                    all_preds_dicts.append(preds_dict)
            
            # --- Save Pickle ---
            pred_dir = os.path.join(args.output_dir, "Final_prediction")
            os.makedirs(pred_dir, exist_ok=True)
            final_pickle_path = os.path.join(pred_dir, f'VisionFM_pred_{task}_{seed}.pickle')
            with open(final_pickle_path, 'wb') as file:
                pickle.dump(all_preds_dicts, file)
            
            # --- Metrics Calculation ---
            preds_all = torch.cat(preds, dim=0) 
            targets_all = torch.cat(targets, dim=0)
            
            # Confusion Matrix
            confusion_matrix_figure(preds_all, targets_all, task, seed, args.output_dir)
            
            # Accuracy
            acc1, = utils.accuracy(preds_all, targets_all, topk=(1,))
            acc1 = acc1.item()
            
            # AUC
            num_class = preds_all.shape[1]
            target_onehot = torch.nn.functional.one_hot(targets_all, num_class).float()
            
            auc_list = compute_roc_auc2(preds_all, target_onehot, average="none")
            mean_auc, ci = bootstrap_auc_ci(preds_all, target_onehot, average="macro")
            print(f"AUC List: {auc_list}")
            print(f"Mean AUC: {mean_auc:.4f}")
            print(f"95% Confidence Interval: {ci}")
            
            # Precision, Recall, F1
            # Note: average=None returns metrics for each class. 
            # We average [1:] assuming class 0 is normal/background if that is the intent,
            # otherwise adjust logic to average all classes.
            pcf = precision_recall_fscore_support(targets_all.numpy(), preds_all.argmax(dim=1).numpy(), average=None)
            precision = pcf[0][1:].mean()
            recall = pcf[1][1:].mean()
            f1 = pcf[2][1:].mean()
            
            # Per-class Accuracy
            predictions = torch.argmax(preds_all, dim=1)
            cm = confusion_matrix(targets_all.numpy(), predictions.numpy())
            
            # Sensitivity & Specificity
            sensitivities = []
            specificities = []
            
            print("\n【Per-class Performance】")
            for i in range(num_class):
                TP = cm[i, i]
                FN = cm[i, :].sum() - TP
                FP = cm[:, i].sum() - TP
                TN = cm.sum() - (TP + FN + FP)
                
                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                sensitivities.append(sensitivity)
                specificities.append(specificity)
                
                print(f"Class {i}:")
                print(f"├─ Sensitivity: {sensitivity:.4f}")
                print(f"└─ Specificity: {specificity:.4f}")

            print(f"\n* Macro Avg Sensitivity: {np.mean(sensitivities):.4f}")
            print(f"* Macro Avg Specificity: {np.mean(specificities):.4f}")
            
            # Update meters
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

            # Save JSON predictions
            if dst_json_path is not None:
                save_preds(targets, preds, img_paths, dst_json_path)
                
        finally:
            sys.stdout = original_stdout

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Setup Output Directory
    args.output_dir = os.path.join(args.output_dir, args.name)
    if not os.path.exists(args.output_dir):
        print(f"Creating output_dir: {args.output_dir}")
        os.makedirs(args.output_dir)

    utils.fix_random_seeds(args.seed)

    # --- Data Preparation ---
    mean, std = utils.get_stats(args.modality)
    print(f"Using {args.modality} stats: Mean={mean}, Std={std}")

    train_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
        self_transforms.RandomHorizontalFlip(),
        self_transforms.RandomVerticalFlip(),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    val_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    if args.dataset_format == 'vfm':
        dataset_train = ClsImgs(root=args.data_path, split='training', transform=train_transform)
        dataset_val = ClsImgs(root=args.data_path, split='test', transform=val_transform) 
    elif args.dataset_format == 'ImageNet':
        dir_train = os.path.join(args.data_path, 'train')
        dataset_train = ImageFolderDataset(dir_train, train_transform)
        dir_val = os.path.join(args.data_path, 'test')
        dataset_val = ImageFolderDataset(dir_val, val_transform)
    else:
        raise NotImplementedError("Unknown dataset format")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    
    if sys.gettrace():
        print("Debug mode detected: Setting num_workers to 0")
        args.num_workers = 0
        
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    print(f"Data loaded: {len(dataset_train)} train images, {len(dataset_val)} val images.")

    # --- Build Network ---
    model = models.__dict__[args.arch](
        img_size=[args.input_size],
        patch_size=args.patch_size,
        num_classes=0,
        use_mean_pooling=False
    )
    embed_dim = model.embed_dim
    model.cuda()
    print(f"Model {args.arch} ({args.patch_size}x{args.patch_size}) built.")
    
    # Load Pretrained Backbone
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    # Build Decoder (Head)
    if args.avgpool_patchtokens == 0:
        # Concatenate CLS tokens from last n blocks
        input_dim = embed_dim * args.n_last_blocks
        linear_classifier = ClsHead(embed_dim=input_dim, num_classes=args.num_labels, layers=1)
    elif args.avgpool_patchtokens == 1:
        # Average pool patch tokens
        linear_classifier = ClsHead(embed_dim=embed_dim, num_classes=args.num_labels, layers=3)
    else:
        raise NotImplementedError("Check avgpool_patchtokens setting.")

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

    # Resume Training
    to_restore = {"epoch": 0, "best_f1": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(
            args.load_from,
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler
        )

    # Inference Only Mode
    if args.load_from and args.test:
        print("Running in Inference Mode...")
        dst_json_path = os.path.join(args.output_dir, f"{args.name}_{args.modality}_results.json")
        model.eval()
        linear_classifier.eval()
        
        # Re-use validate_network logic, but for test set if needed
        # Note: Code implies dataset_val is used for testing in standard flow
        validate_network(val_loader, 0, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args, dst_json_path)
        return

    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_f1"]

    # --- Main Loop ---
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.eval() # Backbone remains frozen/eval mode usually
        linear_classifier.train()
        
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_classifier.eval()
            test_stats = validate_network(val_loader, epoch, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args)
            
            # Aggregate stats for display
            # (Note: test_stats already contains averages, but if list structure returned, handle here)
            print(f"Epoch {epoch} Results:")
            print(f"Acc: {test_stats.get('acc1', 0):.4f}, AUC: {test_stats.get('auc', 0):.4f}, F1: {test_stats.get('f1', 0):.4f}")

            log_stats = {**{k: v for k, v in log_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}}

            # Save Best Model
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
    parser = argparse.ArgumentParser('VisionFM Classification Decoder Training')
    
    # Experiment Config
    parser.add_argument('--name', type=str, default='visionfm_experiment', help='Name of the experiment')
    parser.add_argument('--output_dir', default="./results", help='Path to save logs and checkpoints')
    parser.add_argument('--seed', default=1000, choices=[0, 32, 64, 128, 256, 1000], type=int, help='Random seed')
    
    # Model Config
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'], help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution')
    parser.add_argument('--pretrained_weights', default='./VFM_Fundus_weights.pth', type=str, help='Path to backbone weights')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key in checkpoint (e.g., teacher)')
    
    # Decoder Config
    parser.add_argument('--n_last_blocks', default=4, type=int, help="Concatenate [CLS] tokens for the `n` last blocks")
    parser.add_argument('--avgpool_patchtokens', default=1, choices=[0, 1, 2], type=int, help='Pooling strategy')
    parser.add_argument('--num_labels', default=4, type=int, help='Number of classes')

    # Data Config
    parser.add_argument('--data_path', default='/home/ps/data/jinjingzhu/VisionFM/dataset/fundus', type=str, help='Path to dataset')
    parser.add_argument('--dataset_format', default='vfm', choices=['vfm', 'ImageNet'], type=str)
    parser.add_argument('--modality', default='Fundus', type=str)
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--Task', default='E', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G'], type=str, help='Task ID')
    
    # Training Config
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Batch size per GPU')
    parser.add_argument('--num_workers', default=10, type=int, help='Data loading workers')
    parser.add_argument('--val_freq', default=1, type=int, help="Validation frequency")
    parser.add_argument('--load_from', default=None, help='Path to resume checkpoint')
    parser.add_argument('--test', action='store_true', help='Inference only')

    # Distributed Config
    parser.add_argument("--dist_url", default="env://", type=str, help="Distributed training URL")
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank")
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')

    args = parser.parse_args()

    # Create Output Directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run for each checkpoint key
    for checkpoint_key in args.checkpoint_key.split(','):
        print(f"Starting evaluation for key: {checkpoint_key}")
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        main(args_copy)