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
import torch.nn.functional as F
import numpy as np
from models.head import ClsHead
from collections import defaultdict
from pathlib import Path
from torch import nn
import os
import contextlib
from datetime import datetime
import time
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision import datasets
import transforms as self_transforms
from dataset import ClsImgs, ImageFolderDataset
from sklearn.metrics import precision_recall_fscore_support
from monai.metrics import compute_roc_auc
from evaluation_funcs import performance_single_cls
import numpy as np
from typing import Union, List
from sklearn.utils import resample
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageDraw
import random
    
def confusion_matrix_figure(prediction,target):
    pred_labels = torch.argmax(prediction,1).cpu().numpy()
    true_labels = target.cpu().numpy()
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["HD", "BJ", "XN", "YS"]  # 替换为你的实际类别名称
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(
        ax=ax,
        cmap=plt.cm.Purples,
        text_kw={'fontsize': 16}, # 调整矩阵内部数字大小，ßß
        colorbar=False  
    )
    for i in range(4):
        for j in range(4):
            text = ax.texts[i * 4 + j]
            new_text = f"{cm_normalized[i,j]:.2f}\n(n={cm[i,j]})"
            text.set_text(new_text)
            if i == j:
                text.set_color('white')
    ax.set_xlabel('Predicted label', fontsize=16)  # X轴标签ß
    ax.set_ylabel('True label', fontsize=16)       # Y轴标签
    ax.tick_params(axis='both', which='major', labelsize=14)  # 刻度标签
    
    plt.savefig("../VisionFM/domain_confusion_matrix/RETFound_newest.png", dpi=600, bbox_inches='tight')
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

class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logits = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        
        p_t = torch.exp(logits) * targets_one_hot
        p_t = p_t.sum(dim=1)
        
        loss = -self.alpha * (1 - p_t) ** self.gamma * logits.gather(1, targets.unsqueeze(1))
        
        if self.weight is not None:
            loss = loss * self.weight[targets]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    cudnn.benchmark = True

    args.output_dir = os.path.join(args.output_dir, args.name)
    if not os.path.exists(args.output_dir):
        print(f"Create the output_dir: {args.output_dir}")
        os.makedirs(args.output_dir)

    utils.fix_random_seeds(args.seed)

    with open("../VisionFM/Final_feature/RETFound_training_Domain.pickle", 'rb') as file:
        train_feats = pickle.load(file)
    with open("../VisionFM/Final_feature/RETFound_test_Domain.pickle", 'rb') as file:
        val_feats = pickle.load(file)
    train_path = '../VisionFM/dataset/fundus/training/training_labels.txt'
    test_path = '../VisionFM/dataset/fundus/test/test_labels.txt'

    train_feats_data = [feat['feats'] for feat in train_feats]
    train_labels_data = [feat['labels'] for feat in train_feats]
    train_path_data = [feat['img_path'] for feat in train_feats]
    with open(train_path, 'r') as file:
        txt_img_paths = file.readlines()
        txt_img_paths = [path.strip().split(';')[0] for path in txt_img_paths]
    selected_train_feats_data = []
    selected_train_feats_labels = []
    for feat, label in zip(train_feats, train_labels_data):
        img_path = feat['img_path']#.replace("/test/", "/A/")
        if img_path in txt_img_paths:
            selected_train_feats_data.append(feat['feats'])
            selected_train_feats_labels.append(label)
    print(selected_train_feats_labels)
    selected_train_feats_data = torch.tensor(selected_train_feats_data)
    selected_train_feats_labels = torch.tensor(selected_train_feats_labels)
    print("train_data:", selected_train_feats_data.shape)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(selected_train_feats_data), torch.tensor(selected_train_feats_labels))

    val_feats_data = [feat['feats'] for feat in val_feats]
    val_labels_data = [feat['labels'] for feat in val_feats]
    with open(test_path, 'r') as file:
        txt_img_paths = file.readlines()
        txt_img_paths = [path.strip().split(';')[0] for path in txt_img_paths]
    selected_val_feats_data = []
    selected_val_feats_labels = []
    img_path_to_feat = {feat['img_path']: feat['feats'] for feat in val_feats}
    img_path_to_label = {feat['img_path']: label for feat, label in zip(val_feats, val_labels_data)}

    for img_path in txt_img_paths:
        img_path = img_path#.replace("/C/", "/test/")
        if img_path in img_path_to_feat:  # 确保路径存在于 val_feats 中
            selected_val_feats_data.append(img_path_to_feat[img_path])
            selected_val_feats_labels.append(img_path_to_label[img_path])
    print(len(selected_val_feats_labels))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(selected_val_feats_data), torch.tensor(selected_val_feats_labels))

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

    embed_dim = 768
    if args.avgpool_patchtokens == 0:
        linear_classifier = ClsHead(embed_dim=1024, num_classes=args.num_labels, layers=1)
    elif args.avgpool_patchtokens == 1:
        linear_classifier = ClsHead(embed_dim=embed_dim, num_classes=args.num_labels, layers=3)

    print(utils.get_parameter_number(linear_classifier))
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    parameters = linear_classifier.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        betas=(0.9, 0.999), weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

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

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        linear_classifier.train()
        train_stats = train(linear_classifier, optimizer, train_loader, epoch)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            linear_classifier.eval()
            test_stats = validate_network(val_loader, linear_classifier,epoch)

            all_acc = []
            all_auc = []
            all_f1 = []
            all_precision = []
            all_recall = []
            for key, val in test_stats.items():
                if 'acc' in key:
                    all_acc.append(val)
                elif 'auc' in key:
                    all_auc.append(val)
                elif 'f1' in key:
                    all_f1.append(val)
                elif 'precision' in key:
                    all_precision.append(val)
                elif 'recall' in key:
                    all_recall.append(val)
            all_acc = np.asarray(all_acc).mean()
            all_auc = np.asarray(all_auc).mean()
            all_f1 = np.asarray(all_f1).mean()
            all_precision = np.asarray(all_precision).mean()
            all_recall = np.asarray(all_recall).mean()

            print(f"Mean acc at epoch {epoch} of the network on the {len(val_dataset)} test images: {all_acc:.4f}")
            print(f"Mean auc at epoch {epoch} of the network on the {len(val_dataset)} test images: {all_auc:.4f}")
            print(f"Mean F1 at epoch {epoch} of the network on the {len(val_dataset)} test images: {all_f1:.4f}")
            print(f"Mean Precision at epoch {epoch} of the network on the {len(val_dataset)} test images: {all_precision:.4f}")
            print(f"Mean Recall at epoch {epoch} of the network on the {len(val_dataset)} test images: {all_recall:.4f}")

            log_stats = {**{k: v for k, v in log_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process() and (test_stats["f1"] >= best_f1):
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": test_stats["f1"],
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(args.checkpoint_key)))

            best_f1 = max(best_f1, test_stats["f1"])
            print(f'Max F1 so far: {best_f1:.4f}')
    print("Training of the supervised linear classifier on frozen features completed.\nAnd the best F1-score: {f1:.4f}".format(f1=best_f1))


def train(linear_classifier, optimizer, loader, epoch):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if epoch==19: 
        all_preds=[]
        for (inp, target) in metric_logger.log_every(loader, 20, header):
            preds_dict = {}
            inp = inp.cuda(non_blocking=True)
            batch_size = inp.shape[0]
            inp = inp.view(inp.shape[0],-1)
            target = target.cuda(non_blocking=True).long()
            if len(target.shape) == 2:
                target = target.squeeze()
            output = linear_classifier(inp)
            loss = nn.CrossEntropyLoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_list =[torch.nn.functional.softmax(output,dim=1)]
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            for batch_idx in range(batch_size):
                preds_dict = {
                    'preds': [pred[batch_idx].detach().cpu().numpy() for pred in output_list],
                    'labels': target[batch_idx].cpu().numpy()
                }
                all_preds.append(preds_dict)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    else:
        for (inp, target) in metric_logger.log_every(loader, 20, header):
            inp = inp.cuda(non_blocking=True)
            inp = inp.view(inp.shape[0],-1)
            target = target.cuda(non_blocking=True).long()
            if len(target.shape) == 2:
                target = target.squeeze()
            output = linear_classifier(inp)
            loss = nn.CrossEntropyLoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def binary2multi(input_tensor):
    if len(input_tensor.shape) == 2 and input_tensor.shape[1] == 1:
        return torch.cat([1.0 - input_tensor, input_tensor], dim=1)
    elif len(input_tensor.shape) == 1:
        return torch.cat([1.0 - input_tensor.unsqueeze(dim=1), input_tensor.unsqueeze(dim=1)], dim=1)
    else:
        raise NotImplementedError
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

def validate_network(loader, linear_classifier,epoch):
    log_file = "../VisionFM/domain_logs/RETFound_4_domain.log"
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n\n{'='*40}\n")
        f.write(f"执行时间: {timestamp}\n")
        
        original_stdout = sys.stdout
        sys.stdout = DualOutput(f, original_stdout)
        
        try:
            print(f"\n{'='*30} Epoch {epoch} {'='*30}")
            all_preds=[]
            linear_classifier.eval()
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Test:'
            targets, preds = [], []
            all_preds=[]
            for inp, target in metric_logger.log_every(loader, 20, header):
                preds_dict = {}
                inp = inp.cuda(non_blocking=True)
                batch_size = inp.shape[0]
                inp = inp.view(inp.shape[0],-1)
                target = target.cuda(non_blocking=True).long()
                if len(target.shape) == 2:
                    target = target.squeeze()
                
                output = linear_classifier(inp)
                num_class = output.shape[1]
                if num_class > 1:
                    loss = nn.CrossEntropyLoss()(output, target)
                else:
                    loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())

                if num_class > 1:
                    preds.append(output.softmax(dim=1).detach().cpu())
                    targets.append(target.detach().cpu())
                else:
                    targets.append(target.detach().cpu())
                    preds.append(output.detach().cpu().sigmoid())

                metric_logger.update(loss=loss.item())
                output_list =[torch.nn.functional.softmax(output,dim=1)]
                num_class = output.shape[1]
                for batch_idx in range(batch_size):
                    preds_dict = {
                        'preds': [pred[batch_idx].detach().cpu().numpy() for pred in output_list],
                        'labels': target[batch_idx].cpu().numpy()
                    }
                    all_preds.append(preds_dict)
            preds_all = torch.cat(preds, dim=0) # torch.size([1497,4])
            targets_all = torch.cat(targets, dim=0)
            confusion_matrix_figure(preds_all,targets_all)
            num_class = preds_all.shape[1]
            acc1, = utils.accuracy(preds_all, targets_all, topk=(1,))
            acc1 = acc1.item()
            target_onehot = torch.nn.functional.one_hot(targets_all, num_class).float()
            auc_dr_grading_list = compute_roc_auc(preds_all, target_onehot, average="none")
            print(auc_dr_grading_list)
            auc_dr_grading_list2 = compute_roc_auc2(preds_all, target_onehot, average="none")
            print(np.array(auc_dr_grading_list2).mean())
            mean_auc, ci = bootstrap_auc_ci(preds_all, target_onehot, average="macro")
            print(f"Mean AUC: {mean_auc}")
            print(f"95% Confidence Interval: {ci}")
            auc = np.array(auc_dr_grading_list).mean()
            pcf = precision_recall_fscore_support(targets_all.numpy(), preds_all.argmax(dim=1).numpy(), average=None)
            precision = pcf[0][1:].mean()
            recall = pcf[1][1:].mean()
            f1 = pcf[2][1:].mean()
            class_correct = list(0. for i in range(num_class))
            class_total = list(0. for i in range(num_class))
            predictions = torch.argmax(preds_all, dim=1)
            correct = (predictions == targets_all).squeeze()
            for i in range(len(targets_all)):
                label = targets_all[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            class_accuracy = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in range(num_class)]
            print("每个类别的准确率:", class_accuracy)
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

            batch_size = 1
            metric_logger.meters['acc1'].update(acc1, n=batch_size)
            metric_logger.meters['auc'].update(auc, n=batch_size)
            metric_logger.meters['precision'].update(precision.item(), n=batch_size)
            metric_logger.meters['recall'].update(recall.item(), n=batch_size)
            metric_logger.meters['f1'].update(f1.item(), n=batch_size)

            print(
                '* Acc@1 {top1.global_avg:.4f} loss {losses.global_avg:.4f} AUC {auc.global_avg:.4f} Pre {precision.global_avg:.4f} Recall {recall.global_avg:.4f} F1 {f1.global_avg:.4f}'
                .format(top1=metric_logger.acc1, losses=metric_logger.loss, auc=metric_logger.auc,
                        precision=metric_logger.precision, recall=metric_logger.recall, f1=metric_logger.f1))
        finally:
            sys.stdout = original_stdout

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of training""")
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Per-GPU batch-size')

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=1, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='../VisionFM/dataset/fundus', type=str, help='Please specify path to the dataset.')
    parser.add_argument('--dataset_format', default='vfm',choices=['vfm', 'ImageNet'], type=str, help='Please specify path to the dataset.')
    parser.add_argument('--seed', default=0, type=int)
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