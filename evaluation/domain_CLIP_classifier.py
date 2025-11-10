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
import pickle
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
import torch
import cv2
import argparse
import json
import copy
import torch
import pickle
import torch.backends.cudnn as cudnn
import utils
import models
import pickle
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
import clip
import torch
import numpy as np
from typing import Union, List
from sklearn.utils import resample
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

class SavePILImage:
    """在指定步骤保存PIL格式图像的辅助类"""
    def __init__(self, save_dir, prefix="processed"):
        self.save_dir = save_dir
        self.prefix = prefix
        self.counter = 0  # 用于生成唯一文件名
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, img):
        filename = os.path.join(
            self.save_dir, 
            f"{self.prefix}_{self.counter:04d}.jpg"
        )
        img.save(filename)
        self.counter += 1
        return img
    
def apply_frequency_filter(img_gray, mask):
    """频域滤波核心函数（复用）"""
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mask_3d = np.stack([mask]*2, axis=-1)
    fshift = dft_shift * mask_3d
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

class SpatialLowPass:
    """空域低通滤波（高斯模糊）"""
    def __init__(self, kernel_size=15, apply_prob=0.5):
        self.kernel_size = kernel_size
        self.apply_prob = apply_prob  # 应用该滤波的概率

    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
            
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        filtered = cv2.GaussianBlur(img_cv, (self.kernel_size, self.kernel_size), 0)
        
        filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

class SpatialHighPass:
    """空域高通滤波（边缘增强）"""
    def __init__(self, kernel_size=3, apply_prob=0.5):
        self.kernel_size = kernel_size
        self.apply_prob = apply_prob

    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
            
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=self.kernel_size)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        sharpened = cv2.addWeighted(gray, 1.5, laplacian_abs, -0.5, 0)
        
        sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        filtered_rgb = cv2.cvtColor(sharpened_rgb, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

class FrequencyLowPass:
    """频域低通滤波（模糊）"""
    def __init__(self, cutoff=30, apply_prob=0.5):
        self.cutoff = cutoff
        self.apply_prob = apply_prob

    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
            
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-self.cutoff:crow+self.cutoff, ccol-self.cutoff:ccol+self.cutoff] = 1
        
        filtered = apply_frequency_filter(gray, mask)
        
        filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        filtered_rgb = cv2.cvtColor(filtered_rgb, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

class FrequencyHighPass:
    """频域高通滤波（边缘提取）"""
    def __init__(self, cutoff=30, apply_prob=0.5):
        self.cutoff = cutoff
        self.apply_prob = apply_prob

    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
            
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-self.cutoff:crow+self.cutoff, ccol-self.cutoff:ccol+self.cutoff] = 0
        
        filtered = apply_frequency_filter(gray, mask)
        filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        filtered_rgb = cv2.cvtColor(filtered_rgb, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

class CircleCrop:
    def __init__(self, threshold=30, save_dir="./outputs"):
        self.threshold = threshold
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # 自动创建输出目录
        
    def _generate_filename(self):
        """生成唯一文件名：时间戳+随机数"""
        return f"{int(time.time())}_{np.random.randint(1000)}.jpg"

    def __call__(self, img):
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        mid_row = img_bgr[h//2, :, :]
        non_bg = np.where(np.any(mid_row > self.threshold, axis=1))[0]
        
        if len(non_bg) < 2:
            self._save_image(img, "original")  # 保存原始图像用于调试
            return img

        x1, x2 = non_bg[[0, -1]]
        diameter = x2 - x1
        radius = diameter // 2
        center = (w//2, h//2)
        
        y_start = max(0, center[1]-radius)
        y_end = min(h, center[1]+radius)
        x_start = max(0, center[0]-radius)
        x_end = min(w, center[0]+radius)
        cropped = img_bgr[y_start:y_end, x_start:x_end]

        square = np.zeros((diameter, diameter, 3), dtype=np.uint8)
        y_off = (diameter - cropped.shape[0]) // 2
        x_off = (diameter - cropped.shape[1]) // 2
        square[y_off:y_off+cropped.shape[0], x_off:x_off+cropped.shape[1]] = cropped

        pil_img = Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
        return pil_img

    def _save_image(self, img, prefix):
        """通用保存方法"""
        filename = os.path.join(self.save_dir, f"{prefix}_{self._generate_filename()}")
        img.save(filename)

def random_center_crop(image):
    scale = 0.95*0.02 # random.uniform(0.5, 1.0)
    width, height = image.size
    
    crop_diameter = int(min(width, height) * scale)
    radius = crop_diameter // 2  # 实际裁剪的圆形半径
    
    crop = transforms.CenterCrop(crop_diameter)
    cropped_square = crop(image)
    
    mask = Image.new('L', (crop_diameter, crop_diameter), 0)  # 全黑遮罩
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, crop_diameter, crop_diameter), fill=255)  # 绘制白色圆形
    
    circular_image = Image.new('RGB', (crop_diameter, crop_diameter), (0, 0, 0))
    circular_image.paste(cropped_square, (0, 0), mask=mask)
    
    background = Image.new('RGB', (width, height), (0, 0, 0))
    
    offset = ((width - crop_diameter) // 2, (height - crop_diameter) // 2)
    background.paste(circular_image, offset)
    
    background.save("output.jpg", "JPEG")
    return background
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
    
    plt.savefig("../VisionFM/domain_confusion_matrix/CLIP_newes_final.png", dpi=600, bbox_inches='tight')
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

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    args.output_dir = os.path.join(args.output_dir, args.name) # set new output dir with name
    if not os.path.exists(args.output_dir):
        print(f"Create the output_dir: {args.output_dir}")
        os.makedirs(args.output_dir)

    utils.fix_random_seeds(args.seed)

    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")

    train_transform = self_transforms.Compose([
        CircleCrop(threshold=30, save_dir="./processed_images"),
        self_transforms.Resize(size=(args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(random_center_crop),  # 每次随机裁剪不同比例
        
        self_transforms.RandomHorizontalFlip(),
        self_transforms.RandomVerticalFlip(),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    val_transform = self_transforms.Compose([
        CircleCrop(threshold=30, save_dir="./processed_images"),
        self_transforms.Resize(size=(args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(random_center_crop),  # 每次随机裁剪不同比例
        
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])


    if args.dataset_format == 'vfm':
        dataset_train = ClsImgs(root=args.data_path, split='training', transform=train_transform)
        dataset_val = ClsImgs(root=args.data_path, split='test', transform=val_transform) # set split='val' if there are val set in datasets
    elif args.dataset_format == 'ImageNet':
        dir_train = os.path.join(args.data_path, 'train')
        dataset_train = ImageFolderDataset(dir_train, train_transform)
        dir_val = os.path.join(args.data_path, 'test')
        dataset_val = ImageFolderDataset(dir_val, val_transform)
    else:
        raise NotImplementedError

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    if sys.gettrace():
        print(f"Now in debug mode")
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
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")


    linear_classifier = ClsHead(embed_dim=768, num_classes=args.num_labels, layers=1)

    print(utils.get_parameter_number(linear_classifier)) # compute the parameters
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    parameters = linear_classifier.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        betas=(0.9, 0.999), weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    to_restore = {"epoch": 0, "best_f1": 0.}
    if args.load_from: # load the weights to re-start training the model
        utils.restart_from_checkpoint(
            args.load_from,
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler)

    if args.load_from and args.test:
        if args.dataset_format == 'vfm':
            dataset_test = ClsImgs(root=args.data_path, split='test', transform=val_transform)
        elif args.dataset_format == 'ImageNet':
            dir_test = os.path.join(args.data_path, 'test')
            dataset_test = ImageFolderDataset(dir_test, val_transform)
        else:
            raise NotImplementedError
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dst_json_path = os.path.join(args.output_dir, f"{args.name}_{args.modality}_results.json")
        model.eval()
        linear_classifier.eval()
        test_stats = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, dst_json_path)
        print('Compute performance of single task: ')
        performance_single_cls(dst_json_path)
        exit()

    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_f1"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.eval()

        linear_classifier.train()
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_classifier.eval()
            test_stats = validate_network(val_loader, epoch, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            
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

            print(f"Mean acc at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_acc:.4f}")
            print(f"Mean auc at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_auc:.4f}")
            print(f"Mean F1 at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_f1:.4f}")
            print(f"Mean Precision at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_precision:.4f}")
            print(f"Mean Recall at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_recall:.4f}")

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
            print(f'Max F1 so far: {best_f1:.4f}%')
    print("Training of the supervised linear classifier on frozen features completed.\nAnd the best F1-score: {f1:.4f}".format(f1=best_f1))


def train(model, linear_classifier, optimizer, loader, epoch, n, avg_pool):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_feats = []
    for (inp, target, extras) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).long() # [B]
        lam = np.random.beta(0.4, 0.4)
        indices = torch.randperm(inp.size(0))          
        if len(target.shape) == 2:
            target = target.squeeze()
        
        with torch.no_grad():
            intermediate_output = model.encode_image(inp).float()
            output = intermediate_output  # only retain CLS tokens
            mixed_intermediate_output = lam * intermediate_output + (1 - lam) * intermediate_output[indices]
        mixed_target = lam * target + (1 - lam) * target[indices]
        mixed_output = linear_classifier(mixed_intermediate_output)
        output = linear_classifier(output)
        num_class = output.shape[1]
        if num_class > 1:  # for multi-class case
            loss = nn.CrossEntropyLoss()(output, target)
        else: # for binary class case
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

@torch.no_grad()
def validate_network(val_loader, epoch, model, linear_classifier, n, avg_pool, dst_json_path = None):
    log_file = "../VisionFM/domain_logs/CLIP_size_2.log"
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
                    
            linear_classifier.eval()
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Test:'
            targets, preds, img_paths = [], [], []
            all_feats = []
            all_preds=[]
            for inp, target, extras in metric_logger.log_every(val_loader, 20, header):
                
                inp = inp.cuda(non_blocking=True)
                batch_size = inp.shape[0]
                target = target.cuda(non_blocking=True).long() # [B]
                if len(target.shape) == 2:
                    target = target.squeeze()

                with torch.no_grad():
                    intermediate_output = model.encode_image(inp).float()
                    output = intermediate_output
                output = linear_classifier(output)
                num_class = output.shape[1]
                if num_class > 1:  # multi-class case
                    loss = nn.CrossEntropyLoss()(output, target)
                else:
                    loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())

                if num_class > 1:  # multi-class
                    preds.append(output.softmax(dim=1).detach().cpu())
                    targets.append(target.detach().cpu())
                else:  # num_labels=1 binary classification
                    targets.append(target.detach().cpu())  # [[B, 1]]
                    preds.append(output.detach().cpu().sigmoid())
                img_paths += extras['img_path']

                metric_logger.update(loss=loss.item())
                output_list =[torch.nn.functional.softmax(output,dim=1)]
                num_class = output.shape[1]
                for batch_idx in range(batch_size):
                    preds_dict = {
                        'preds': [pred[batch_idx].detach().cpu().numpy() for pred in output_list],
                        'labels': target[batch_idx].cpu().numpy(),
                        'img_path':extras['img_path'][batch_idx]
                    }
                    all_preds.append(preds_dict)
            preds_all = torch.cat(preds, dim=0) 
            targets_all = torch.cat(targets, dim=0)
            confusion_matrix_figure(preds_all,targets_all)
            num_class = preds_all.shape[1]
            acc1, = utils.accuracy(preds_all, targets_all, topk=(1,))
            acc1 = acc1.item()
            target_onehot = torch.nn.functional.one_hot(targets_all, num_class).float()
            auc_dr_grading_list = compute_roc_auc(preds_all, target_onehot, average="none")
            print("auc_list:",auc_dr_grading_list)
            auc_dr_grading_list2 = compute_roc_auc2(preds_all, target_onehot, average="none")
            print(np.array(auc_dr_grading_list2).mean())
            mean_auc, ci = bootstrap_auc_ci(preds_all, target_onehot, average="macro")
            print(f"Mean AUC: {mean_auc}")
            print(f"95% Confidence Interval: {ci}")
            auc = np.array(auc_dr_grading_list).mean()  # only positives
            pcf = precision_recall_fscore_support(targets_all.numpy(), preds_all.argmax(dim=1).numpy(), average=None)
            precision = pcf[0][1:].mean() # only positive class
            recall = pcf[1][1:].mean()
            f1 = pcf[2][1:].mean()
            batch_size = 1
            metric_logger.meters['acc1'].update(acc1, n=batch_size)
            metric_logger.meters['auc'].update(auc, n=batch_size)
            metric_logger.meters['precision'].update(precision.item(), n=batch_size)
            metric_logger.meters['recall'].update(recall.item(), n=batch_size)
            metric_logger.meters['f1'].update(f1.item(), n=batch_size)
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

            print(
                '* Acc@1 {top1.global_avg:.4f} loss {losses.global_avg:.4f} AUC {auc.global_avg:.4f} Pre {precision.global_avg:.4f} Recall {recall.global_avg:.4f} F1 {f1.global_avg:.4f}'
                .format(top1=metric_logger.acc1, losses=metric_logger.loss, auc=metric_logger.auc,
                        precision=metric_logger.precision, recall=metric_logger.recall, f1=metric_logger.f1))

            if dst_json_path is not None:
                save_preds(targets, preds, img_paths, dst_json_path)
        finally:
            sys.stdout = original_stdout

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
def save_preds(targets, preds, paths:list,dst_json_path):
    results = defaultdict()

    targets_th = torch.cat(targets, dim=0) # [N]
    preds_th = torch.cat(preds, dim=0) # [N, C]
    num_class = preds_th.shape[1]
    if num_class > 1:
        targets_th = torch.nn.functional.one_hot(targets_th, num_class).float() # convert to one-hot


    num = preds_th.shape[0]
    for idx in range(num):
        img_path = paths[idx] # e.g. Sjchoi86/test/Glaucoma_025.png
        key = img_path
        if key not in results.keys():
            if len(preds_th[idx]) > 1:
                results[key] = {'gt': targets_th[idx].tolist(),
                                'pred': preds_th[idx].tolist()}
            else:
                results[key] = {'gt': [targets_th[idx].item()],
                            'pred':[preds_th[idx].item()]}

    print(f"there are {len(results.keys())} images in test set")
    with open(dst_json_path, "w") as f:
        f.write(json.dumps(results, indent=4))
    print(f"write {dst_json_path} success. ")


def binary2multi(input_tensor):
    if len(input_tensor.shape) == 2 and input_tensor.shape[1] == 1:
        return torch.cat([1.0 - input_tensor, input_tensor], dim=1)
    elif len(input_tensor.shape) == 1:
        return torch.cat([1.0 - input_tensor.unsqueeze(dim=1), input_tensor.unsqueeze(dim=1)], dim=1)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training a classification decoder on pretrained decoder')
    parser.add_argument('--name', type=str, default='single_cls_debug', help='the trial name')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. """)
    parser.add_argument('--avgpool_patchtokens', default=1, choices=[0, 1, 2], type=int)
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