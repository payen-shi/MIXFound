import sys
sys.path.append('./')
import os
import argparse
import copy
import torch
import pickle
import torch.backends.cudnn as cudnn
import utils
import models
import transforms as self_transforms
from evaluation.dataset import ClsImgs, SegImgs
from RETFound import models_vit

def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model


def extract_feats(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.fix_random_seeds(args.seed)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    print(f"Use the {args.modality} mean and std")
    train_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size)),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    val_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size)),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    dataset_train = ClsImgs(root=args.data_path, dst_root=args.dst_root,  split='training', transform=train_transform)
    dataset_val = ClsImgs(root=args.data_path, dst_root=args.dst_root, split='test', transform=val_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
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
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    chkpt_dir = '../VisionFM/RETFound/RETFound_cfp_weights.pth'
    model = prepare_model(chkpt_dir, 'vit_large_patch16')
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    start_epoch = 0
    existing_feats=[]
    for epoch in range(start_epoch, args.epochs): 
        train_loader.sampler.set_epoch(epoch)
        model.eval()
        all_feats = extract_features(model, train_loader, epoch)
        existing_feats.extend(all_feats)
    print(len(existing_feats),"hahhah")
    final_pickle_path = '../VisionFM/Final_feature/RETFound_E_new.pickle'
    with open(final_pickle_path, 'wb') as file:
        pickle.dump(existing_feats, file)
    print(f"所有批次的特征已保存到 {final_pickle_path}")
    

@torch.no_grad()
def extract_features(model, data_loader, epoch,):
    metric_logger = utils.MetricLogger(delimiter="  ")
    all_feats = []
    for samples, labs, extras in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        feats_dict = {}
        batch_size = samples.shape[0] #64,3,224,224
        
        
        intermediate_output = model.forward_features(samples) # features of last n blocks in vit

        output = [intermediate_output]
        img_paths = extras['img_path']
        for batch_idx in range(batch_size):
            feats_dict = {
                'feats': [feat[batch_idx].cpu().numpy() for feat in output],
                'labels': labs[batch_idx].numpy(),
                'img_path': img_paths[batch_idx]
            }
            all_feats.append(feats_dict)
    return all_feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
                                                                          'vit_large', 'vit_huge'], help='Architecture.')
    parser.add_argument('--input_size', type=int, default=224, help='input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='../VisionFM/VFM_Fundus_weights.pth', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs of training.')

    parser.add_argument('--data_path', default='../VisionFM/dataset/fundus', type=str, help='Please specify path to the dataset.')
    parser.add_argument('--dst_root', type=str, help='The root dir to save the extracted features')
    parser.add_argument('--modality', type=str, default='Fundus', choices=['Fundus', 'OCT', 'FFA', 'SlitLamp', 'UltraSound', 'MRI', 'External', 'UBM'], help='the modality of the dataset')
    parser.add_argument('--mode', type=str, default='cls', choices=['cls', 'seg']) # the extraction for classification and segmentation task

    args = parser.parse_args()

    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        extract_feats(args_copy)