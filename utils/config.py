import argparse
import json
from pathlib import Path


DEFAULT_CROPFORMER_ROOT = Path("third_party/Entity/Entityv2/CropFormer")
DEFAULT_CROPFORMER_CONFIG = DEFAULT_CROPFORMER_ROOT / "configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml"
DEFAULT_CROPFORMER_CHECKPOINT = Path("checkpoints/cropformer/Mask2Former_hornet_3x_576d0b.pth")

def update_args(args):
    config_path = f'configs/{args.config}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        if getattr(args, key, None) is None:
            setattr(args, key, config[key])

    if getattr(args, 'cropformer_root', None) is None:
        args.cropformer_root = str(DEFAULT_CROPFORMER_ROOT)
    if getattr(args, 'cropformer_config', None) is None:
        args.cropformer_config = str(DEFAULT_CROPFORMER_CONFIG)
    if getattr(args, 'cropformer_path', None) is None:
        args.cropformer_path = str(DEFAULT_CROPFORMER_CHECKPOINT)
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str)
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config', type=str, default='scannet')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--cropformer-root', type=str)
    parser.add_argument('--cropformer-config', type=str)
    parser.add_argument('--cropformer-path', type=str)
    parser.add_argument('--cuda-list', type=str, default='0')
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--class-agnostic-only', action="store_true")
    parser.add_argument('--skip-mask-prediction', action="store_true")
    parser.add_argument('--skip-clustering', action="store_true")
    parser.add_argument('--skip-evaluation', action="store_true")

    args = parser.parse_args()
    args = update_args(args)
    return args

def get_dataset(args):
    if args.dataset == 'scannet':
        from dataset.scannet import ScanNetDataset

        dataset = ScanNetDataset(args.seq_name)
    elif args.dataset == 'scannetpp':
        from dataset.scannetpp import ScanNetPPDataset

        dataset = ScanNetPPDataset(args.seq_name)
    elif args.dataset == 'matterport3d':
        from dataset.matterport import MatterportDataset

        dataset = MatterportDataset(args.seq_name)
    elif args.dataset == 'demo':
        from dataset.demo import DemoDataset

        dataset = DemoDataset(args.seq_name)
    else:
        print(args.dataset)
        raise NotImplementedError
    return dataset
