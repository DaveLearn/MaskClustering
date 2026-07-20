import argparse
import json
from pathlib import Path


# DEG integration: default locations for the vendored CropFormer sources and
# checkpoint. Imported by segment.py / segmenter.py. Kept here (rather than the
# per-dataset json configs) so the DEG wrapper can import them without pulling
# in the dataset modules below.
DEFAULT_CROPFORMER_ROOT = Path("third_party/Entity/Entityv2/CropFormer")
DEFAULT_CROPFORMER_CONFIG = DEFAULT_CROPFORMER_ROOT / "configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml"
DEFAULT_CROPFORMER_CHECKPOINT = Path("checkpoints/cropformer/Mask2Former_hornet_3x_576d0b.pth")

def update_args(args):
    config_path = f'configs/{args.config}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        setattr(args, key, config[key])
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str)
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config', type=str, default='scannet')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    args = update_args(args)
    return args

def get_dataset(args):
    # DEG integration: import dataset modules lazily so that importing
    # utils.config (for the CropFormer defaults above) does not require the
    # dataset dependencies, which are not installed in the DEG pixi env.
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
