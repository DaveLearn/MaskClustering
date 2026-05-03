import os
import subprocess
import time

from tqdm import tqdm

from utils.config import get_args


def run_command(command):
    return command, subprocess.call(command, shell=True)


def execute_commands(commands_list, command_type, process_num):
    print('====> Start', command_type)
    from multiprocessing import Pool

    pool = Pool(process_num)
    failed = []
    for command, returncode in tqdm(pool.imap_unordered(run_command, commands_list), total=len(commands_list)):
        if returncode != 0:
            failed.append((command, returncode))
    pool.close()
    pool.join()
    pool.terminate()

    if failed:
        command, returncode = failed[0]
        raise RuntimeError(f'{command_type} failed with exit code {returncode}: {command}')

    print('====> Finish', command_type)


def get_seq_name_list(dataset, seq_name_list_arg):
    if seq_name_list_arg:
        return [seq_name for seq_name in seq_name_list_arg.split('+') if seq_name]

    if dataset == 'scannet':
        file_path = 'splits/scannet.txt'
    elif dataset == 'scannetpp':
        file_path = 'splits/scannetpp.txt'
    elif dataset == 'matterport3d':
        file_path = 'splits/matterport3d.txt'
    else:
        raise NotImplementedError(dataset)
    with open(file_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    return seq_name_list


def parallel_compute(general_command, command_name, resource_type, cuda_list, seq_name_list):
    cuda_num = len(cuda_list)
    
    if resource_type == 'cuda':
        commands = []
        for i, cuda_id in enumerate(cuda_list):
            process_seq_name = seq_name_list[i::cuda_num]
            if len(process_seq_name) == 0:
                continue
            process_seq_name = '+'.join(process_seq_name)
            command = f'CUDA_VISIBLE_DEVICES={cuda_id} {general_command % process_seq_name}'
            commands.append(command)
        execute_commands(commands, command_name, cuda_num)
    elif resource_type == 'cpu':
        commands = []
        for seq_name in seq_name_list:
            commands.append(f'{general_command} --seq_name {seq_name}')
        execute_commands(commands, command_name, cuda_num)


def parse_cuda_list(cuda_list_str):
    cuda_ids = [item.strip() for item in cuda_list_str.split(',') if item.strip()]
    if not cuda_ids:
        raise ValueError('cuda_list must contain at least one device id')
    return cuda_ids


def validate_class_agnostic_inputs(args):
    if args.skip_mask_prediction:
        return

    required_paths = {
        'CropFormer root': args.cropformer_root,
        'CropFormer config': args.cropformer_config,
        'CropFormer checkpoint': args.cropformer_path,
    }
    for label, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'{label} not found: {path}')

def get_label_text_feature(cuda_id):
    label_text_feature_path = 'data/text_features/matterport3d.npy'
    if os.path.exists(label_text_feature_path):
        return
    command = f'CUDA_VISIBLE_DEVICES={cuda_id} python -m semantics.extract_label_featrues'
    os.system(command)

def main(args):
    dataset = args.dataset
    config = args.config
    cropformer_path = args.cropformer_path
    cuda_list = parse_cuda_list(args.cuda_list)

    if dataset == 'scannet':
        root = 'data/scannet/processed'
        image_path_pattern = 'color/*0.jpg' # stride = 10
        gt = 'data/scannet/gt'
    elif dataset == 'scannetpp':
        root = 'data/scannetpp/data'
        image_path_pattern = 'iphone/rgb/*0.jpg'
        gt = 'data/scannetpp/gt'
    elif dataset == 'matterport3d':
        root = 'data/matterport3d/scans'
        image_path_pattern = '*/undistorted_color_images/*.jpg' # stride = 1
        gt = 'data/matterport3d/gt'
    else:
        raise NotImplementedError(dataset)

    t0 = time.time()
    seq_name_list = get_seq_name_list(dataset, args.seq_name_list)
    print('There are %d scenes' % len(seq_name_list))

    validate_class_agnostic_inputs(args)
    
    # Step 1: use Cropformer to get 2D instance masks for all sequences.
    if not args.skip_mask_prediction:
        parallel_compute(
            f'python mask_predict.py --cropformer-root "{args.cropformer_root}" --config-file "{args.cropformer_config}" --root "{root}" --image_path_pattern "{image_path_pattern}" --dataset {args.dataset} --seq_name_list %s --confidence-threshold {args.confidence_threshold} --opts MODEL.WEIGHTS "{cropformer_path}"',
            'predict mask',
            'cuda',
            cuda_list,
            seq_name_list,
        )

    # # Step 2: Mask clustering using our proposed method.
    if not args.skip_clustering:
        parallel_compute(f'python main.py --config {config} --seq_name_list %s', 'mask clustering', 'cuda', cuda_list, seq_name_list)
    
    # Step 3: Evaluate the class-agnostic results.
    if not args.skip_evaluation:
        returncode = subprocess.call(
            f'python -m evaluation.evaluate --pred_path data/prediction/{config}_class_agnostic --gt_path {gt} --dataset {dataset} --no_class',
            shell=True,
        )
        if returncode != 0:
            raise RuntimeError(f'class-agnostic evaluation failed with exit code {returncode}')

    if args.class_agnostic_only:
        print('total time', (time.time() - t0)//60, 'min')
        print('Average time', (time.time() - t0) / len(seq_name_list), 'sec')
        return

    # Step 4: Get the open-vocabulary semantic features for each 2D masks.
    parallel_compute(f'python -m semantics.get_open-voc_features --config {config}  --seq_name_list %s', 'get open-vocabulary semantic features using CLIP', 'cuda', cuda_list, seq_name_list)

    # Step 5: Get the text CLIP features for each label.
    get_label_text_feature(cuda_list[0])
    
    # Step 6: Get labels for each 3D instances.
    parallel_compute(f'python -m semantics.open-voc_query --config {config}', 'get text labels', 'cpu', cuda_list, seq_name_list)
    
    # Step 7: Evaluate the class-aware results.
    returncode = subprocess.call(
        f'python -m evaluation.evaluate --pred_path data/prediction/{config} --gt_path {gt} --dataset {dataset}',
        shell=True,
    )
    if returncode != 0:
        raise RuntimeError(f'class-aware evaluation failed with exit code {returncode}')

    print('total time', (time.time() - t0)//60, 'min')
    print('Average time', (time.time() - t0) / len(seq_name_list), 'sec')

if __name__ == '__main__':
    args = get_args()
    main(args)
