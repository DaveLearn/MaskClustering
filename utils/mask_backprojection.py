import numpy as np
from pytorch3d.ops import ball_query
import torch
import open3d as o3d
from utils.geometry import denoise
from torch.nn.utils.rnn import pad_sequence

COVERAGE_THRESHOLD = 0.2
DISTANCE_THRESHOLD = 0.005 #0.03
FEW_POINTS_THRESHOLD = 25
DEPTH_TRUNC = 20
BBOX_EXPAND = 0.1


def backproject(depth, intrinisc_cam_parameters, extrinsics):
    """
    convert color and depth to view pointcloud
    """
    depth = o3d.geometry.Image(depth)
    pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinisc_cam_parameters, depth_scale=1, depth_trunc=DEPTH_TRUNC)
    pcld.transform(extrinsics)
    return pcld


def get_neighbor(valid_points, scene_points, lengths_1, lengths_2):
    _, neighbor_in_scene_pcld, _ = ball_query(valid_points, scene_points, lengths_1, lengths_2, K=20, radius=DISTANCE_THRESHOLD, return_nn=False)
    return neighbor_in_scene_pcld


def prepare_depth_for_backprojection(depth):
    finite_mask = np.isfinite(depth)
    valid_depth_mask = finite_mask & (depth > 0) & (depth < DEPTH_TRUNC)
    clipped_depth = np.where(valid_depth_mask, depth, 0).astype(np.float32)
    depth_mask = torch.from_numpy(valid_depth_mask.reshape(-1)).cuda()
    return clipped_depth, depth_mask


def crop_scene_points(mask_points, scene_points):
    x_min, x_max = torch.min(mask_points[:, 0]), torch.max(mask_points[:, 0])
    y_min, y_max = torch.min(mask_points[:, 1]), torch.max(mask_points[:, 1])
    z_min, z_max = torch.min(mask_points[:, 2]), torch.max(mask_points[:, 2])

    selected_point_mask = (
        (scene_points[:, 0] > x_min)
        & (scene_points[:, 0] < x_max)
        & (scene_points[:, 1] > y_min)
        & (scene_points[:, 1] < y_max)
        & (scene_points[:, 2] > z_min)
        & (scene_points[:, 2] < z_max)
    )
    selected_point_ids = torch.where(selected_point_mask)[0]
    cropped_scene_points = scene_points[selected_point_ids]
    return cropped_scene_points, selected_point_ids


def turn_mask_to_point(dataset, scene_points, mask_image, frame_id, debug_trace=None):
    intrinisc_cam_parameters = dataset.get_intrinsics(frame_id)
    extrinsics = dataset.get_extrinsic(frame_id)
    if np.sum(np.isinf(extrinsics)) > 0:
        return {}, [], set()

    mask_image = torch.from_numpy(mask_image).cuda().reshape(-1)
    ids = torch.unique(mask_image).cpu().numpy()
    ids.sort()
    
    depth = dataset.get_depth(frame_id)
    clipped_depth, depth_mask = prepare_depth_for_backprojection(depth)

    colored_pcld = backproject(clipped_depth, intrinisc_cam_parameters, extrinsics)
    view_points_raw = np.asarray(colored_pcld.points)
    valid_depth_count = int(depth_mask.sum().item())
    view_points = view_points_raw
    removed_non_finite_points = 0
    removed_zero_points = 0
    if len(view_points_raw) != valid_depth_count:
        finite_point_mask = np.isfinite(view_points_raw).all(axis=1)
        removed_non_finite_points = int(len(view_points_raw) - np.count_nonzero(finite_point_mask))
        view_points = view_points_raw[finite_point_mask]

        non_zero_point_mask = np.any(np.abs(view_points) > 1e-8, axis=1)
        removed_zero_points = int(len(view_points) - np.count_nonzero(non_zero_point_mask))
        view_points = view_points[non_zero_point_mask]

    if len(view_points) != valid_depth_count:
        raise RuntimeError(
            "Backprojected point count does not match valid depth mask size for "
            f"frame {frame_id}: open3d_points_raw={len(view_points_raw)} open3d_points_filtered={len(view_points)} "
            f"valid_depth_pixels={valid_depth_count} depth_trunc={DEPTH_TRUNC} "
            f"depth_ge_trunc={int(np.count_nonzero(depth >= DEPTH_TRUNC))} "
            f"depth_non_finite={int(np.count_nonzero(~np.isfinite(depth)))} "
            f"removed_non_finite_points={removed_non_finite_points} removed_zero_points={removed_zero_points}"
        )

    mask_points_list = []
    mask_points_num_list = []
    scene_points_list = []
    scene_points_num_list = []
    selected_point_ids_list = []
    initial_valid_mask_ids = []
    initial_raw_mask_point_counts = []
    initial_denoised_mask_point_counts = []
    for mask_id in ids:
        if mask_id == 0:
            continue
        mask_trace = None if debug_trace is None else debug_trace.setdefault(int(mask_id), {})
        segmentation = mask_image == mask_id
        valid_mask = segmentation[depth_mask].cpu().numpy()

        mask_pcld = o3d.geometry.PointCloud()
        mask_points = view_points[valid_mask]
        raw_mask_point_count = int(len(mask_points))
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            if mask_trace is not None:
                mask_trace["backprojection"] = "few_points_raw"
                mask_trace["raw_mask_points"] = raw_mask_point_count
            continue
        mask_pcld.points = o3d.utility.Vector3dVector(mask_points)

        mask_pcld = mask_pcld.voxel_down_sample(voxel_size=DISTANCE_THRESHOLD)
        mask_pcld, _ = denoise(mask_pcld)
        mask_points = np.asarray(mask_pcld.points)
        denoised_mask_point_count = int(len(mask_points))
        
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            if mask_trace is not None:
                mask_trace["backprojection"] = "few_points_denoised"
                mask_trace["raw_mask_points"] = raw_mask_point_count
                mask_trace["denoised_mask_points"] = denoised_mask_point_count
            continue
        
        mask_points = torch.tensor(mask_points).float().cuda()
        cropped_scene_points, selected_point_ids = crop_scene_points(mask_points, scene_points)
        initial_valid_mask_ids.append(mask_id)
        initial_raw_mask_point_counts.append(raw_mask_point_count)
        initial_denoised_mask_point_counts.append(denoised_mask_point_count)
        mask_points_list.append(mask_points)
        scene_points_list.append(cropped_scene_points)
        mask_points_num_list.append(len(mask_points))
        scene_points_num_list.append(len(cropped_scene_points))
        selected_point_ids_list.append(selected_point_ids)

    if len(initial_valid_mask_ids) == 0:
        return {}, [], []
    mask_points_tensor = pad_sequence(mask_points_list, batch_first=True, padding_value=0)
    scene_points_tensor = pad_sequence(scene_points_list, batch_first=True, padding_value=0)

    lengths_1 = torch.tensor(mask_points_num_list).cuda()
    lengths_2 = torch.tensor(scene_points_num_list).cuda()
    neighbor_in_scene_pcld = get_neighbor(mask_points_tensor, scene_points_tensor, lengths_1, lengths_2)

    valid_mask_ids = []
    mask_info = {}
    frame_point_ids = set()

    for i, mask_id in enumerate(initial_valid_mask_ids):
        mask_neighbor = neighbor_in_scene_pcld[i] # P, 20
        mask_point_num = mask_points_num_list[i] # Pi
        mask_neighbor = mask_neighbor[:mask_point_num] # Pi, 20
        raw_mask_point_count = initial_raw_mask_point_counts[i]
        denoised_mask_point_count = initial_denoised_mask_point_counts[i]

        valid_neighbor = mask_neighbor != -1 # Pi, 20
        neighbor = torch.unique(mask_neighbor[valid_neighbor])
        neighbor_in_complete_scene_points = selected_point_ids_list[i][neighbor].cpu().numpy()
        coverage = torch.any(valid_neighbor, dim=1).sum().item() / mask_point_num

        if coverage < COVERAGE_THRESHOLD:
            if debug_trace is not None:
                mask_trace = debug_trace.setdefault(int(mask_id), {})
                mask_trace["backprojection"] = "low_coverage"
                mask_trace["raw_mask_points"] = raw_mask_point_count
                mask_trace["denoised_mask_points"] = mask_point_num
                mask_trace["coverage"] = float(coverage)
                mask_trace["scene_support_points"] = int(len(neighbor_in_complete_scene_points))
            continue
        valid_mask_ids.append(mask_id)
        mask_info[mask_id] = set(neighbor_in_complete_scene_points)
        frame_point_ids.update(mask_info[mask_id])
        if debug_trace is not None:
            mask_trace = debug_trace.setdefault(int(mask_id), {})
            mask_trace["backprojection"] = "kept"
            mask_trace["coverage"] = float(coverage)
            mask_trace["scene_support_points"] = int(len(mask_info[mask_id]))
            mask_trace["raw_mask_points"] = raw_mask_point_count
            mask_trace["denoised_mask_points"] = denoised_mask_point_count

    return mask_info, valid_mask_ids, list(frame_point_ids)


def frame_backprojection(dataset, scene_points, frame_id, debug_trace=None):
    mask_image = dataset.get_segmentation(frame_id, align_with_depth=True)
    mask_info, _, frame_point_ids = turn_mask_to_point(dataset, scene_points, mask_image, frame_id, debug_trace=debug_trace)
    return mask_info, frame_point_ids
