"""MaskClustering DEG bridge module.

Adapts MaskClustering's CropFormer + view-consensus clustering pipeline to the
DEG external segmenter contract.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch

from graph.construction import mask_graph_construction
from graph.iterative_clustering import iterative_clustering
from initializerdefs import InstanceMaskObjectsDef, ObjectSegmentations, ObservationFrame, Observations, SceneSetup
from psdframe import Frame
from utils.config import DEFAULT_CROPFORMER_CHECKPOINT, DEFAULT_CROPFORMER_CONFIG, DEFAULT_CROPFORMER_ROOT
from utils.post_process import dbscan_process, filter_point, find_represent_mask, merge_overlapping_objects


logger = logging.getLogger("maskclustering-segmenter")


@dataclass
class ClusteredObject:
    point_ids: np.ndarray
    mask_list: List[Tuple[int, int, float]]
    repre_mask_list: List[Tuple[int, int, float]]


@dataclass
class TempScene:
    processed_root: Path
    scene_dir: Path
    scene_id: str
    export_to_frame: Dict[int, ObservationFrame]
    frame_name_to_export: Dict[str, int]


class TempScanNetDataset:
    def __init__(self, scene_dir: Path, scene_points: np.ndarray, image_size: Tuple[int, int]) -> None:
        self.scene_dir = scene_dir
        self.scene_points = scene_points.astype(np.float32)
        self.image_size = image_size
        self.rgb_dir = scene_dir / "color"
        self.depth_dir = scene_dir / "depth"
        self.segmentation_dir = scene_dir / "output" / "mask"
        self.extrinsics_dir = scene_dir / "pose"
        self.intrinsic_dir = scene_dir / "intrinsic"
        self.depth_scale = 1000.0

    def get_frame_list(self, step: int) -> List[int]:
        image_list = sorted(self.rgb_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        if not image_list:
            return []
        end = int(image_list[-1].stem) + 1
        return list(np.arange(0, end, step, dtype=np.int32))

    def get_intrinsics(self, frame_id: int) -> o3d.camera.PinholeCameraIntrinsic:
        intrinsic_path = self.intrinsic_dir / "intrinsic_depth.txt"
        intrinsics = np.loadtxt(intrinsic_path)
        camera = o3d.camera.PinholeCameraIntrinsic()
        camera.set_intrinsics(
            self.image_size[0],
            self.image_size[1],
            intrinsics[0, 0],
            intrinsics[1, 1],
            intrinsics[0, 2],
            intrinsics[1, 2],
        )
        return camera

    def get_extrinsic(self, frame_id: int) -> np.ndarray:
        return np.loadtxt(self.extrinsics_dir / f"{frame_id}.txt")

    def get_depth(self, frame_id: int) -> np.ndarray:
        depth = cv2.imread(str(self.depth_dir / f"{frame_id}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth image not found for frame {frame_id}")
        return (depth.astype(np.float32) / self.depth_scale).astype(np.float32)

    def get_rgb(self, frame_id: int, change_color: bool = True) -> np.ndarray:
        rgb = cv2.imread(str(self.rgb_dir / f"{frame_id}.jpg"))
        if rgb is None:
            raise FileNotFoundError(f"RGB image not found for frame {frame_id}")
        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def get_segmentation(self, frame_id: int, align_with_depth: bool = False) -> np.ndarray:
        segmentation = cv2.imread(str(self.segmentation_dir / f"{frame_id}.png"), cv2.IMREAD_UNCHANGED)
        if segmentation is None:
            raise FileNotFoundError(f"Segmentation not found for frame {frame_id}")
        if align_with_depth:
            segmentation = cv2.resize(segmentation, self.image_size, interpolation=cv2.INTER_NEAREST)
        return segmentation

    def get_scene_points(self) -> np.ndarray:
        return self.scene_points


def get_dataset_frame_from_observation_frame(observation_frame: ObservationFrame) -> Frame:
    return Frame(
        id=observation_frame.id,
        name=observation_frame.name,
        color=torch.tensor(observation_frame.color).cuda(),
        X_WV=torch.tensor(observation_frame.X_WV),
        K=torch.tensor(observation_frame.K),
        depth=(torch.tensor(observation_frame.depth).cuda() if observation_frame.depth is not None else None),
    )


def _to_cam_open3d(frame: Frame) -> o3d.camera.PinholeCameraParameters:
    intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.w, frame.h, frame.fl_x, frame.fl_y, frame.cx, frame.cy)
    extrinsic = frame.X_VW_opencv.cpu().numpy()
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic
    return camera


def _post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_to_keep: int = 1000) -> o3d.geometry.TriangleMesh:
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, _ = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) == 0:
        return mesh_0
    n_cluster = np.sort(cluster_n_triangles.copy())[-min(cluster_to_keep, len(cluster_n_triangles))]
    n_cluster = max(int(n_cluster), 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    logger.info("mesh vertices raw %d -> post %d", len(mesh.vertices), len(mesh_0.vertices))
    return mesh_0


@torch.no_grad()
def _extract_mesh_bounded(
    frames: List[Frame],
    voxel_size: float = 0.004,
    sdf_trunc: float = 0.02,
    depth_trunc: float = 3,
) -> o3d.geometry.TriangleMesh:
    logger.info(
        "TSDF integration: voxel_size=%.4f  sdf_trunc=%.4f  depth_trunc=%.2f",
        voxel_size,
        sdf_trunc,
        depth_trunc,
    )
    for frame in frames:
        assert frame.depth is not None and frame.color is not None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for frame in frames:
        rgb = frame.color.cpu().numpy()
        depth = frame.depth.cpu().numpy() if frame.depth is not None else None
        assert depth is not None

        ci = o3d.geometry.Image((rgb * 255).astype(np.uint8))
        di = o3d.geometry.Image(depth)
        cam = _to_cam_open3d(frame)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            ci,
            di,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        volume.integrate(rgbd, intrinsic=cam.intrinsic, extrinsic=cam.extrinsic)

    return volume.extract_triangle_mesh()


def _extract_mesh_bounded_with_res(frames: List[Frame], depth_trunc: float = 2, mesh_res: int = 1024) -> o3d.geometry.TriangleMesh:
    voxel_size = depth_trunc / mesh_res
    sdf_trunc = 5.0 * voxel_size
    raw_mesh = _extract_mesh_bounded(frames, voxel_size, sdf_trunc, depth_trunc)
    return _post_process_mesh(raw_mesh, cluster_to_keep=50)


def _erode_voxel_grid_xy(voxel_grid: o3d.geometry.VoxelGrid, layers: int) -> o3d.geometry.VoxelGrid:
    if layers <= 0 or not voxel_grid.has_voxels():
        return voxel_grid

    voxel_indices = [tuple(int(idx) for idx in voxel.grid_index) for voxel in voxel_grid.get_voxels()]
    xy_occupied = {(x, y) for x, y, _ in voxel_indices}

    for _ in range(layers):
        if not xy_occupied:
            break
        prev_xy = xy_occupied
        xy_occupied = {
            (x, y) for (x, y) in prev_xy if ((x - 1, y) in prev_xy and (x + 1, y) in prev_xy and (x, y - 1) in prev_xy and (x, y + 1) in prev_xy)
        }

    for voxel_index in voxel_indices:
        if (voxel_index[0], voxel_index[1]) not in xy_occupied:
            voxel_grid.remove_voxel(voxel_index)

    return voxel_grid


def get_workspace_voxels(scene: SceneSetup, shrink_xy_m: float = 0.04) -> o3d.geometry.VoxelGrid:
    table_xyz = scene.ground_gaussians.xyz
    table_plane = scene.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    table_pcd_extruded = np.array(table_xyz).copy()

    desired_height = 1.0
    below_table_height = 0.10
    voxel_size = 0.02
    iters = int(np.ceil(desired_height / voxel_size))
    for i in range(iters):
        new_points = table_xyz + table_normal * voxel_size * i
        table_pcd_extruded = np.append(table_pcd_extruded, new_points, axis=0)

    below_table_iters = int(np.ceil(below_table_height / voxel_size))
    for i in range(below_table_iters):
        table_pcd_extruded = np.append(table_pcd_extruded, table_xyz - table_normal * voxel_size * (i + 1), axis=0)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_pcd_extruded))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    layers = max(0, int(np.round(shrink_xy_m / voxel_grid.voxel_size)))
    return _erode_voxel_grid_xy(voxel_grid, layers)


def _crop_mesh_to_workspace_bbox(
    mesh: o3d.geometry.TriangleMesh,
    workspace_voxels: o3d.geometry.VoxelGrid,
    padding_m: float = 0.2,
) -> o3d.geometry.TriangleMesh:
    voxel_size = float(workspace_voxels.voxel_size)
    origin = np.asarray(workspace_voxels.origin, dtype=np.float32)
    voxels = workspace_voxels.get_voxels()
    if len(voxels) == 0:
        return mesh

    indices = np.array([v.grid_index for v in voxels], dtype=np.float32)
    min_corner = origin + indices.min(axis=0) * voxel_size - padding_m
    max_corner = origin + (indices.max(axis=0) + 1.0) * voxel_size + padding_m
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
    return mesh.crop(aabb)


def _crop_mesh_to_workspace(mesh: o3d.geometry.TriangleMesh, workspace_voxels: o3d.geometry.VoxelGrid) -> o3d.geometry.TriangleMesh:
    if not mesh.has_triangles() or not mesh.has_vertices() or not workspace_voxels.has_voxels():
        return mesh

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        return mesh

    in_workspace = np.asarray(workspace_voxels.check_if_included(o3d.utility.Vector3dVector(vertices)), dtype=bool)
    keep_triangles = in_workspace[triangles].all(axis=1)
    if keep_triangles.all():
        return mesh

    mesh.remove_triangles_by_mask(~keep_triangles)
    mesh.remove_unreferenced_vertices()
    return mesh


def _get_instance_id_mask_for_frame(instance_id: int, masks: Dict[str, np.ndarray], frame: Frame) -> torch.Tensor:
    frame_mask = masks[frame.name]
    instance_mask = frame_mask == instance_id
    return torch.tensor(instance_mask, device=frame.color.device, dtype=torch.bool)


def determine_table_instance_id(
    frames: List[Frame],
    masks: Dict[str, np.ndarray],
    table_plane: Tuple[float, float, float, float],
    object_ids: np.ndarray,
) -> int:
    if len(object_ids) == 0:
        return -1

    table_instance_candidates: List[int] = []
    table_instance_counts: List[int] = []
    for frame in frames:
        assert frame.depth is not None
        h, w = frame.depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=frame.depth.device),
            torch.arange(w, device=frame.depth.device),
            indexing="ij",
        )
        valid_mask = frame.depth > 0
        z = frame.depth
        x_world = (x - frame.cx) * z / frame.fl_x
        y_world = (y - frame.cy) * z / frame.fl_y
        points = torch.stack([x_world, y_world, z, torch.ones_like(z)], dim=0)
        points = frame.X_WV_opencv.cuda() @ points.reshape(4, -1)
        points = points.reshape(4, h, w)

        a, b, c, d = table_plane
        plane_dist = (a * points[0] + b * points[1] + c * points[2] + d) / math.sqrt(a * a + b * b + c * c)
        table_mask = torch.abs(plane_dist) < 0.02
        table_mask = table_mask & valid_mask

        for instance_id in object_ids:
            inst_id_val = int(instance_id.item()) if hasattr(instance_id, "item") else int(instance_id)
            instance_mask = _get_instance_id_mask_for_frame(inst_id_val, masks, frame)
            instance_mask_valid = instance_mask & valid_mask
            instance_mask_near_table = instance_mask & table_mask
            valid_count = instance_mask_valid.sum()
            if valid_count > 0 and instance_mask_near_table.sum() / valid_count > 0.7:
                table_instance_candidates.append(inst_id_val)
                table_instance_counts.append(instance_mask_near_table.sum().item())

    if len(table_instance_candidates) == 0:
        logger.warning("No table candidates found")
        return -1

    best_idx = int(np.argmax(table_instance_counts))
    return table_instance_candidates[best_idx]


def _sanitize_scene_id(raw_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch if ch in allowed else "_" for ch in raw_id)
    cleaned = cleaned.strip("_")
    return cleaned or "scene"


def _write_scannet_temp_dataset(frames: List[Frame], scene_id: str, mesh: o3d.geometry.TriangleMesh, work_root: Path) -> TempScene:
    processed_root = work_root / "data" / "scannet" / "processed"
    scene_dir = processed_root / scene_id
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"
    output_mask_dir = scene_dir / "output" / "mask"

    for path in (color_dir, depth_dir, pose_dir, intrinsic_dir, output_mask_dir):
        path.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise RuntimeError("No frames to export")

    k = frames[0].K.cpu().numpy()
    k4 = np.eye(4, dtype=np.float64)
    k4[:3, :3] = k
    np.savetxt(intrinsic_dir / "intrinsic_depth.txt", k4, fmt="%.8f")

    export_to_frame: Dict[int, ObservationFrame] = {}
    frame_name_to_export: Dict[str, int] = {}
    for idx, frame in enumerate(frames):
        color = (frame.color.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(color_dir / f"{idx}.jpg"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        depth = frame.depth.cpu().numpy() if frame.depth is not None else None
        if depth is None:
            raise RuntimeError("Depth is required for MaskClustering pipeline")
        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{idx}.png"), depth_mm)

        pose = frame.X_WV_opencv.cpu().numpy()
        np.savetxt(pose_dir / f"{idx}.txt", pose, fmt="%.8f")

        export_to_frame[idx] = ObservationFrame(
            id=frame.id,
            name=frame.name,
            color=frame.color.cpu().numpy(),
            X_WV=frame.X_WV.cpu().numpy(),
            K=frame.K.cpu().numpy(),
            depth=frame.depth.cpu().numpy() if frame.depth is not None else None,
        )
        frame_name_to_export[frame.name] = idx

    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    if not o3d.io.write_triangle_mesh(str(ply_path), mesh):
        raise RuntimeError(f"Failed to write mesh to {ply_path}")

    logger.info("Wrote temp ScanNet dataset to %s (%d frames)", processed_root, len(frames))
    return TempScene(
        processed_root=processed_root,
        scene_dir=scene_dir,
        scene_id=scene_id,
        export_to_frame=export_to_frame,
        frame_name_to_export=frame_name_to_export,
    )


def _run_cropformer_prediction(
    temp_scene: TempScene,
    cropformer_root: Path,
    cropformer_config: Path,
    cropformer_checkpoint: Path,
    confidence_threshold: float,
) -> None:
    project_root = Path(__file__).parent
    if not cropformer_root.exists():
        raise FileNotFoundError(f"CropFormer root not found: {cropformer_root}")
    if not cropformer_config.exists():
        raise FileNotFoundError(f"CropFormer config not found: {cropformer_config}")
    if not cropformer_checkpoint.exists():
        raise FileNotFoundError(
            f"CropFormer checkpoint not found: {cropformer_checkpoint}. Run the checkpoint download task or place the checkpoint manually."
        )

    cmd = [
        sys.executable,
        str(project_root / "mask_predict.py"),
        "--cropformer-root",
        str(cropformer_root),
        "--config-file",
        str(cropformer_config),
        "--root",
        str(temp_scene.processed_root),
        "--image_path_pattern",
        "color/*.jpg",
        "--dataset",
        "scannet",
        "--seq_name_list",
        temp_scene.scene_id,
        "--confidence-threshold",
        str(confidence_threshold),
        "--opts",
        "MODEL.WEIGHTS",
        str(cropformer_checkpoint),
    ]
    logger.info("Running CropFormer prediction: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=os.environ.copy(),
        stdout=sys.stderr,
        stderr=sys.stderr,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CropFormer prediction failed with exit code {result.returncode}")


def _build_clustering_args(
    step: int,
    mask_visible_threshold: float,
    undersegment_filter_threshold: float,
    view_consensus_threshold: float,
    contained_threshold: float,
    point_filter_threshold: float,
    debug: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        step=step,
        mask_visible_threshold=mask_visible_threshold,
        undersegment_filter_threshold=undersegment_filter_threshold,
        view_consensus_threshold=view_consensus_threshold,
        contained_threshold=contained_threshold,
        point_filter_threshold=point_filter_threshold,
        debug=debug,
    )


def _cluster_objects(dataset: TempScanNetDataset, args: SimpleNamespace) -> List[ClusteredObject]:
    scene_points = dataset.get_scene_points()
    frame_list = dataset.get_frame_list(args.step)
    if not frame_list:
        raise RuntimeError("No frames available for MaskClustering")

    nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = mask_graph_construction(args, scene_points, frame_list, dataset)
    object_list = iterative_clustering(nodes, observer_num_thresholds, args.view_consensus_threshold, args.debug)

    total_point_ids_list: List[np.ndarray] = []
    total_bbox_list: List[List[np.ndarray]] = []
    total_mask_list: List[List[Tuple[int, int, float]]] = []
    for node in object_list:
        if len(node.mask_list) < 2:
            continue

        pcld, point_ids = node.get_point_cloud(scene_points)
        pcld_list, point_ids_list = dbscan_process(pcld, point_ids)
        point_ids_list, bbox_list, mask_list = filter_point(
            point_frame_matrix,
            node,
            pcld_list,
            point_ids_list,
            mask_point_clouds,
            frame_list,
            args,
        )
        total_point_ids_list.extend(point_ids_list)
        total_bbox_list.extend(bbox_list)
        total_mask_list.extend(mask_list)

    total_point_ids_list, total_mask_list = merge_overlapping_objects(
        total_point_ids_list,
        total_bbox_list,
        total_mask_list,
        overlapping_ratio=0.8,
    )

    clustered_objects: List[ClusteredObject] = []
    for point_ids, mask_list in zip(total_point_ids_list, total_mask_list):
        clustered_objects.append(
            ClusteredObject(
                point_ids=np.asarray(point_ids, dtype=np.int32),
                mask_list=list(mask_list),
                repre_mask_list=find_represent_mask(list(mask_list)),
            )
        )

    return clustered_objects


def _save_object_dict(clustered_objects: Sequence[ClusteredObject], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    object_dict = {
        idx: {
            "point_ids": obj.point_ids,
            "mask_list": obj.mask_list,
            "repre_mask_list": obj.repre_mask_list,
        }
        for idx, obj in enumerate(clustered_objects)
    }
    np.save(output_path, object_dict, allow_pickle=True)


def _compute_workspace_pixel_mask(frame: Frame, workspace_voxels: o3d.geometry.VoxelGrid) -> np.ndarray:
    assert frame.depth is not None
    depth = frame.depth
    h, w = depth.shape
    valid_mask = depth > 0
    if not bool(valid_mask.any()):
        return np.zeros((h, w), dtype=bool)

    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device),
        torch.arange(w, device=depth.device),
        indexing="ij",
    )
    z = depth[valid_mask]
    x_world = (x[valid_mask] - frame.cx) * z / frame.fl_x
    y_world = (y[valid_mask] - frame.cy) * z / frame.fl_y
    points = torch.stack([x_world, y_world, z, torch.ones_like(z)], dim=0)
    world_points = (frame.X_WV_opencv.cuda() @ points).T[:, :3].detach().cpu().numpy()

    included = np.asarray(workspace_voxels.check_if_included(o3d.utility.Vector3dVector(world_points)), dtype=bool)
    workspace_mask = np.zeros((h, w), dtype=bool)
    workspace_mask[valid_mask.detach().cpu().numpy()] = included
    return workspace_mask


def _build_instance_groups_from_clustered_masks(
    clustered_objects: Sequence[ClusteredObject],
    dataset: TempScanNetDataset,
    export_to_frame: Dict[int, ObservationFrame],
    frames: Sequence[Frame],
    workspace_voxels: o3d.geometry.VoxelGrid,
) -> Dict[str, np.ndarray]:
    export_frame_ids = sorted(export_to_frame)
    segmentation_images = {frame_id: dataset.get_segmentation(frame_id, align_with_depth=True) for frame_id in export_frame_ids}
    workspace_masks = {frame_id: _compute_workspace_pixel_mask(frames[frame_id], workspace_voxels) for frame_id in export_frame_ids}
    result_by_export: Dict[int, np.ndarray] = {
        frame_id: np.zeros_like(segmentation_images[frame_id], dtype=np.int32) for frame_id in export_frame_ids
    }

    for frame_id in export_frame_ids:
        paint_ops: List[Tuple[float, int, int]] = []
        for object_id, obj in enumerate(clustered_objects, start=1):
            for source_frame_id, mask_id, coverage in obj.mask_list:
                if source_frame_id != frame_id:
                    continue
                paint_ops.append((float(coverage), object_id, int(mask_id)))

        for coverage, object_id, mask_id in sorted(paint_ops, key=lambda item: (-item[0], item[1], item[2])):
            del coverage
            source_mask = segmentation_images[frame_id] == mask_id
            source_mask &= workspace_masks[frame_id]
            unassigned = result_by_export[frame_id] == 0
            paint_mask = source_mask & unassigned
            result_by_export[frame_id][paint_mask] = object_id

    return {export_to_frame[frame_id].name: result_by_export[frame_id] for frame_id in export_frame_ids}


def initialize_scene(
    observations: Observations,
    scene: SceneSetup,
    intermediate_outputs_path: Optional[Path] = None,
    cropformer_root: Path = DEFAULT_CROPFORMER_ROOT,
    cropformer_config: Path = DEFAULT_CROPFORMER_CONFIG,
    cropformer_checkpoint: Path = DEFAULT_CROPFORMER_CHECKPOINT,
    confidence_threshold: float = 0.5,
    step: int = 1,
    mask_visible_threshold: float = 0.3,
    undersegment_filter_threshold: float = 0.3,
    view_consensus_threshold: float = 0.9,
    contained_threshold: float = 0.8,
    point_filter_threshold: float = 0.5,
    debug: bool = False,
) -> ObjectSegmentations:
    frames = [get_dataset_frame_from_observation_frame(frame) for frame in observations.frames]
    if not frames:
        raise ValueError("No frames in observations")

    logger.info("Reconstructing TSDF mesh from %d frames ...", len(frames))
    mesh = _extract_mesh_bounded_with_res(frames, depth_trunc=2, mesh_res=1024)

    logger.info("Building workspace voxels ...")
    workspace_voxels = get_workspace_voxels(scene)
    mesh = _crop_mesh_to_workspace_bbox(mesh, workspace_voxels)
    mesh = _crop_mesh_to_workspace(mesh, workspace_voxels)
    if not mesh.has_triangles() or len(np.asarray(mesh.triangles)) == 0:
        raise RuntimeError("Mesh is empty after workspace cropping")

    if intermediate_outputs_path is not None:
        work_root = intermediate_outputs_path / "maskclustering_work"
    else:
        work_root = Path(tempfile.mkdtemp(prefix="maskclustering_"))
    work_root.mkdir(parents=True, exist_ok=True)

    scene_id = _sanitize_scene_id(observations.id or "scene")
    temp_scene = _write_scannet_temp_dataset(frames, scene_id, mesh, work_root)

    _run_cropformer_prediction(
        temp_scene=temp_scene,
        cropformer_root=Path(cropformer_root),
        cropformer_config=Path(cropformer_config),
        cropformer_checkpoint=Path(cropformer_checkpoint),
        confidence_threshold=confidence_threshold,
    )

    scene_points = np.asarray(mesh.vertices).astype(np.float32)
    if len(scene_points) == 0:
        raise RuntimeError("No mesh vertices remain after workspace cropping")
    dataset = TempScanNetDataset(temp_scene.scene_dir, scene_points=scene_points, image_size=(frames[0].w, frames[0].h))
    clustering_args = _build_clustering_args(
        step=step,
        mask_visible_threshold=mask_visible_threshold,
        undersegment_filter_threshold=undersegment_filter_threshold,
        view_consensus_threshold=view_consensus_threshold,
        contained_threshold=contained_threshold,
        point_filter_threshold=point_filter_threshold,
        debug=debug,
    )
    clustered_objects = _cluster_objects(dataset, clustering_args)
    if len(clustered_objects) == 0:
        raise RuntimeError("MaskClustering produced no clustered objects")

    _save_object_dict(clustered_objects, work_root / "object_dict.npy")
    instance_groups = _build_instance_groups_from_clustered_masks(
        clustered_objects,
        dataset,
        temp_scene.export_to_frame,
        frames,
        workspace_voxels,
    )

    all_label_ids = np.array(
        sorted({int(label_id) for mask in instance_groups.values() for label_id in np.unique(mask) if label_id > 0}),
        dtype=np.int32,
    )
    all_label_ids = all_label_ids[all_label_ids > 0]
    frame_counts = {label_id: 0 for label_id in all_label_ids}
    for mask in instance_groups.values():
        for label_id in all_label_ids:
            if np.any(mask == label_id):
                frame_counts[label_id] += 1

    valid_ids = np.array([label_id for label_id, count in frame_counts.items() if count >= 3], dtype=np.int32)
    logger.info("Labels in >= 3 frames: %d / %d", len(valid_ids), len(all_label_ids))
    for frame_name in instance_groups:
        instance_groups[frame_name][~np.isin(instance_groups[frame_name], valid_ids)] = 0

    table_id = determine_table_instance_id(frames, instance_groups, scene.ground_plane, valid_ids)
    logger.info("Table instance id: %d", table_id)
    if table_id > 0:
        valid_ids = valid_ids[valid_ids != table_id]
        for frame_name in instance_groups:
            instance_groups[frame_name][instance_groups[frame_name] == table_id] = 0

    for frame_name in instance_groups:
        instance_groups[frame_name][~np.isin(instance_groups[frame_name], valid_ids)] = 0

    frame_ids: List[int] = []
    pixel_masks: List[np.ndarray] = []
    results_path = None
    if intermediate_outputs_path is not None:
        results_path = intermediate_outputs_path / "instances"
        results_path.mkdir(parents=True, exist_ok=True)

    for obs_frame in observations.frames:
        frame_ids.append(obs_frame.id)
        mask = instance_groups.get(obs_frame.name, np.zeros((frames[0].h, frames[0].w), dtype=np.int32))
        pixel_masks.append(mask)
        if results_path is not None:
            cv2.imwrite(str(results_path / f"{obs_frame.name}.png"), mask.astype(np.uint16))

    instance_mask_objects = InstanceMaskObjectsDef(frame_ids=frame_ids, pixel_object_ids=pixel_masks)
    logger.info("Initialized %d objects (after table removal)", len(valid_ids))
    return ObjectSegmentations(object_segmentations=instance_mask_objects)
