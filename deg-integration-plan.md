# MaskClustering DEG Integration Plan

This document describes how to integrate `dependencies/MaskClustering` as a DEG external segmenter in the same style as the existing `SegmentAnything3D`, `SAI3D`, `Open3DIS`, and `SAMPro3D` wrappers.

## Goal

Make `dependencies/MaskClustering` callable from DEG through the external segmenter contract:

`pixi run --frozen segment_external <observations.pkl> <scene.pkl> [extra args...]`

The wrapper should:

- accept DEG `Observations` and `SceneSetup`
- use the same TSDF reconstruction and workspace cropping behavior as the other external segmenters
- run MaskClustering on those exported observations
- return `ObjectSegmentations` containing per-frame instance-id masks

## Fairness Requirements

Because this integration is for comparison in an academic paper, the wrapper should preserve the same scene input assumptions as the other external segmenters.

- Reconstruct the scene mesh from the DEG observations using the same TSDF pipeline already used by `SAMPro3D` and `Open3DIS`.
- Apply the same workspace voxel construction and workspace mesh cropping before passing the scene to MaskClustering.
- Avoid giving MaskClustering a larger or different 3D scene representation than the other baselines.
- Prefer reconstructing final per-view instance masks directly from the clustered CropFormer masks that formed each object, rather than raycasting a segmented mesh back into the views.

The direct 2D-mask reconstruction is the preferred path because MaskClustering is fundamentally a 2D-mask-first method. Raycasting should be treated as a fallback if the direct path proves incomplete or inconsistent.

## Current Findings

The existing codebase already contains most of the pieces needed for the integration.

- DEG external segmenters are launched through `src/deg/segmentationinitializers/external.py` and registered in `scripts/external_segmentation_initializers/__init__.py`.
- `SAMPro3D`, `SAI3D`, and `Open3DIS` already implement the common DEG wrapper pattern.
- `SAMPro3D/segmenter.py` already contains a temporary ScanNet-style dataset export helper, but its exact layout does not match MaskClustering's expected dataset layout.
- `MaskClustering` is a 2D-mask-first pipeline:
  - `mask_predict.py` writes per-frame CropFormer masks to `output/mask/*.png`
  - `graph/construction.py` builds a graph from backprojected 2D masks
  - `utils/mask_backprojection.py` backprojects 2D masks to scene points
  - `utils/post_process.py` exports final clustered objects and preserves the source mask memberships in `object_dict.npy`
- MaskClustering's ScanNet dataset loader expects a scene layout under `data/scannet/processed/<scene_id>/...`, with intrinsics at `intrinsic/intrinsic_depth.txt`.

## Recommended Design

Implement MaskClustering as a dedicated DEG wrapper under `dependencies/MaskClustering`.

The wrapper should:

1. Load DEG `Observations` and `SceneSetup`.
2. Convert the DEG observations into `psdframe.Frame` objects.
3. Reconstruct a TSDF mesh from those frames.
4. Build workspace voxels from the DEG table estimate.
5. Crop the mesh to the workspace.
6. Export a temporary ScanNet-style scene with the cropped mesh and all observation frames.
7. Run CropFormer mask prediction on the exported RGB frames.
8. Run MaskClustering clustering against the cropped mesh vertices.
9. Reconstruct DEG per-frame instance masks directly from the grouped CropFormer masks.
10. Apply the same post-filtering conventions as the other wrappers.
11. Save `ObjectSegmentations` and print `objects_path: <path>`.

## Key Design Decisions

### 1. Use the same TSDF and workspace pipeline as the other wrappers

Copy or refactor the already-proven helper logic from `SAMPro3D/segmenter.py` or `Open3DIS/segment.py`:

- `get_dataset_frame_from_observation_frame`
- `_to_cam_open3d`
- `_extract_mesh_bounded`
- `_extract_mesh_bounded_with_res`
- `get_workspace_voxels`
- `_crop_mesh_to_workspace_bbox`
- `_crop_mesh_to_workspace`
- `determine_table_instance_id`

The wrapper should use the same parameters already used elsewhere unless there is a clear reason not to:

- `depth_trunc=2`
- `mesh_res=1024`
- workspace shrink in XY of `0.04m`

### 2. Export a MaskClustering-compatible temporary ScanNet scene

MaskClustering's dataset loader expects a slightly different layout from SAMPro3D. The temporary export should therefore look like this:

`<work_root>/data/scannet/processed/<scene_id>/color/*.jpg`

`<work_root>/data/scannet/processed/<scene_id>/depth/*.png`

`<work_root>/data/scannet/processed/<scene_id>/pose/*.txt`

`<work_root>/data/scannet/processed/<scene_id>/intrinsic/intrinsic_depth.txt`

`<work_root>/data/scannet/processed/<scene_id>/<scene_id>_vh_clean_2.ply`

Export details:

- Write RGB as `.jpg`.
- Write depth as uint16 millimeters.
- Write camera poses in the format expected by `dataset/scannet.py` and `utils/mask_backprojection.py`.
- Write the cropped mesh as `<scene_id>_vh_clean_2.ply`.
- Renumber frames sequentially as `0, 1, 2, ...`.
- Keep an explicit mapping from exported frame index to original DEG `ObservationFrame.id` and `ObservationFrame.name`.

### 3. Prefer direct 2D reconstruction of final instance masks

This is the preferred output strategy.

MaskClustering preserves, for each final clustered object, a list of source `(frame_id, mask_id, coverage)` tuples in `object_dict.npy`.

That means the wrapper can:

- read the final clustered objects
- read the original per-frame CropFormer mask PNGs
- assign a new DEG object id to each final clustered object
- for each `(frame_id, mask_id)` membership, paint the matching mask pixels in that frame with the DEG object id

This avoids an extra mesh-to-image rendering step and better matches the algorithm's own native representation.

Important behavior for mask painting:

- Build one `H x W` integer mask per exported frame.
- Process objects in a deterministic order.
- If two final objects claim overlapping pixels in the same frame, resolve collisions deterministically.
- A reasonable default is to prioritize the higher `coverage` tuple first for that frame, then keep the first assignment.

### 4. Keep mesh raycasting as a fallback, not the primary path

If the direct 2D reconstruction path fails in practice, the fallback is:

- construct per-vertex labels from the final clustered point ids
- reuse the mesh raycasting helper pattern already implemented in `SAMPro3D`, `SAI3D`, and `Open3DIS`
- produce per-frame instance-id masks from the segmented mesh

This fallback is technically straightforward, but it is less faithful to MaskClustering's own 2D-mask-driven behavior.

### 5. Run MaskClustering in-process after mask prediction

Do not rely on `python main.py --config scannet` as the main integration entrypoint.

Reason:

- `main.py`, `run.py`, and the dataset classes assume fixed repo-relative paths such as `./data/scannet/processed/...`.
- The DEG wrapper needs to operate on a temporary one-scene export without mutating benchmark datasets.

Recommended approach:

- run `mask_predict.py` as a subprocess, because that is already a self-contained CLI for 2D mask generation
- run the clustering stage in-process inside the wrapper by importing and calling:
  - `graph.construction.mask_graph_construction`
  - `graph.iterative_clustering.iterative_clustering`
  - `utils.post_process.dbscan_process`
  - `utils.post_process.filter_point`
  - `utils.post_process.merge_overlapping_objects`
- provide a small local dataset adapter object that exposes the methods MaskClustering expects

This avoids invasive upstream changes while keeping the wrapper deterministic and scoped to one temp scene.

## Files To Add

### 1. `dependencies/MaskClustering/segment.py`

Thin CLI entrypoint matching the other external wrappers.

Responsibilities:

- define `tyro` args
- load `Observations` and `SceneSetup`
- seed randomness for determinism
- choose an output directory
- call `segmenter.initialize_scene(...)`
- save the returned `ObjectSegmentations`
- print `objects_path: <path>` to stdout

Recommended CLI parameters:

- positional `observations_path`
- positional `scene_path`
- optional `intermediate_outputs_path`
- optional CropFormer configuration overrides
- optional MaskClustering thresholds if needed
- optional `debug` flag

### 2. `dependencies/MaskClustering/segmenter.py`

Main DEG bridge module.

Recommended public function:

`initialize_scene(observations: Observations, scene: SceneSetup, ...) -> ObjectSegmentations`

Responsibilities:

- DEG observation loading and frame conversion
- TSDF reconstruction
- workspace voxel construction and mesh cropping
- temporary ScanNet-style export
- CropFormer mask prediction
- MaskClustering graph construction and clustering
- conversion of clustered objects into DEG frame masks
- frame-count filtering and table removal
- optional debug artifact export

### 3. `dependencies/MaskClustering/deg_integration_plan.md`

This file.

## Files To Modify

### 1. `dependencies/MaskClustering/pyproject.toml`

Add a DEG-facing pixi task:

`segment_external = "python segment.py"`

Potential dependency additions:

- `tyro`
- `imageio` if used for export
- `psdframe`
- `initializerdefs`

If the local pixi environment does not already expose the DEG packages, add them under `[tool.pixi.pypi-dependencies]` using local editable paths, as done in other wrappers.

### 2. `scripts/external_segmentation_initializers/__init__.py`

Register a new external initializer:

- name: `maskclustering`
- command: `bash .../maskclustering.sh`

### 3. `scripts/external_segmentation_initializers/maskclustering.sh`

New launcher script:

- `cd` into `dependencies/MaskClustering`
- run `pixi run --frozen segment_external "$@"`

### 4. `src/deg/segmentationinitializers/runner.py`

Add `"maskclustering"` to the `SegmentationInitializerConfig.type` literal.

## Detailed Implementation Steps

### Step 1. Add the CLI wrapper

Mirror the structure of:

- `dependencies/SAI3D/segment.py`
- `dependencies/SAMPro3D/segment.py`

The CLI should redirect wrapper logs to stderr and reserve stdout for the final `objects_path:` line.

### Step 2. Reuse the common DEG frame and mesh helpers

Use the smallest correct amount of copied code from `SAMPro3D` or `Open3DIS`.

If there is interest in reducing duplication across wrappers later, those helpers can be moved into a shared utility module, but that refactor is not required for the MaskClustering integration itself.

### Step 3. Export a one-scene temporary dataset

Implement a helper such as:

`_write_scannet_temp_dataset(frames, scene_id, mesh, work_root) -> tuple[Path, dict[int, ObservationFrame]]`

Requirements:

- write the dataset in MaskClustering's expected ScanNet format
- return the temp scene root
- return frame index mappings needed to rebuild masks in original DEG order

### Step 4. Add a local dataset adapter for the temporary export

The in-process clustering path needs an object with the methods MaskClustering expects, including:

- `get_frame_list(step)`
- `get_intrinsics(frame_id)`
- `get_extrinsic(frame_id)`
- `get_depth(frame_id)`
- `get_rgb(frame_id, change_color=True)`
- `get_segmentation(frame_id, align_with_depth=False)`
- `get_scene_points()`

This adapter should point at the temporary export rather than the repo's benchmark dataset folders.

Use the cropped mesh vertices as scene points so the clustering sees the same workspace-filtered 3D geometry as the other wrappers.

### Step 5. Run CropFormer mask prediction

Invoke `mask_predict.py` on the exported scene.

Recommended arguments:

- `--cropformer-root`
- `--config-file`
- `--root <temp_processed_root>`
- `--image_path_pattern color/*.jpg`
- `--dataset scannet`
- `--seq_name_list <scene_id>`
- `--confidence-threshold <value>`
- `--opts MODEL.WEIGHTS <checkpoint>`

Prerequisite handling:

- validate that CropFormer source exists
- validate that CropFormer ops are built
- validate that the checkpoint exists, or fail with a clear error telling the user how to obtain it

The wrapper should not silently switch to a different 2D segmenter.

### Step 6. Run MaskClustering graph construction and clustering

Construct a minimal args object or dataclass carrying the fields used by:

- `mask_graph_construction`
- `iterative_clustering`
- `filter_point`

At minimum, this includes config fields such as:

- `mask_visible_threshold`
- `undersegment_filter_threshold`
- `view_consensus_threshold`
- `contained_threshold`
- `point_filter_threshold`
- `step`
- `debug`

Recommended default behavior:

- default to using every exported observation frame, meaning `step=1`

Reason:

- the DEG observation set is already the controlled comparison input
- skipping most frames with ScanNet's original `step=10` would unnecessarily discard information relative to the other wrappers

If strict reproduction of MaskClustering's original ScanNet evaluation protocol is later required, make `step` configurable but keep `1` as the DEG default.

### Step 7. Convert clustered objects into DEG frame masks

This is the most important wrapper-specific step.

Recommended conversion flow:

1. For each final clustered object, assign a DEG object id starting at `1`.
2. Create an empty integer mask for each exported frame.
3. For each object's `mask_list`, load that frame's predicted CropFormer PNG and select pixels equal to `mask_id`.
4. Paint those pixels with the DEG object id.
5. Resolve overlaps deterministically.
6. Reorder the masks back to the original DEG observation order using the stored mapping.

This yields `pixel_object_ids` directly from the 2D masks that generated the clustering result.

### Step 8. Apply the same DEG-style cleanup used by the other wrappers

After reconstructing per-frame masks:

- remove objects that appear in fewer than 3 frames
- remove the table instance using the DEG `scene.ground_plane`
- zero out all invalid labels

Reusing the same logic as the other wrappers helps keep the benchmark comparison aligned.

### Step 9. Build the final DEG object container

Return:

`ObjectSegmentations(object_segmentations=InstanceMaskObjectsDef(frame_ids=..., pixel_object_ids=...))`

The frame ids must correspond to the original DEG `ObservationFrame.id` values.

## Validation Checklist

### Basic wrapper checks

- `pixi run --frozen segment_external -h`
- confirm the CLI prints help correctly
- confirm stdout is reserved for the final `objects_path:` line

### Export checks

- confirm the temporary scene layout matches MaskClustering's expected ScanNet structure
- confirm `intrinsic/intrinsic_depth.txt` is present and readable
- confirm frame numbering is sequential and consistent across color, depth, and pose
- confirm the cropped mesh is written as `<scene_id>_vh_clean_2.ply`

### Pipeline checks

- confirm CropFormer produces `output/mask/*.png`
- confirm graph construction runs on the exported scene points
- confirm clustered objects are non-empty
- confirm `object_dict` contains source `mask_list` information

### Output checks

- confirm `pixel_object_ids` are returned in original DEG observation order
- confirm object ids are `0` for background and `1..N` for instances
- confirm table removal behaves similarly to the other wrappers
- confirm frame-count filtering behaves similarly to the other wrappers

### Comparison sanity checks

- visually compare the workspace-cropped mesh with the mesh used by `SAMPro3D` or `Open3DIS`
- visually compare a few final frame masks against the underlying CropFormer masks to ensure direct 2D reconstruction is working as intended
- if direct reconstruction and mesh raycasting are both implemented, compare them on one scene to verify the preferred direct path is not dropping obvious object support

## Risks And Likely Pitfalls

- MaskClustering's code assumes repo-relative dataset paths in several places, so the wrapper should avoid depending on `main.py` as-is.
- CropFormer setup is heavy and may fail if the third-party source tree or compiled ops are missing.
- The CropFormer checkpoint is gated and may need manual access.
- Overlap handling when repainting final instance masks from grouped source masks needs to be deterministic.
- MaskClustering uses point-cloud vertices rather than mesh surfaces for clustering, so the wrapper should ensure the exported scene points come from the same cropped mesh geometry used elsewhere.

## Recommended Scope For The First Implementation

Keep the first implementation minimal and aligned with the current wrappers.

- Do not refactor common wrapper helpers into shared utilities yet.
- Do not add semantic classification support.
- Do not try to support the full benchmark `run.py` flow.
- Focus only on the class-agnostic DEG external segmenter path.
- Implement direct 2D reconstruction first.
- Add mesh-raycast fallback only if the direct path proves insufficient.

## Expected Deliverables

1. `dependencies/MaskClustering/segment.py`
2. `dependencies/MaskClustering/segmenter.py`
3. `dependencies/MaskClustering/pyproject.toml` updates
4. `scripts/external_segmentation_initializers/maskclustering.sh`
5. `scripts/external_segmentation_initializers/__init__.py` update
6. `src/deg/segmentationinitializers/runner.py` update
7. local validation that one DEG scene can be segmented end-to-end
