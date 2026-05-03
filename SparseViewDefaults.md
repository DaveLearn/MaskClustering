# Sparse-View Defaults

This note records a recommended hyperparameter preset for sparse-view tabletop evaluation in DEG, where a scene may be observed with as few as 5 RGB-D frames, typically from roughly three side views and two top views.

The original MaskClustering defaults were tuned for dense image coverage. In the sparse-view regime, those settings become overly strict because multi-view support is quantized into only a few observations. We therefore consider the following defaults to be a fairer operating point for this setup.

## Recommended Defaults

```text
confidence_threshold            = 0.40
step                            = 1
mask_visible_threshold          = 0.15
undersegment_filter_threshold   = 0.50
view_consensus_threshold        = 0.67
contained_threshold             = 0.60
point_filter_threshold          = 0.40
```

## Justification

`confidence_threshold = 0.40`

- This threshold is applied during CropFormer mask selection in `mask_predict.py`.
- With only 5 views, a single missed 2D mask removes 20% of the available evidence for an object.
- Lowering the threshold from `0.50` to `0.40` improves recall modestly while later geometric and multi-view filters still reject weak proposals.

`step = 1`

- The original ScanNet configuration uses temporal subsampling because it assumes dense video coverage.
- In a sparse-view setting, subsampling would discard a substantial fraction of the already-limited evidence.
- Using every available frame is therefore the fairest choice.

`mask_visible_threshold = 0.15`

- This threshold determines whether a mask is considered visible in another frame during graph construction.
- Sparse wide-baseline views frequently produce partial observations rather than near-complete reprojections.
- Reducing the threshold from `0.30` to `0.15` allows such partial but valid support to contribute without changing the underlying representation.

`undersegment_filter_threshold = 0.50`

- This threshold rejects masks that appear split across too many supporting views.
- In a 5-view setting, even one split observation can represent a large fraction of the total visible support.
- Raising the threshold from `0.30` to `0.50` permits one inconsistent view among a small number of supporting views while still filtering masks that are unstable in the majority of views.

`view_consensus_threshold = 0.67`

- This threshold controls graph connectivity through the ratio of supporting views to jointly observing views.
- In sparse-view evaluation, the default `0.90` effectively requires near-unanimous agreement, which is disproportionately strict when only 2-5 overlapping views are available.
- A value of `0.67` retains a clear majority requirement while tolerating one disagreeing view once at least three supporting observations exist.

`contained_threshold = 0.60`

- This threshold determines whether a candidate mask in another frame is treated as containing the same object support.
- With sparse and topologically diverse views, valid reprojections are more likely to be truncated or fragmented.
- Reducing the threshold from `0.80` to `0.60` preserves a majority criterion while avoiding unnecessary rejection of partially observed correspondences.

`point_filter_threshold = 0.40`

- This threshold removes points whose detection ratio within a cluster is too low during post-processing.
- When the total number of views is small, a strict ratio disproportionately removes points that are only visible or recoverable in a subset of the available views.
- Lowering the threshold from `0.50` to `0.40` preserves object support under sparse coverage while continuing to suppress inconsistent points.

## Summary

These defaults remove dense-video assumptions that would otherwise penalize the method disproportionately when evaluated on a deliberately sparse observation set. The resulting preset preserves the same 2D-mask, 3D-backprojection, and multi-view-consensus pipeline, while recalibrating the acceptance thresholds to the available evidence.
