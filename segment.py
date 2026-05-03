"""CLI entry point for MaskClustering external segmenter.

Usage (invoked by pixi task):
    python segment.py <observations_path> <scene_path>

Outputs ``objects_path: <path>`` to stdout for the parent process to read.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import tyro

from initializerdefs import Observations, SceneSetup
from segmenter import initialize_scene
from utils.config import DEFAULT_CROPFORMER_CHECKPOINT, DEFAULT_CROPFORMER_CONFIG, DEFAULT_CROPFORMER_ROOT


DEFAULT_SEED = 0


@dataclass
class Args:
    observations_path: tyro.conf.Positional[Path]
    """Path to the pickled Observations."""

    scene_path: tyro.conf.Positional[Path]
    """Path to the pickled SceneSetup."""

    cropformer_root: Path = DEFAULT_CROPFORMER_ROOT
    """Path to the upstream CropFormer source tree."""

    cropformer_config: Path = DEFAULT_CROPFORMER_CONFIG
    """Path to the CropFormer config file."""

    cropformer_checkpoint: Path = DEFAULT_CROPFORMER_CHECKPOINT
    """Path to the CropFormer checkpoint."""

    confidence_threshold: float = 0.4
    """Minimum score for CropFormer instance predictions."""

    step: int = 1
    """Frame stride used by MaskClustering graph construction."""

    mask_visible_threshold: float = 0.15
    """Minimum visibility ratio for a mask to count in a frame."""

    undersegment_filter_threshold: float = 0.5
    """Maximum split ratio before a mask is treated as undersegmented."""

    view_consensus_threshold: float = 0.67
    """View-consensus threshold used during iterative clustering."""

    contained_threshold: float = 0.6
    """Containment threshold between masks across views."""

    point_filter_threshold: float = 0.4
    """Minimum detection ratio for a point to remain in a cluster."""

    debug: bool = False
    """Enable verbose MaskClustering debugging behavior."""


def run() -> None:
    logger = logging.getLogger("maskclustering-segmenter")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))
    logger.addHandler(ch)

    args = tyro.cli(Args)

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    with contextlib.redirect_stdout(sys.stderr):
        logger.info("--------------")
        logger.info("Starting MaskClustering initialization")
        logger.info("params: %s", args)
        logger.info("Determinism enabled with seed=%d", DEFAULT_SEED)

        logger.info("Loading observations from %s ...", args.observations_path)
        dataset: Observations = Observations.load(args.observations_path)
        logger.info("Observations loaded.")

        logger.info("Loading scene setup from %s ...", args.scene_path)
        scene = SceneSetup.load(args.scene_path)
        logger.info("Scene loaded.")

        if dataset.id is None:
            logger.info("Dataset has no id, using transient id")
            dataset.id = f"transient_{time.strftime('%Y%m%d-%H%M%S')}"

        project_root = Path(__file__).parent
        output_dir = project_root / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{dataset.id}"

        logger.info("Initializing scene ...")
        objects = initialize_scene(
            dataset,
            scene,
            intermediate_outputs_path=output_dir,
            cropformer_root=args.cropformer_root,
            cropformer_config=args.cropformer_config,
            cropformer_checkpoint=args.cropformer_checkpoint,
            confidence_threshold=args.confidence_threshold,
            step=args.step,
            mask_visible_threshold=args.mask_visible_threshold,
            undersegment_filter_threshold=args.undersegment_filter_threshold,
            view_consensus_threshold=args.view_consensus_threshold,
            contained_threshold=args.contained_threshold,
            point_filter_threshold=args.point_filter_threshold,
            debug=args.debug,
        )

        output_path = output_dir / "objectsdef.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        objects.save(output_path)

        logger.info("Objects saved to %s", output_path)

    print(f"objects_path: {output_path}")


if __name__ == "__main__":
    run()
