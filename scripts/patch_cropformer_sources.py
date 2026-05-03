from __future__ import annotations

import argparse
from pathlib import Path


PATCHES = {
    "make.sh": [
        (
            "python setup.py build install",
            "set -euo pipefail\n\nexport PYTHONNOUSERSITE=1\n\nrm -rf build\nrm -rf MultiScaleDeformableAttention.egg-info\nrm -f MultiScaleDeformableAttention*.so\n\npython setup.py build_ext --inplace",
        ),
    ],
    "src/ms_deform_attn.h": [
        ("value.type().is_cuda()", "value.is_cuda()"),
    ],
    "src/cuda/ms_deform_attn_cuda.cu": [
        ("value.type().is_cuda()", "value.is_cuda()"),
        ("spatial_shapes.type().is_cuda()", "spatial_shapes.is_cuda()"),
        ("level_start_index.type().is_cuda()", "level_start_index.is_cuda()"),
        ("sampling_loc.type().is_cuda()", "sampling_loc.is_cuda()"),
        ("attn_weight.type().is_cuda()", "attn_weight.is_cuda()"),
        ("grad_output.type().is_cuda()", "grad_output.is_cuda()"),
        ("AT_DISPATCH_FLOATING_TYPES(value.type(),", "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cropformer_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ops_root = args.cropformer_root / "mask2former" / "modeling" / "pixel_decoder" / "ops"

    for relative_path, replacements in PATCHES.items():
        file_path = ops_root / relative_path
        text = file_path.read_text()
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            file_path.write_text(updated)


if __name__ == "__main__":
    main()
