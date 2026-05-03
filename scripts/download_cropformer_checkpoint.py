from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError


DEFAULT_REPO_ID = "qqlu1992/Adobe_EntitySeg"
DEFAULT_REPO_FILE = "CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x/Mask2Former_hornet_3x_576d0b.pth"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/cropformer/Mask2Former_hornet_3x_576d0b.pth"),
        help="Local output path for the checkpoint.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-file", default=DEFAULT_REPO_FILE)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token for gated checkpoint access.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()

    if output_path.exists() and not args.force:
        print(f"checkpoint_path: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cached_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=args.repo_file,
                token=args.token,
            )
        )
    except GatedRepoError as exc:
        raise SystemExit(
            'CropFormer checkpoint access is gated on Hugging Face. '\
            'Set HF_TOKEN (or pass --token) after being granted access, '\
            f'or place the checkpoint manually at {output_path}.\nOriginal error: {exc}'
        ) from exc

    shutil.copy2(cached_path, output_path)
    print(f"checkpoint_path: {output_path}")


if __name__ == "__main__":
    main()
