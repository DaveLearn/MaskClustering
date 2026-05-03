#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CROPFORMER_ROOT="${PROJECT_ROOT}/third_party/Entity/Entityv2/CropFormer"
CROPFORMER_ROOT="${CROPFORMER_ROOT:-${DEFAULT_CROPFORMER_ROOT}}"
CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-}}"

if [[ ! -d "${CROPFORMER_ROOT}" ]]; then
    printf 'CropFormer root not found: %s\n' "${CROPFORMER_ROOT}" >&2
    printf 'Run bootstrap_cropformer first.\n' >&2
    exit 1
fi

if [[ -z "${CUDA_HOME}" || ! -d "${CUDA_HOME}" ]]; then
    printf 'CUDA_HOME is not set and could not be inferred from CONDA_PREFIX.\n' >&2
    exit 1
fi

export CUDA_HOME

ENTITY_API_DIR="${CROPFORMER_ROOT}/entity_api/PythonAPI"
OPS_DIR="${CROPFORMER_ROOT}/mask2former/modeling/pixel_decoder/ops"

if [[ ! -d "${ENTITY_API_DIR}" || ! -d "${OPS_DIR}" ]]; then
    printf 'CropFormer source tree is incomplete under %s\n' "${CROPFORMER_ROOT}" >&2
    exit 1
fi

python "${PROJECT_ROOT}/scripts/patch_cropformer_sources.py" "${CROPFORMER_ROOT}"
make -C "${ENTITY_API_DIR}"
(cd "${OPS_DIR}" && bash make.sh)

printf 'CropFormer ops built under %s\n' "${CROPFORMER_ROOT}"
