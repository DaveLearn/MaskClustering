#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CROPFORMER_ROOT="${PROJECT_ROOT}/third_party/Entity/Entityv2/CropFormer"
CROPFORMER_ROOT="${CROPFORMER_ROOT:-${DEFAULT_CROPFORMER_ROOT}}"
CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-}}"
export PYTHONNOUSERSITE=1
unset TORCH_CUDA_ARCH_LIST

# Compiler flags leak in from an already-activated parent shell (e.g. a
# launcher run from a root `deg` pixi shell): the parent's own cuda-toolkit
# activation exports CFLAGS/CXXFLAGS/LDFLAGS pointing at its own environment,
# and this project's own gcc/g++ activation prepends onto that rather than
# replacing it. The result compiles the extension against a mix of both
# environments' headers/libs, producing a .so that builds without error but
# fails on import with "does not define module export function". Clear them
# so only this environment's own compiler defaults apply.
unset CFLAGS CXXFLAGS LDFLAGS

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
printf 'Ignoring inherited TORCH_CUDA_ARCH_LIST and using torch auto-detected CUDA architecture flags\n'

ENTITY_API_DIR="${CROPFORMER_ROOT}/entity_api/PythonAPI"
OPS_DIR="${CROPFORMER_ROOT}/mask2former/modeling/pixel_decoder/ops"
SITE_PACKAGES_DIR="$(python -c "import site; print(next(path for path in site.getsitepackages() if path.endswith('site-packages')))")"

if [[ ! -d "${ENTITY_API_DIR}" || ! -d "${OPS_DIR}" ]]; then
    printf 'CropFormer source tree is incomplete under %s\n' "${CROPFORMER_ROOT}" >&2
    exit 1
fi

python "${PROJECT_ROOT}/scripts/patch_cropformer_sources.py" "${CROPFORMER_ROOT}"
make -C "${ENTITY_API_DIR}"

rm -rf "${OPS_DIR}/build"
rm -rf "${OPS_DIR}/MultiScaleDeformableAttention.egg-info"
rm -f "${OPS_DIR}"/MultiScaleDeformableAttention*.so
rm -rf "${SITE_PACKAGES_DIR}"/MultiScaleDeformableAttention*.egg-info
rm -f "${SITE_PACKAGES_DIR}"/MultiScaleDeformableAttention*.so

(cd "${OPS_DIR}" && python setup.py build_ext --inplace)

PYTHONPATH="${OPS_DIR}:${PYTHONPATH:-}" python -c "from pathlib import Path; import MultiScaleDeformableAttention; built = Path(MultiScaleDeformableAttention.__file__).resolve(); expected = Path('${OPS_DIR}').resolve(); print(built); assert expected in built.parents, f'expected in-tree extension under {expected}, got {built}'"

printf 'CropFormer ops built under %s\n' "${CROPFORMER_ROOT}"
