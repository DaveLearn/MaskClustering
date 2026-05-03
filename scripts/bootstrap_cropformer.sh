#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"
ENTITY_DIR="${THIRD_PARTY_DIR}/Entity"
ENTITY_REPO="https://github.com/qqlu/Entity.git"
ENTITY_COMMIT="6e7e13ac91ef508088e1b848167c01f19b00b512"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${ENTITY_DIR}/.git" ]]; then
    git clone "${ENTITY_REPO}" "${ENTITY_DIR}"
fi

git -C "${ENTITY_DIR}" fetch --depth 1 origin "${ENTITY_COMMIT}"
git -C "${ENTITY_DIR}" checkout "${ENTITY_COMMIT}"

CROPFORMER_ROOT="${ENTITY_DIR}/Entityv2/CropFormer"

if [[ ! -d "${CROPFORMER_ROOT}" ]]; then
    printf 'Expected CropFormer at %s\n' "${CROPFORMER_ROOT}" >&2
    exit 1
fi

printf 'CropFormer ready at %s\n' "${CROPFORMER_ROOT}"
