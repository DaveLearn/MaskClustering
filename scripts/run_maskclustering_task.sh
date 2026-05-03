#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'Usage: %s <pipeline|predict|cluster|evaluate|visualize> [args...]\n' "$0" >&2
    exit 1
fi

MODE="$1"
shift

if [[ ${1-} == "--" ]]; then
    shift
fi

case "${MODE}" in
    pipeline)
        exec python run.py --class-agnostic-only "$@"
        ;;
    predict)
        exec python run.py --class-agnostic-only --skip-clustering --skip-evaluation "$@"
        ;;
    cluster)
        exec python run.py --class-agnostic-only --skip-mask-prediction --skip-evaluation "$@"
        ;;
    evaluate)
        exec python run.py --class-agnostic-only --skip-mask-prediction --skip-clustering "$@"
        ;;
    visualize)
        exec python -m visualize.vis_scene "$@"
        ;;
    *)
        printf 'Unknown mode: %s\n' "${MODE}" >&2
        exit 1
        ;;
esac
