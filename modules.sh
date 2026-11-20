#!/bin/bash
# CLEMU software stack for JSC HPC (Stage 2026), environment managed with uv.
#
# Source it, do not execute it -- `module` is a shell function and the venv
# activation has to stay in the calling shell:
#
#   source modules.sh              # load the stack, sync the uv environment
#   source modules.sh --recreate   # drop and rebuild the environment
#   source modules.sh --check      # ... and print a torch/CUDA/PyG report
#
# Environment:
#   CLEMU_STAGE   module stage to load   (default: Stages/2026)
#   CLEMU_VENV    environment location   (default: $PROJECT/$USER/venv, else <repo>/venv)
#
# Why a venv on top of the module stack instead of a uv-managed torch: the
# PyTorch module is built against the stage's CUDA (CUDA/13 on JURECA-DC,
# JUPITER, JUWELS Booster) and therefore stays supported by the installed driver.
# A PyPI torch wheel would be a second, untested copy plus ~2.5 GB in a $HOME
# quota of 10 GB. The venv is created with --system-site-packages and
# `uv sync --no-install-package torch` stops uv from installing its own.

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "modules.sh must be sourced:  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_clemu_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
_clemu_stage="${CLEMU_STAGE:-Stages/2026}"

# Toolchain first: CUDA/13 and the Python/PyTorch builds in the stage are tied to
# the GCC and OpenMPI versions, so they have to come from the same stack.
_clemu_modules=(Stages/"${_clemu_stage#Stages/}" GCC OpenMPI CUDA Python PyTorch SciPy-bundle)

module purge
for _clemu_module in "${_clemu_modules[@]}"; do
    if ! module load "${_clemu_module}" 2>/dev/null; then
        echo "[modules] required module '${_clemu_module}' is not available in ${_clemu_stage}" >&2
        echo "[modules] check with: module avail 2>&1 | grep -i ${_clemu_module##*/}" >&2
        return 1
    fi
done

# Optional: only needed once MPI runs next to the GPUs.
module load MPI-settings/CUDA 2>/dev/null \
    || echo "[modules] optional module 'MPI-settings/CUDA' not available -- skipped"
unset _clemu_module

# --- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "${HOME}/.local/bin/uv" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    elif command -v pip >/dev/null 2>&1; then
        echo "[modules] installing uv into ~/.local (login node, needs internet)"
        python -m pip install --quiet --user uv
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "[modules] uv is missing and pip is not on PATH; install uv with:" >&2
        echo "           curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        return 1
    fi
fi

# --- environment ------------------------------------------------------------
if [ -z "${CLEMU_VENV:-}" ]; then
    if [ -n "${PROJECT:-}" ] && [ -n "${USER:-}" ]; then
        CLEMU_VENV="${PROJECT}/${USER}/venv"   # $PROJECT, not the $HOME quota
    else
        CLEMU_VENV="${_clemu_repo}/venv"
    fi
fi
export CLEMU_VENV

_clemu_recreate=0
_clemu_check=0
for _clemu_arg in "$@"; do
    case "${_clemu_arg}" in
        --recreate) _clemu_recreate=1 ;;
        --check)    _clemu_check=1 ;;
        *) echo "[modules] unknown argument '${_clemu_arg}'" >&2; return 1 ;;
    esac
done
unset _clemu_arg

if [ "${_clemu_recreate}" -eq 1 ] && [ -d "${CLEMU_VENV}" ]; then
    echo "[modules] removing ${CLEMU_VENV}"
    rm -rf -- "${CLEMU_VENV}"
fi

if [ ! -x "${CLEMU_VENV}/bin/python" ]; then
    echo "[modules] creating ${CLEMU_VENV} (system-site-packages: torch comes from the module)"
    mkdir -p -- "$(dirname -- "${CLEMU_VENV}")"
    uv venv --system-site-packages --python "$(command -v python3)" "${CLEMU_VENV}"
fi

# Point uv at that environment instead of the default ./.venv.
export UV_PROJECT_ENVIRONMENT="${CLEMU_VENV}"

# --no-install-package torch: torch_geometric declares torch as a requirement;
# this keeps the module build authoritative and skips a 2.5 GB download.
# --inexact: `uv sync` otherwise *removes* anything it does not manage, which
# would delete a torch that the `local` group installed for a laptop run.
uv sync --project "${_clemu_repo}" --no-progress --inexact \
    --no-install-package torch --extra plot

# shellcheck disable=SC1091
source "${CLEMU_VENV}/bin/activate"

echo "[modules] ${_clemu_stage} | python $(python -V 2>&1 | cut -d' ' -f2) | venv ${CLEMU_VENV}"

if [ "${_clemu_check}" -eq 1 ]; then
    python - <<'PY'
import torch
import torch_geometric
print(f"torch {torch.__version__} (cuda build {torch.version.cuda})")
print(f"torch_geometric {torch_geometric.__version__}")
print(f"cuda available: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
print(f"nccl built: {torch.distributed.is_nccl_available()}")
if torch.cuda.is_available():
    print(f"device 0: {torch.cuda.get_device_name(0)} | arch list {torch.cuda.get_arch_list()}")
PY
fi

unset _clemu_repo _clemu_stage _clemu_modules _clemu_recreate _clemu_check
