#!/bin/bash
#SBATCH --account=jicg41
#SBATCH --partition=dc-gpu-devel
#SBATCH --job-name=gnn_test
#SBATCH --output=gnn_test_%j.out
#SBATCH --error=gnn_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

# Usage, from the repository root:
#     sbatch train.sbatch                   # example run, config.yaml
#     sbatch train.sbatch --epochs 5        # extra args go to src/train.py
# out/ is tracked (via out/.gitkeep) so the checkpoint directory exists in a fresh
# clone; out/log/ is created below.
#
# Partition: dc-gpu-devel has *internet access* (dc-gpu does not) and a 2 h cap,
# which is what the example needs -- the Cora download and a first `pip install`
# both want the network. For a production run switch to
#     #SBATCH --partition=dc-gpu
#     #SBATCH --gres=gpu:4
# JURECA charges the whole node regardless of the number of GPUs requested, and
# production jobs are expected to use all of them; `--gres=gpu:1` on dc-gpu is
# fine for a smoke test but wastes three A100s.
#
# Job log: kept next to the script because Slurm opens --output/--error *before*
# the script runs, so a path under a directory that does not exist yet kills the
# job instantly. The resolved config is snapshotted into out/log/ instead, which
# this script can create itself.

set -euo pipefail

curr_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$curr_dir"

mkdir -p out out/log
cp config.yaml "out/log/config_${SLURM_JOB_ID}.yaml"

# Stage 2026 toolchain (GCC 14.3, OpenMPI 5.0, CUDA/13 on JURECA-DC) + venv.
# shellcheck source=modules.sh
source "$curr_dir/modules.sh"

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$SRUN_CPUS_PER_TASK"

# Number of GPUs granted by --gres (Slurm exports e.g. "gpu:1" or "1").
gpus="${SLURM_GPUS_ON_NODE:-1}"
gpus="${gpus##*:}"
if ! [[ "$gpus" =~ ^[0-9]+$ ]] || (( gpus < 1 )); then
    echo "could not determine the number of GPUs (SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE:-}')" >&2
    exit 1
fi

extra_args=("$@")

echo "job ${SLURM_JOB_ID:-local} on $(hostname): ${gpus} GPU(s), ${SRUN_CPUS_PER_TASK} CPU(s) per task"

# Step 1: preflight. Runs the environment checks and one forward/backward of the
# real model on the real device, so a CUDA/NCCL/shape problem surfaces here with
# a named check instead of as a device-side assert in the middle of training.
srun --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores --kill-on-bad-exit=1 \
     python -u "$curr_dir/test/preflight.py" --device cuda

# Step 2: train. DDP only when more than one GPU was requested.
if (( gpus > 1 )); then
    export OMP_NUM_THREADS=$(( SRUN_CPUS_PER_TASK / gpus ))

    master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" || true)"
    # No `| head -n 1`: under `set -o pipefail` an scontrol that fails (or takes
    # EPIPE when head exits early) fails the whole pipeline and aborts the job
    # with no explanation, so `|| true` + an explicit check below is used instead.
    master_addr="${master_addr%%$'\n'*}"
    if [ -z "$master_addr" ]; then
        echo "could not resolve the master node from SLURM_JOB_NODELIST='${SLURM_JOB_NODELIST:-}'" >&2
        exit 1
    fi
    case "${SYSTEMNAME:-}" in
        # Address the InfiniBand interface on the JSC machines that have one;
        # don't append twice if Slurm already reports the i-name
        # (cf. pytorch/pytorch#73656).
        juwels|juwelsbooster|jurecadc|jusuf)
            case "${master_addr}" in
                *i) : ;;
                *) master_addr="${master_addr}i" ;;
            esac
            ;;
    esac
    export MASTER_ADDR="$master_addr"
    export MASTER_PORT="${MASTER_PORT:-54123}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"

    # Slurm binds one GPU per *task*, so with --ntasks-per-node=1 only GPU 0 is
    # visible. Unsetting the binding lets torchrun see all granted GPUs and map
    # local ranks onto them; --nproc-per-node="gpu" would instead spawn one
    # process per *visible* GPU, which is wrong when --gres and CUDA_VISIBLE_DEVICES
    # disagree. Hence the explicit count.
    #
    # Static rendezvous on purpose: --master-addr/--master-port/--node-rank skips
    # torchrun's elastic c10d path, which is exactly where JSC hostnames break
    # (pytorch/pytorch#73656: c10d matches MASTER_ADDR against socket.gethostname()
    # and never finds the InfiniBand "i"-suffixed name, and --master_addr is
    # ignored unless the rendezvous backend is static -> deadlock). If you switch
    # to elastic c10d you must add upstream's workaround instead:
    #   --rdzv_backend=c10d \
    #   --rdzv_conf=is_host=$(( SLURM_NODEID == 0 ? 1 : 0 ))
    # torchrun_jsc is not needed either way: it existed to work around this issue
    # on older PyTorch versions and is a no-op with Stage 2026's PyTorch 2.9.
    cmd=(env -u CUDA_VISIBLE_DEVICES
         torchrun
         --nproc-per-node="$gpus"
         --nnodes="${SLURM_JOB_NUM_NODES:-1}"
         --node-rank="${SLURM_NODEID:-0}"
         --master-addr="$MASTER_ADDR"
         --master-port="$MASTER_PORT"
         "$curr_dir/src/train.py")
else
    cmd=(python -u "$curr_dir/src/train.py")
fi

srun --cpus-per-task="$SRUN_CPUS_PER_TASK" --cpu-bind=cores --kill-on-bad-exit=1 \
     "${cmd[@]}" "${extra_args[@]}"

# Evaluate the best checkpoint. Not "optional" in the shell sense: with `set -e`
# a failing evaluate marks the job FAILED, which is what you want -- a job whose
# deliverable is missing should not look successful.
python -u "$curr_dir/src/evaluate.py" --config "$curr_dir/config.yaml" --split all
