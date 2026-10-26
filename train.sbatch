#!/bin/bash
#SBATCH --account=jicg41
#SBATCH --partition=dc-gpu
#SBATCH --job-name=gnn_test
#SBATCH --output=out/log/gnn_test_%j.out
#SBATCH --error=out/log/gnn_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00


#https://gitlab.jsc.fz-juelich.de/sdlaml/pytorch-at-jsc/-/blob/main/pytorch-ddp-example/run.sbatch?ref_type=heads
curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
curr_dir="$(dirname "$curr_file")"

cp config.yaml "$curr_dir"/out/log/config_"$SLURM_JOB_ID".yaml

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source "$curr_dir"/../../../venv/activate.sh

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi

export MASTER_PORT=54123
# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0


# Train the model (GPU required)
echo "Train model"
srun env -u CUDA_VISIBLE_DEVICES \
    python -u -m torchrun_jsc \
    --nproc_per_node=gpu \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --rdzv_backend=c10d \
    "$curr_dir"/src/train.py "$@"