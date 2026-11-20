# CLEMU

Graph-neural-network emulator for CLM5 carbon/nitrogen pools and fluxes
(see `scheme.md` for the design notes). This repository currently holds the
model skeleton, the training loop and a Planetoid example run used to verify the
pipeline end to end before the real CLM5 graph exists.

## Quickstart

```bash
uv sync --group local                               # local machine: torch from PyPI
mkdir -p out                                        # checkpoint directory
uv run python test/planetoid_split_save.py          # -> data/Cora.pt (needs internet)
uv run python src/train.py                          # -> out/GNN_Cora.pt
uv run python src/evaluate.py --split all
```

On JSC HPC (JUWELS / JURECA-DC / JUPITER, Stage 2026) torch comes from the
module stack instead, so the environment is created and synced like this:

```bash
source modules.sh --check                           # module stack + uv environment
```

and for a batch job:

```bash
mkdir -p out && sbatch train.sbatch                 # preflight -> train -> evaluate
```


## Layout

| Path | Purpose |
|---|---|
| `config.yaml` | model / training / evaluation configuration, single source of truth |
| `src/train.py` | training loop; plain `python` (1 GPU, no DDP) or `torchrun` (DDP) |
| `src/evaluate.py` | evaluate a checkpoint on a split, optional history plot |
| `src/models/gnn.py` | GCN / SAGE / GAT stack, node-level readout |
| `src/data/preprocess.py` | graph loading, integrity validation, feature normalisation |
| `src/utils/engine.py` | one epoch, shared by train and evaluate |
| `src/utils/helpers.py` | device, dimension inference, loss/optimizer factories |
| `test/planetoid_split_save.py` | Cora/CiteSeer/PubMed -> masked PyG graph |
| `test/preflight.py` | environment + device + data check, first step of every job |
| `test/synthetic_graph_data.py` | random graph for smoke tests without a download |
| `modules.sh` | Stage 2026 module stack + uv environment, to be sourced |
| `train.sbatch` | Slurm job: preflight, train, evaluate |
| `pyproject.toml` | dependencies for uv; torch is left out on purpose (see below) |

## Configuration contract

- `data.path` points at the `.pt` file; `test/planetoid_split_save.py` writes it.
- `model.input_dim` / `model.output_dim`: leave `null` to infer them from the
  graph. Cora infers 1433 in, 7 out. Setting them explicitly is only safe if
  they match the graph.
- `train.task`: `classification` (CrossEntropyLoss, `y` long) or `regression`
  (MSELoss/L1Loss, `y` float, reshaped to the model output). The CLM5 emulator
  target is regression; the Cora example exercises the same code path with
  classification.
- Splits are **node masks**, not separate files. The graph stays intact: a GNN
  needs the edges that connect train nodes to test nodes, and dropping nodes
  without reindexing `edge_index` corrupts the graph.
- The checkpoint stores its own model config; `evaluate.py` rebuilds the model
  from it, so a later `config.yaml` edit cannot break a saved run.

## Notes for the cluster

- Load the stack by sourcing `modules.sh`; it purges and reloads Stages/2026,
  then creates/syncs the uv environment (default `$PROJECT/$USER/venv`) with
  `--system-site-packages`, so torch comes from the `PyTorch` module rather than
  from a 2.5 GB wheel in `$HOME`. `uv` itself is installed into `~/.local` on
  first use if it is missing.
- `uv sync` is always invoked with `--no-install-package torch`: `torch_geometric`
  requires torch, and installing it would shadow the module build.
- `dc-gpu` has **no internet access**; `dc-gpu-devel` does. Download datasets,
  sync the environment and install uv on a login node or in a devel job.
- The example job uses `--gres=gpu:1` on `dc-gpu-devel`. Production runs go to
  `dc-gpu` with `--gres=gpu:4`: JURECA charges the whole node either way.
- DDP is only initialised when more than one GPU is requested; a one-GPU run has
  no NCCL, no rendezvous and no `CUDA_VISIBLE_DEVICES` rewriting.
