"""One training/evaluation epoch, shared by ``train.py`` and ``evaluate.py``.

Keeping it here means a checkpoint is always evaluated exactly the way it was
trained: same mask handling, same target shaping, same loss weighting (by node
count, not by batch count -- a graph is one batch, so averaging over batches
would silently weight graphs by their size).
"""

from typing import Optional

import torch

from utils.helpers import prepare_targets


def run_epoch(
    model: torch.nn.Module,
    loader,
    loss_function: torch.nn.Module,
    task: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    split: str = "train",
) -> dict:
    """Run one pass over ``loader``.

    ``optimizer=None`` puts the model in eval mode and runs without gradients.
    If the graph carries a ``{split}_mask`` only masked nodes contribute to the
    loss; without one (e.g. the synthetic smoke-test graph) every node does.
    """
    is_train = optimizer is not None
    model.train(is_train)
    mask_name = f"{split}_mask"

    total_loss = 0.0
    total_correct = 0
    total_nodes = 0
    masked = False

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(device)
            mask = getattr(batch, mask_name, None)
            if mask is not None:
                masked = True

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = model(batch.x, batch.edge_index)
            if mask is not None:
                out_sel, y_sel = out[mask], batch.y[mask]
            else:
                out_sel, y_sel = out, batch.y

            if y_sel.numel() == 0:
                continue

            targets = prepare_targets(y_sel, out_sel, task)
            loss = loss_function(out_sel, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            n = int(targets.size(0))
            total_loss += float(loss.item()) * n
            total_nodes += n
            if task == "classification":
                total_correct += int((out_sel.argmax(dim=-1) == targets).sum().item())

    if total_nodes == 0:
        raise RuntimeError(
            f"No nodes selected for split {split!r}: the mask is empty or the loader is empty."
        )

    metrics = {
        "loss": total_loss / total_nodes,
        "nodes": total_nodes,
        "masked": masked,
    }
    if task == "classification":
        metrics["accuracy"] = total_correct / total_nodes
    return metrics
