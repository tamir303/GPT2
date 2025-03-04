import torch

from src.impl.utils import get_batch


@torch.no_grad()
def estimate_loss(model, data, eval_iters=50):
    model.eval()

    losses = {'train': 0.0, 'validation': 0.0}
    with torch.no_grad():
        for split in ['train', 'validation']:
            for _ in range(eval_iters):
                src, tgt = get_batch(data)  # Adjust for your data splits
                _, loss = model(src, tgt)
                losses[split] += loss.item()
            losses[split] /= eval_iters

    return losses
