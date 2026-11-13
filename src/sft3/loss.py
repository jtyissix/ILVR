"""ProLaViT Eq. (7)-(9), with fixed knobs to train only backbone parameters."""
import torch
from torch.nn import functional as F

LOSS_NAMES = ("ce", "collapse_text", "collapse_interaction", "collapse_visual")


def distance_weighted_diversity(steps, margin=0.8, alpha=1.0, epsilon=0.1):
    """Return a pair SUM, not a mean, for one example and one latent type.

    steps is [K, N, H]: mean N vectors for each reasoning step, then compare
    unordered distinct steps. No token pairs within a step or across examples.
    FP32 keeps both centroid averaging and cosine stable under BF16 training.
    """
    if steps.ndim != 3:
        raise ValueError("Latent steps must have shape [steps, tokens, hidden]")
    if not (0 <= margin <= 1 and alpha >= 1 and 0 <= epsilon <= 1):
        raise ValueError("Invalid diversity margin, alpha or epsilon")
    if steps.shape[1] == 0 or steps.shape[2] == 0:
        raise ValueError("Each latent step needs nonempty tokens and hidden dimensions")
    if len(steps) < 2:
        return steps.float().sum() * 0.0
    centroids = F.normalize(steps.float().mean(dim=1), dim=-1, eps=1e-8)
    i, j = torch.triu_indices(len(steps), len(steps), offset=1, device=steps.device)
    similarity = (centroids[i] * centroids[j]).sum(-1)
    weights = epsilon + (1.0 - epsilon) * ((j - i).float() / (len(steps) - 1)).pow(alpha)
    return (weights * (similarity - margin).clamp_min(0.0)).sum()
