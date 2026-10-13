"""Boolean causal masks for the attention utility interfaces."""
import torch

class TriangularCausalMask:
    def __init__(self, B, L, device='cpu'):
        with torch.no_grad():
            self._mask = torch.triu(torch.ones((B, 1, L, L), dtype=torch.bool, device=device), diagonal=1)
    @property
    def mask(self):
        return self._mask

class ProbMask:
    def __init__(self, B, H, L, index, scores, device='cpu'):
        base = torch.ones((L, scores.shape[-1]), dtype=torch.bool, device=device).triu(1)
        expanded = base[None, None, :, :].expand(B, H, L, scores.shape[-1])
        selected = expanded[torch.arange(B, device=device)[:, None, None], torch.arange(H, device=device)[None, :, None], index, :]
        self._mask = selected.reshape_as(scores)
    @property
    def mask(self):
        return self._mask
