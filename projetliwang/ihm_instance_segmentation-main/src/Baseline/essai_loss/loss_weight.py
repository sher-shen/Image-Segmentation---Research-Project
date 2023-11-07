import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedLoss(nn.Module):
    def __init__(self, pos_weight = torch.tensor([1.0])): # Default weight of 1.0 for balanced classes
        super(BalancedLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pos_weight = torch.as_tensor(self.pos_weight, device=input.device, dtype=input.dtype)
            # Compute the binary cross entropy loss with a weighting factor for positive examples
            loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight)
        else:
            # Compute the binary cross entropy loss without weighting 
            loss = F.binary_cross_entropy_with_logits(input, target)
        return loss
