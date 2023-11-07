import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, bce_weight=0.5, focal_weight=0.5):
        super(BalancedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        # Compute the binary cross-entropy loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Compute the focal loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        # Combine the two losses
        combined_loss = (self.bce_weight * bce_loss) + (self.focal_weight * focal_loss.mean())
        
        return combined_loss
