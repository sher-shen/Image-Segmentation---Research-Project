import torch

class BalancedLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(BalancedLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid activation to get the probabilities
        input = torch.sigmoid(input)
        
        # Calculate intersection and union
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        
        intersection = (input_flat * target_flat).sum()
        dice_coefficient = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coefficient
        
        return dice_loss
