import torch

class BalancedLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(BalancedLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid activation to get the probabilities
        input = torch.sigmoid(input)
        
        # Flatten input and target to calculate the sums
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        
        # Calculate weights: each weight is the inverse of the square of the frequency of each class
        class_frequencies = torch.bincount(target_flat.int())
        class_weights = (1. / (class_frequencies + self.smooth) ** 2)
        weights = class_weights[target_flat.int()]
        
        # Calculate intersection and union
        intersection = torch.sum(weights * input_flat * target_flat)
        denominator = torch.sum(weights * (input_flat * input_flat + target_flat * target_flat))
        
        # Compute generalized Dice loss
        dice_loss = 1 - (2 * intersection + self.smooth) / (denominator + self.smooth)

        return dice_loss
