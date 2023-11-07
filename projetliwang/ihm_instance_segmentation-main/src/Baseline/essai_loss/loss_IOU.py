import torch
class BalancedLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(BalancedLoss, self).__init__()
        self.smooth = smooth# Smoothing factor to avoid division by zero errors

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Activating the input; model outputs are transformed into probability values through the sigmoid function
        input = torch.sigmoid(input)

        # Flatten the label and prediction tensors
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        # Calculate the intersection and the union
        intersection = (input_flat * target_flat).sum()
        total = (input_flat + target_flat).sum()
        union = total - intersection

        #Calculate the Jaccard index (Intersection over Union)
        IoU = (intersection + self.smooth) / (union + self.smooth)

        # Calculate the Jaccard loss
        return 1 - IoU
