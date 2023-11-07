import torch

class BalancedLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.3, dice_weight=0.7):
        """Initializes the BalancedLoss module.

        Args:
            smooth (float): A smoothing factor to avoid division by zero errors in the Dice loss.
            bce_weight (float): The weight of the BCE (Binary Cross Entropy) loss in the combined loss.
            dice_weight (float): The weight of the Dice loss in the combined loss.
        """
        super(BalancedLoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the combined BCE and Dice loss.

        Args:
            input (torch.Tensor): The predictions of the model (before sigmoid activation).
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The calculated combined loss.
        """
        # Calculate the BCE loss
        bce_loss = self.bce_loss(input, target)
        
        # Apply sigmoid activation function to the input to prepare for Dice loss calculation
        input_sig = torch.sigmoid(input)
        # Flatten the input and target tensors for loss computation
        input_flat = input_sig.view(-1)
        target_flat = target.view(-1)
        
        # Calculate the Dice coefficient and loss
        intersection = (input_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score

        # Combine the BCE loss and Dice loss according to specified weights
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return combined_loss
