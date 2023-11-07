import torch
import torch.nn as nn
import torchvision.models as models

class Segmentor(nn.Module):
    def __init__(self) -> None:
        super(Segmentor, self).__init__()

        # Load a pre-trained ResNet-101 model
        resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # Retain the layers of the ResNet-101 model up to the third block
        # (where the output usually has 1024 channels)
        modules = list(resnet101.children())[:-4]  # Layers up to the end of the third block
        self.feature_extractor = nn.Sequential(*modules)
        
        # Freeze the parameters of the feature extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Adjust conv1 to match the output channel count of ResNet-101
        self.conv1 = nn.Conv2d(
            in_channels=512,  # Match the channel number from the last layer of the feature extractor
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same',
            bias=False
        )
        
        # Define additional convolutional and batch normalization layers
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,  # Modify kernel_size to 3 to be consistent with conv1 and conv3
            stride=1,
            padding='same',
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding='same',
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)
        self.act_func = torch.nn.ReLU6()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Forward pass through the feature extractor
        x = self.feature_extractor(input)
        # Apply successive convolutional and normalization layers with activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        # Final convolution to produce the segmentation map
        x = self.conv4(x)
        x = self.bn4(x)

        output = x 
        return output
