import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class Segmentor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Initialize a ResNet-18 model pre-trained on ImageNet
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the last 4 modules to use as a feature extractor
        modules = list(resnet18.children())[:-4]
        resnet18 = nn.Sequential(*modules)

        # Freeze the parameters (weights) of the feature extractor layers
        for p in resnet18.parameters():
            p.requires_grad = False

        # Store the modified ResNet-18 as the feature extractor
        self.feature_extractor = resnet18

        # Define a convolutional layer that maintains the spatial dimensions (padding='same')
        self.conv1 = torch.nn.Conv2d(
            in_channels=128,  # Ensure this matches the output channels of the previous layer
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",
        )
        
        # Batch normalization layer following the convolution
        self.bn1 = torch.nn.BatchNorm2d(num_features=128)

        # Define a second convolutional layer, with a larger receptive field
        self.conv2 = torch.nn.Conv2d(
            in_channels=128,  # Match the number of input channels to the output of the previous conv layer
            out_channels=256,
            kernel_size=7,  # Larger kernel size
            stride=1,
            bias=False,
            padding="same",
        )
        
        # Batch normalization layer following the second convolution
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)

        # Third convolutional layer with a smaller kernel size to refine features
        self.conv3 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",
        )
        
        # Batch normalization layer for the third convolution
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)

        # Final convolutional layer that reduces the channel dimension to 1 for segmentation
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            padding="same",
        )
        
        # Final batch normalization layer
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)

        # Activation function to be applied after each batch normalization
        self.act_func = torch.nn.ReLU6()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Pass the input through the feature extractor to get the base features
        x = self.feature_extractor(input)

        # Sequentially apply convolutions, batch normalization, and activation function
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        # Final convolution without an activation, this is common for segmentation outputs
        x = self.conv4(x)
        x = self.bn4(x)
        
        # The output is the segmentation map
        output = x
        return output
