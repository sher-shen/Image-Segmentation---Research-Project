import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import timm  # timm is a library that provides pre-trained models

# Define the Segmentor class which is a type of neural network for image segmentation
class Segmentor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Load a pre-trained EfficientNet model from the timm library
        efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        # Choose the appropriate layers to serve as the feature extractor
        modules = list(efficientnet.children())[:-4]  # Omitting the last 4 layers
        efficientnet = nn.Sequential(*modules)
        # Freeze the parameters of the feature extractor to avoid updating them during training
        for p in efficientnet.parameters():
            p.requires_grad = False
        self.feature_extractor = efficientnet
        
        # Define a series of convolutional layers to process the extracted features
        self.conv1 = torch.nn.Conv2d(
            in_channels=320,  # The number of input channels should match the feature extractor's output channels
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",  # The padding is set to 'same' to preserve the spatial dimensions
        )

        self.bn1 = torch.nn.BatchNorm2d(num_features=128)  # Batch normalization layer for the first conv layer
        self.conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=7,
            stride=1,
            bias=False,
            padding="same",  # Same padding for conv2
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)  # Batch normalization layer for the second conv layer
        self.conv3 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",  # Same padding for conv3
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)  # Batch normalization layer for the third conv layer
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            padding="same",  # Same padding for conv4, though it doesn't alter dimensions because kernel size is 1
        )
        # Upsample the final feature map to the desired output size
        self.upsample = torch.nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True)
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)  # Batch normalization layer for the final conv layer
        self.act_func = torch.nn.ReLU6()  # ReLU6 activation function
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Pass the input through the feature extractor
        x = self.feature_extractor(input)
        
        # Process the extracted features through consecutive conv-bn-activation layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        # Final convolution to get to the desired channel number
        x = self.conv4(x)
        x = self.bn4(x)
        
        # Upsample the output to match the target resolution
        x = self.upsample(x)
        output = x
        return output
