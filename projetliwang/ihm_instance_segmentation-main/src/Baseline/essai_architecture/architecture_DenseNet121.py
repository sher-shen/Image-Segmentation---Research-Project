import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

# Define a segmentor neural network module
class Segmentor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Using DenseNet121 as the backbone feature extractor
        densenet121 = models.densenet121(pretrained=True).features
        # Freeze the parameters of the DenseNet121 to not compute gradients (this will save memory and computations)
        for p in densenet121.parameters():
            p.requires_grad = False
        
        # The feature map output channel of DenseNet121 is 1024
        self.feature_extractor = densenet121
        self.conv1 = nn.Conv2d(
            in_channels=1024,  # Adjust input channels based on DenseNet121 output
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1  # Use padding=1 to keep feature map size the same ('same' padding)
        )
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=7,
            stride=1,
            bias=False,
            padding=3,  # 'same' padding is equivalent to padding=(kernel_size-1)/2 when stride=1
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)
        self.conv3 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            padding=1,  # 'same' padding, adjusting due to PyTorch not supporting 'same' directly
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            padding=0,  # No padding required for 1x1 convolution
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)
        self.act_func = torch.nn.ReLU6()  # Using ReLU6 activation function for non-linearity

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Pass input through the feature extractor
        x = self.feature_extractor(input)
        # Sequentially pass through each conv-bn-activation layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)
        # Upsample to target resolution (e.g., 28x28)
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        x = self.conv4(x)
        x = self.bn4(x)
        output = x  # This is the final output
        return output
