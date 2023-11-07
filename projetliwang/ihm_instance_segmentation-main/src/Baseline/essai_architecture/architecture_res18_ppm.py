import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# networks to capture context information at different regions.
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, model_size):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        # For each pool size, create a sequence of adaptive pooling followed by convolution
        for size in pool_sizes:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),  # Adaptive average pooling to the specified size
                nn.Conv2d(in_channels, model_size, 1, bias=False),  # 1x1 convolution to reduce dimension
                nn.BatchNorm2d(model_size),  # Batch normalization
                nn.ReLU(inplace=True)  # ReLU activation function
            ))
        self.features = nn.ModuleList(self.features)  # Wrap as a ModuleList

    def forward(self, x):
        # Forward propagation definition
        cat = [x]  # Start with the input feature map in the list
        # Go through the pooling+convolution modules defined earlier
        for feature in self.features:
            # Upsample the feature maps to the original input size and add them to the list
            upsampled = F.interpolate(feature(x), size=x.size()[2:], mode='bilinear', align_corners=False)
            cat.append(upsampled)
        # Concatenate the original feature map with the upsampled feature maps along the channel dimension
        return torch.cat(cat, 1)

class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()
        self.original_size = None  # To store the original size of the input image

        # Use a pretrained ResNet18 as a feature extractor, but remove the last few layers
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-4])
        
        # Freeze the parameters of the feature extractor so they are not updated during training
        for param in self.features.parameters():
            param.requires_grad = False

        # Instantiate the Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(in_channels=128, pool_sizes=[1, 2, 3, 6], model_size=64)

        # Define the final layers of the model, including convolution, normalization, and activation functions
        self.final = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1, bias=False),  # Update the input channel number to 384
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # A sequence of transposed convolutions for upsampling the output
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),
            nn.ReLU6(),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),
            nn.ReLU6(),
        )

    def forward(self, x):
        # Store the original size of the input
        self.original_size = (x.size(2), x.size(3))
        # Pass input through feature extraction and pyramid pooling layers
        x = self.features(x)
        x = self.ppm(x)
        x = self.final(x)
        
        return x
