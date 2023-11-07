import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Define the Pyramid Pooling Module
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, model_size):
        # Initialize the module with input channel count, list of pool sizes, and model size (output channels)
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        # Iterate over different pool sizes and create a pooling+convolution sequence for each size
        for size in pool_sizes:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),  # Adaptive average pooling to the specified size
                nn.Conv2d(in_channels, model_size, 1, bias=False),  # 1x1 convolution for dimensionality reduction
                nn.BatchNorm2d(model_size),  # Batch normalization
                nn.ReLU(inplace=True)  # Activation function
            ))
        self.features = nn.ModuleList(self.features)  # Wrap into a module list

    def forward(self, x):
        # Define the forward propagation process
        cat = [x]  # Place the input feature map into a list
        # Iterate over the pooling+convolution modules defined earlier
        for feature in self.features:
            # Upsample the feature map back to the size of the original input and add it to the list
            upsampled = F.interpolate(feature(x), size=x.size()[2:], mode='bilinear', align_corners=False)
            cat.append(upsampled)
        # Concatenate the original feature map with the upsampled feature maps along the channel dimension
        return torch.cat(cat, 1)

class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()
        # Load a pre-trained EfficientNet using the timm library
        self.efficientnet = timm.create_model('efficientnet_b1', pretrained=True)
        # Remove the classification layer from the EfficientNet
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-4])
        
        # Assuming the last feature map of EfficientNet has 320 channels
        # Create an instance of the Pyramid Pooling Module, with the input channels matching
        # the channel count of the last feature map from EfficientNet
        self.ppm = PyramidPoolingModule(in_channels=320, pool_sizes=[1, 2, 3, 6], model_size=64)
        
        # Define the final layers of the model including convolution, normalization, and activation function
        # The number of input channels is the sum of the original feature map channels plus all the pyramid layer output channels
        self.final = nn.Sequential(
            nn.Conv2d(320 + 64*4, 64, kernel_size=3, padding=1, bias=False),  # Input channels are 320 + 64*4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # Output a single channel for binary segmentation
        )
        # Upsampling layer to resize the output back to the input dimensions
        self.upsample = nn.Upsample(size=(28,28), mode='bilinear', align_corners=True)

    def forward(self, x):
        # Feature extraction through EfficientNet
        x = self.efficientnet(x)
        # Pass through PPM
        x = self.ppm(x)
        # Final convolutional layers
        x = self.final(x)
        # Upsample to the original input size
        x = self.upsample(x)
        return x
