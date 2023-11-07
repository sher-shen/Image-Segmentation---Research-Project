import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module used to capture multi-scale contextual information
    without increasing the number of parameters or the amount of computation.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Different dilation rates to capture multi-scale information
        dilations = [1, 6, 12, 18]

        # Create a module list for holding all the dilated convolutions
        self.aspp_blocks = nn.ModuleList()
        for dilation in dilations:
            # Each block contains a dilated convolution followed by BatchNorm and ReLU
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Additional 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final convolution to combine the features from different dilated convolutions
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Apply the 1x1 convolution
        conv1x1 = self.conv1x1(x)
        aspp_features = [conv1x1]

        # Apply each of the ASPP blocks to get multi-scale features
        for block in self.aspp_blocks:
            aspp_features.append(block(x))

        # Concatenate the multi-scale features and pass through final convolution
        return self.final_conv(torch.cat(aspp_features, dim=1))

class Decoder(nn.Module):
    """
    Decoder module for the segmentation network, used to upsample the feature maps
    to the same size as the input image.
    """
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        # Define the layers of the decoder using sequential
        self.decoder_layers = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        # Pass the input through the decoder layers
        return self.decoder_layers(x)

class Segmentor(nn.Module):
    """
    Segmentor network combining a pretrained EfficientNet backbone with ASPP and a Decoder module.
    This network is designed for semantic segmentation tasks.
    """
    def __init__(self, num_classes=1):
        super(Segmentor, self).__init__()
        # Load a pretrained EfficientNet model as the backbone of the segmentor
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, features_only=True)
        
        # Assuming the output feature size of the EfficientNet backbone
        backbone_output_channels = 320
        # Initialize the ASPP module
        self.aspp = ASPP(in_channels=backbone_output_channels, out_channels=256)
        
        # Initialize the decoder
        self.decoder = Decoder(num_classes=num_classes)
        
    def forward(self, x):
        # Extract high-level features from the backbone
        features = self.backbone(x)[-1]
        # Pass features through the ASPP module
        x = self.aspp(features)
        # Pass the resulting features through the decoder
        x = self.decoder(x)
        
        # Upscale to the input image size using bilinear interpolation
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x
