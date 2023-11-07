import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ASPP Module for capturing multi-scale contextual information
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Dilation rates for the different parallel layers of ASPP
        dilations = [1, 6, 12, 18]

        # A module list to hold all the ASPP convolution blocks
        self.aspp_blocks = nn.ModuleList()
        for dilation in dilations:
            # Each block has a conv layer with a different dilation rate
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),  # Batch Normalization
                nn.ReLU(inplace=True)  # ReLU activation
            ))

        # A 1x1 convolution to process the input feature map
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final convolution layer to combine the features from different ASPP blocks
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Apply the 1x1 conv to the input feature map
        conv1x1 = self.conv1x1(x)
        # Initialize the list of feature maps with the 1x1 conv result
        aspp_features = [conv1x1]

        # Append feature maps from each ASPP block
        for block in self.aspp_blocks:
            aspp_features.append(block(x))

        # Concatenate all feature maps along the channel dimension and apply the final conv
        return self.final_conv(torch.cat(aspp_features, dim=1))

# Decoder Module to refine the segmentation predictions
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        # Define a sequence of conv layers to decode the high-level features to the same
        # number of channels as the number of classes
        self.decoder_layers = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        # Pass the input through the decoder layers
        return self.decoder_layers(x)

# Main Segmentor Module integrating backbone, ASPP, and Decoder
class Segmentor(nn.Module):
    def __init__(self, num_classes=1):
        super(Segmentor, self).__init__()
        # Use EfficientNet B1 as the backbone feature extractor
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        
        # Assuming the last output feature size of the EfficientNet backbone
        backbone_output_channels = 320
        # Instantiate the ASPP module using the expected number of channels from the backbone
        self.aspp = ASPP(in_channels=backbone_output_channels, out_channels=256)
        
        # Instantiate the Decoder module
        self.decoder = Decoder(num_classes=num_classes)
        
    def forward(self, x):
        # Pass the input through the backbone to get the high-level features
        features = self.backbone(x)[-1]
        # Process the features through the ASPP module
        x = self.aspp(features)
        # Further process the output through the Decoder module
        x = self.decoder(x)
        
        # Upscale the output back to the input image size using bilinear interpolation
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x
