import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module creates a spatial pyramid of pooled representations.
    This module is used to aggregate context at different scales.
    """
    def __init__(self, in_channels, pool_sizes, out_channels):
        super(PyramidPoolingModule, self).__init__()
        # For each pool size, we'll have a pooling layer followed by a convolution
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),  # Adaptive pooling to the specified size
                nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1 conv reduces channels
                nn.BatchNorm2d(out_channels),  # Batch normalization
                nn.ReLU(inplace=True)  # ReLU activation
            )
            for pool_size in pool_sizes  # Iterate through each pool size
        ])

    def forward(self, x):
        features = [x]  # Start with the original feature map
        for pool in self.pools:
            pooled = pool(x)  # Apply adaptive pooling
            # Upsample to the original feature map's size and append to the features list
            upsampled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
            features.append(upsampled)
        # Concatenate the original features with the pooled features along the channel dimension
        return torch.cat(features, dim=1)

class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) is designed to capture multi-scale context by applying
    atrous convolution with different dilation rates.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(ASPP, self).__init__()
        # Define dilations for atrous convolution
        dilations = [1, 6, 12, 18]
        self.aspp_blocks = nn.ModuleList()
        for dilation in dilations:
            # Each block in ASPP consists of a dilated convolution, followed by batch normalization,
            # a ReLU activation, and dropout
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)  # Applying dropout for regularization
            ))
        # Additional 1x1 convolution layer
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Gather features processed by each ASPP block and the 1x1 conv layer
        features = [self.conv1x1(x)] + [block(x) for block in self.aspp_blocks]
        # Concatenate the features along the channel dimension
        return torch.cat(features, dim=1)

class Decoder(nn.Module):
    """
    Decoder for the semantic segmentation network that upsamples the combined feature map from
    the backbone, ASPP, and PPM modules to the size of the input image to produce a dense prediction.
    """
    def __init__(self, num_classes, input_channels):
        super(Decoder, self).__init__()
        # The decoder architecture with convolutions, batch normalization, and ReLU activations
        self.decoder_layers = nn.Sequential(
            nn.Conv2d(input_channels, 256, 3, padding=1, bias=False),  # Convolution to process combined features
            nn.BatchNorm2d(256),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Conv2d(256, num_classes, 1)  # Final convolution to predict class scores for each pixel
        )

    def forward(self, x):
        # Pass the feature map through decoder layers
        return self.decoder_layers(x)

class Segmentor(nn.Module):
    """
    Segmentor network that combines a backbone (EfficientNet), ASPP, Pyramid Pooling Module,
    and a Decoder to perform semantic segmentation.
    """
    def __init__(self, num_classes=1):
        super(Segmentor, self).__init__()
        # Load a pretrained EfficientNet model as the feature extraction backbone
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, features_only=True)

        # Backbone output channels are hardcoded based on the EfficientNet implementation
        backbone_output_channels = 320

        # Initialize the ASPP module with specified output channels
        self.aspp = ASPP(backbone_output_channels, 256)

        # Initialize the Pyramid Pooling Module with specified pool sizes and output channels
        pyramid_pool_sizes = [1, 2, 3, 6]
        ppm_out_channels = 64
        self.ppm = PyramidPoolingModule(backbone_output_channels, pyramid_pool_sizes, ppm_out_channels)

        # Compute the input channels for the decoder (combining ASPP and PPM outputs)
        aspp_output_channels = 256 * (len(dilations) + 1)  # Output channels from ASPP blocks
        ppm_output_channels = ppm_out_channels * len(pyramid_pool_sizes) + backbone_output_channels
        decoder_input_channels = aspp_output_channels + ppm_output_channels

        # Initialize the decoder with the calculated input channels and number of classes
        self.decoder = Decoder(num_classes, decoder_input_channels)

    def forward(self, x):
        # Extract features from the backbone model
        features = self.backbone(x)[-1]

        # Process features through ASPP and PPM
        x_aspp = self.aspp(features)
        x_ppm = self.ppm(features)

        # Concatenate ASPP and PPM features
        x = torch.cat((x_aspp, x_ppm), dim=1)

        # Decode the concatenated features to create a segmentation map
        x = self.decoder(x)

        # Upscale the segmentation map back to the size of the input image
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x
