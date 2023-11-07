import torch
import torch.nn as nn
import torchvision.models as models

# Define the Segmentor class as a subclass of nn.Module for segmentation tasks
class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()
        self.original_size = None  # Placeholder for storing the input image size
        
        # Load a pre-trained ResNet18 model
        resnet18 = models.resnet18(pretrained=True)
        
        # Use part of the ResNet18 as a feature extractor
        self.features = nn.Sequential(*list(resnet18.children())[:-4])
        
        # Unfreeze the weights of the last layer in the feature extractor for training
        for param in self.features[-1].parameters():
            param.requires_grad = True
        
        # Additional convolutional layers
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)  # Batch normalization layer for conv1
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)  # Batch normalization layer for conv2
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)   # Batch normalization layer for conv3
        
        # Final classifier layer to generate binary masks
        self.classifier = nn.Conv2d(64, 1, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(1)  # Batch normalization layer for the classifier
        self.act_func = nn.ReLU6()  # ReLU6 activation function

        # Upsampling layer to resize the output back to the input image size
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),
            nn.ReLU6(),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),
            nn.ReLU6(),
            # More layers can be added if higher upsampling is needed
        )


    def forward(self, x):
        # Store the original size of the input for upsampling later
        self.original_size = (x.size(2), x.size(3))
        x = self.features(x)  # Pass input through feature extraction layers
        x = self.act_func(self.bn1(self.conv1(x)))  # Conv1 -> BN1 -> Activation
        x = self.act_func(self.bn2(self.conv2(x)))  # Conv2 -> BN2 -> Activation
        x = self.act_func(self.bn3(self.conv3(x)))  # Conv3 -> BN3 -> Activation
        
        # Apply the classifier to get the raw score maps
        x = self.classifier(x)
        x = self.bn4(x)  # Apply batch normalization to the score maps
        
        return x
