import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard residual block with configurable normalization.

    Implements: output = ReLU(shortcut(x) + conv2(conv1(x)))
    Supports group, batch, instance normalization or no normalization.
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """
        Args:
            in_planes: Number of input channels
            planes: Number of output channels
            norm_fn: Normalization type ('group', 'batch', 'instance', 'none')
            stride: Stride for first convolution (for downsampling)
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        # Configure normalization layers based on norm_fn
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        # Add downsampling shortcut if spatial dimensions change
        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, in_planes, height, width]

        Returns:
            Output tensor [batch, planes, height/stride, width/stride]
        """
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        # Apply shortcut connection
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block (1x1 -> 3x3 -> 1x1 convolutions).

    More efficient than ResidualBlock for deeper networks by reducing
    the number of parameters in the intermediate 3x3 convolution.
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        """
        Args:
            in_planes: Number of input channels
            planes: Number of output channels
            norm_fn: Normalization type ('group', 'batch', 'instance', 'none')
            stride: Stride for middle 3x3 convolution (for downsampling)
        """
        super(BottleneckBlock, self).__init__()

        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        # 3x3 conv on reduced dimensions
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        # 1x1 conv to restore dimensions
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        # Configure normalization layers
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        # Add downsampling shortcut if needed
        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, in_planes, height, width]

        Returns:
            Output tensor [batch, planes, height/stride, width/stride]
        """
        y = x
        # Bottleneck: 1x1 reduce -> 3x3 process -> 1x1 expand
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        # Apply shortcut connection
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

DIM=32  # Base dimension for feature extraction


class BasicEncoder(nn.Module):
    """
    Feature extraction backbone using ResNet-style architecture.

    Extracts multi-scale features from input images for visual odometry.
    Uses residual blocks with configurable normalization.
    """
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        """
        Args:
            output_dim: Dimension of output feature vectors
            norm_fn: Normalization type ('batch', 'group', 'instance', 'none')
            dropout: Dropout probability (0.0 = no dropout)
            multidim: If True, use multi-scale feature extraction (not used in basic version)
        """
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        # Initial normalization layer
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # Initial convolution: RGB -> DIM features with 2x downsampling
        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # Build residual layers with progressive downsampling
        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1)    # Same resolution
        self.layer2 = self._make_layer(2*DIM, stride=2)   # 2x downsample -> 1/4 original
        self.layer3 = self._make_layer(4*DIM, stride=2)   # 2x downsample -> 1/8 original

        # Output projection to desired feature dimension
        self.conv2 = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        # Multi-scale feature extraction (unused in basic version)
        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        # Optional dropout for regularization
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # Initialize weights using Kaiming initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        Create a layer with two residual blocks.

        Args:
            dim: Output channel dimension
            stride: Stride for first block (for downsampling)

        Returns:
            Sequential layer of two residual blocks
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Extract features from input images.

        Args:
            x: Input images [batch, num_frames, 3, height, width]

        Returns:
            Features [batch, num_frames, output_dim, height/8, width/8]
        """
        b, n, c1, h1, w1 = x.shape
        # Flatten batch and frame dimensions for processing
        x = x.view(b*n, c1, h1, w1)

        # Initial feature extraction
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Progressive feature extraction and downsampling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Project to output dimension
        x = self.conv2(x)

        # Reshape back to separate batch and frame dimensions
        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)


class BasicEncoder4(nn.Module):
    """
    Lighter feature extraction backbone with 4x downsampling.

    Similar to BasicEncoder but stops at 1/4 resolution instead of 1/8,
    reducing computation while maintaining good feature quality.
    """
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        """
        Args:
            output_dim: Dimension of output feature vectors
            norm_fn: Normalization type ('batch', 'group', 'instance', 'none')
            dropout: Dropout probability (0.0 = no dropout)
            multidim: Unused, kept for API compatibility
        """
        super(BasicEncoder4, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        # Initial normalization layer
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # Initial convolution: RGB -> DIM features with 2x downsampling
        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # Build residual layers (only 2 stages for 4x total downsampling)
        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1)    # Same resolution
        self.layer2 = self._make_layer(2*DIM, stride=2)   # 2x downsample -> 1/4 original

        # Output projection to desired feature dimension
        self.conv2 = nn.Conv2d(2*DIM, output_dim, kernel_size=1)

        # Optional dropout for regularization
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # Initialize weights using Kaiming initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """
        Create a layer with two residual blocks.

        Args:
            dim: Output channel dimension
            stride: Stride for first block (for downsampling)

        Returns:
            Sequential layer of two residual blocks
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Extract features from input images.

        Args:
            x: Input images [batch, num_frames, 3, height, width]

        Returns:
            Features [batch, num_frames, output_dim, height/4, width/4]
        """
        b, n, c1, h1, w1 = x.shape
        # Flatten batch and frame dimensions for processing
        x = x.view(b*n, c1, h1, w1)

        # Initial feature extraction
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Feature extraction with one downsampling stage
        x = self.layer1(x)
        x = self.layer2(x)

        # Project to output dimension
        x = self.conv2(x)

        # Reshape back to separate batch and frame dimensions
        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
