"""
Model Module
Multiple architectures for hyperspectral material classification
Includes popular deep learning models for spectral analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCNN1D(nn.Module):
    """1D CNN for spectral classification (baseline model)"""
    
    def __init__(self, num_bands, num_classes, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(256 * (num_bands // 8), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class HybridSN(nn.Module):
    """
    Hybrid Spectral-Spatial CNN
    Reference: Roy et al. "HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification"
    """
    def __init__(self, num_bands, num_classes, dropout_rate=0.4):
        super().__init__()
        # 3D convolutions for spectral-spatial features
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.bn3d_3 = nn.BatchNorm3d(32)
        
        # 2D convolutions for spatial features
        self.conv2d_1 = nn.Conv2d(32 * (num_bands - 12), 64, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x shape: (batch, bands)
        # Reshape for 3D conv: (batch, 1, bands, 1, 1)
        x = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))
        
        # Reshape for 2D conv
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1, 1)
        x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        
        x = x.view(batch_size, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class ResNet1D(nn.Module):
    """
    1D ResNet for spectral classification
    Adapted from He et al. "Deep Residual Learning for Image Recognition"
    """
    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
        
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out
    
    def __init__(self, num_bands, num_classes, dropout_rate=0.5):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(self.ResBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(self.ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SpectralAttentionNet(nn.Module):
    """
    Spectral CNN with Attention Mechanism
    Uses channel attention to focus on important spectral bands
    """
    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias=False)
            )
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            b, c, _ = x.size()
            avg_out = self.fc(self.avg_pool(x).view(b, c))
            max_out = self.fc(self.max_pool(x).view(b, c))
            out = self.sigmoid(avg_out + max_out).view(b, c, 1)
            return x * out
    
    def __init__(self, num_bands, num_classes, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.att1 = self.ChannelAttention(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.att2 = self.ChannelAttention(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.att3 = self.ChannelAttention(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(256 * (num_bands // 8), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.att1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.att2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.att3(F.relu(self.bn3(self.conv3(x)))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class DeepSpectralCNN(nn.Module):
    """
    Deep Spectral CNN with multiple layers
    Good for complex material classification tasks
    """
    def __init__(self, num_bands, num_classes, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Legacy class name for backward compatibility
class HyperspectralCNN1D(SpectralCNN1D):
    """
    Legacy 1D CNN for hyperspectral pixel classification.
    Alias for SpectralCNN1D for backward compatibility.
    """

    def __init__(self, num_bands, num_classes=11, dropout_rate=0.3):
        """
        Args:
            num_bands: Number of spectral bands (input dimension)
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(HyperspectralCNN1D, self).__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes

        # First convolutional block
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        # Second convolutional block
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # Third convolutional block
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # Fourth convolutional block
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, num_bands)

        Returns:
            Output tensor (batch_size, num_classes)
        """
        # Add channel dimension: (batch, bands) -> (batch, 1, bands)
        x = x.unsqueeze(1)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch, 256)

        # FC block 1
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # FC block 2
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)

        return x


class HyperspectralCNN1DResidual(nn.Module):
    """
    1D CNN with residual connections for better gradient flow.
    """

    def __init__(self, num_bands, num_classes=11, dropout_rate=0.3):
        super(HyperspectralCNN1DResidual, self).__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes

        # Initial projection
        self.conv_init = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn_init = nn.BatchNorm1d(64)

        # Residual block 1
        self.conv1a = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn1a = nn.BatchNorm1d(64)
        self.conv1b = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn1b = nn.BatchNorm1d(64)

        # Downsampling 1
        self.conv_down1 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_down1 = nn.BatchNorm1d(128)

        # Residual block 2
        self.conv2a = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm1d(128)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm1d(128)

        # Downsampling 2
        self.conv_down2 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn_down2 = nn.BatchNorm1d(256)

        # Residual block 3
        self.conv3a = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm1d(256)
        self.conv3b = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm1d(256)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initial projection
        x = x.unsqueeze(1)
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = F.relu(x)

        # Residual block 1
        identity = x
        out = self.conv1a(x)
        out = self.bn1a(out)
        out = F.relu(out)
        out = self.conv1b(out)
        out = self.bn1b(out)
        out += identity
        x = F.relu(out)

        # Downsampling 1
        x = self.conv_down1(x)
        x = self.bn_down1(x)
        x = F.relu(x)

        # Residual block 2
        identity = x
        out = self.conv2a(x)
        out = self.bn2a(out)
        out = F.relu(out)
        out = self.conv2b(out)
        out = self.bn2b(out)
        out += identity
        x = F.relu(out)

        # Downsampling 2
        x = self.conv_down2(x)
        x = self.bn_down2(x)
        x = F.relu(x)

        # Residual block 3
        identity = x
        out = self.conv3a(x)
        out = self.bn3a(out)
        out = F.relu(out)
        out = self.conv3b(out)
        out = self.bn3b(out)
        out += identity
        x = F.relu(out)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Fully connected
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)

        return x


class DeepCNN1D(nn.Module):
    """
    Deeper 1D CNN with more layers for complex feature extraction.
    """

    def __init__(self, num_bands, num_classes=11, dropout_rate=0.3):
        super(DeepCNN1D, self).__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes

        # Encoder path with increasing channels
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.global_pool(x).squeeze(-1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


class InceptionModule1D(nn.Module):
    """1D Inception module for multi-scale feature extraction."""

    def __init__(self, in_channels, out_channels):
        super(InceptionModule1D, self).__init__()

        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )

        # 1x1 -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )

        # 1x1 -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )

        # max pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionCNN1D(nn.Module):
    """
    Inception-based 1D CNN for hyperspectral classification.
    Multi-scale feature extraction for better performance.
    """

    def __init__(self, num_bands, num_classes=11, dropout_rate=0.3):
        super(InceptionCNN1D, self).__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes

        # Initial conv
        self.conv_init = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn_init = nn.BatchNorm1d(64)

        # Inception modules
        self.inception1 = InceptionModule1D(64, 128)
        self.inception2 = InceptionModule1D(128, 256)
        self.inception3 = InceptionModule1D(256, 512)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn_init(self.conv_init(x)))
        x = F.max_pool1d(x, 2)

        x = self.inception1(x)
        x = F.max_pool1d(x, 2)

        x = self.inception2(x)
        x = F.max_pool1d(x, 2)

        x = self.inception3(x)

        x = self.global_pool(x).squeeze(-1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.fc2(x)

        return x


class LSTM1D(nn.Module):
    """
    LSTM-based model for hyperspectral classification.
    Treats spectral bands as sequential data.
    """

    def __init__(self, num_bands, num_classes=11, dropout_rate=0.3):
        super(LSTM1D, self).__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes

        # LSTM layers
        self.lstm1 = nn.LSTM(1, 128, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, num_bands)
        x = x.unsqueeze(-1)  # (batch, num_bands, 1)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Use last hidden state
        x = x[:, -1, :]

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout3(x)

        x = self.fc2(x)

        return x


class Transformer1D(nn.Module):
    """
    Transformer-based model for hyperspectral classification.
    Uses self-attention to capture spectral relationships.
    """

    def __init__(self, num_bands, num_classes=11, dropout_rate=0.3):
        super(Transformer1D, self).__init__()

        self.num_bands = num_bands
        self.num_classes = num_classes
        self.d_model = 128

        # Input embedding
        self.embedding = nn.Linear(1, self.d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, num_bands, self.d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Classifier
        self.fc1 = nn.Linear(self.d_model, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, num_bands)
        x = x.unsqueeze(-1)  # (batch, num_bands, 1)

        x = self.embedding(x)  # (batch, num_bands, d_model)
        x = x + self.pos_encoding

        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)

        x = self.fc2(x)

        return x


def create_model(num_bands, num_classes=11, model_type='spectral_cnn', dropout_rate=0.3):
    """
    Factory function to create model.

    Args:
        num_bands: Number of input spectral bands
        num_classes: Number of output classes
        model_type: Model architecture type
            - 'spectral_cnn': Basic 1D CNN for spectral classification
            - 'hybrid_sn': Hybrid 3D-2D CNN for spectral-spatial features
            - 'resnet1d': 1D ResNet with residual connections
            - 'attention_net': CNN with channel attention mechanism
            - 'deep_cnn': Deep CNN with 5 layers
            - 'cnn': Alias for spectral_cnn (legacy)
            - 'resnet': 1D ResNet with residual connections (legacy)
            - 'deep': Alias for deep_cnn (legacy)
            - 'inception': Inception-style multi-scale CNN (legacy)
            - 'lstm': Bidirectional LSTM (legacy)
            - 'transformer': Transformer with self-attention (legacy)
        dropout_rate: Dropout probability

    Returns:
        model: PyTorch model
    """
    # New model types
    if model_type == 'spectral_cnn':
        model = SpectralCNN1D(num_bands, num_classes, dropout_rate)
    elif model_type == 'hybrid_sn':
        model = HybridSN(num_bands, num_classes, dropout_rate)
    elif model_type == 'resnet1d':
        model = ResNet1D(num_bands, num_classes, dropout_rate)
    elif model_type == 'attention_net':
        model = SpectralAttentionNet(num_bands, num_classes, dropout_rate)
    elif model_type == 'deep_cnn':
        model = DeepSpectralCNN(num_bands, num_classes, dropout_rate)
    # Legacy model types (backward compatibility)
    elif model_type == 'cnn':
        model = HyperspectralCNN1D(num_bands, num_classes, dropout_rate)
    elif model_type == 'resnet':
        model = HyperspectralCNN1DResidual(num_bands, num_classes, dropout_rate)
    elif model_type == 'deep':
        model = DeepCNN1D(num_bands, num_classes, dropout_rate)
    elif model_type == 'inception':
        model = InceptionCNN1D(num_bands, num_classes, dropout_rate)
    elif model_type == 'lstm':
        model = LSTM1D(num_bands, num_classes, dropout_rate)
    elif model_type == 'transformer':
        model = Transformer1D(num_bands, num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: spectral_cnn, hybrid_sn, resnet1d, attention_net, deep_cnn")

    return model


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    num_bands = 229  # Example with spectral binning
    batch_size = 32

    print("Testing HyperspectralCNN1D...")
    model = create_model(num_bands, num_classes=11, model_type='cnn')
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(batch_size, num_bands)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nTesting HyperspectralCNN1DResidual...")
    model_res = create_model(num_bands, num_classes=11, model_type='resnet')
    print(f"Model parameters: {count_parameters(model_res):,}")

    output_res = model_res(x)
    print(f"Output shape: {output_res.shape}")

    print("\nModel test passed!")
