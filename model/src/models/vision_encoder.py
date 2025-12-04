"""
ResNet vision encoder for extracting visual features from video thumbnails.

From implementation_plan.md: "Vision Encoder: ResNet-50 (25M params, 2048-dim embeddings)"
"""

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResNetVisionEncoder(nn.Module):
    """
    ResNet-50 based vision encoder that extracts 2048-dimensional embeddings from images.

    The model uses pretrained ImageNet weights and can be used with frozen weights initially.
    """

    def __init__(
        self,
        freeze: bool = True,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        """
        Initialize the ResNet vision encoder.

        Args:
            freeze: Whether to freeze ResNet weights initially
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability for the output
        """
        super().__init__()

        # Load pretrained ResNet-50
        logger.info(f"Loading ResNet-50 (pretrained={pretrained})")

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2  # Latest ImageNet weights
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)

        # Remove the final classification layer (fc)
        # ResNet-50 architecture ends with: ... -> avgpool -> fc
        # We want features before fc, which are 2048-dim
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze parameters if requested
        if freeze:
            self.freeze_resnet()

        # Output dimension (ResNet-50 outputs 2048-dim features)
        self.feature_dim = 2048

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        logger.info(f"Initialized ResNetVisionEncoder (feature_dim={self.feature_dim}, frozen={freeze})")

    def freeze_resnet(self):
        """Freeze all ResNet parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        logger.info("ResNet parameters frozen")

    def unfreeze_resnet(self, num_blocks: Optional[int] = None):
        """
        Unfreeze ResNet parameters for fine-tuning.

        ResNet-50 structure:
        - conv1 + bn1 + relu + maxpool
        - layer1 (3 blocks)
        - layer2 (4 blocks)
        - layer3 (6 blocks)
        - layer4 (3 blocks)
        - avgpool

        Args:
            num_blocks: Number of final blocks to unfreeze (None = unfreeze all)
        """
        if num_blocks is None:
            # Unfreeze all parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            logger.info("All ResNet parameters unfrozen")
        else:
            # Unfreeze only the last N blocks
            # Typically unfreeze layer4 first (last residual block)
            children = list(self.feature_extractor.children())

            # Map: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
            # Unfreeze from the end
            for child in children[-(num_blocks + 1):]:
                for param in child.parameters():
                    param.requires_grad = True

            logger.info(f"Unfroze last {num_blocks} ResNet blocks")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet encoder.

        Args:
            images: Input images [batch_size, 3, H, W]

        Returns:
            Visual embeddings [batch_size, feature_dim]
        """
        # Extract features
        features = self.feature_extractor(images)  # [batch_size, 2048, 1, 1]

        # Flatten spatial dimensions
        features = features.squeeze(-1).squeeze(-1)  # [batch_size, 2048]

        # Apply dropout
        features = self.dropout(features)

        return features

    def get_feature_dim(self) -> int:
        """Return the dimensionality of the output features."""
        return self.feature_dim

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class LightweightVisionEncoder(nn.Module):
    """
    Lightweight alternative to ResNet-50 using a smaller CNN.
    Useful for faster experimentation or limited compute resources.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        dropout: float = 0.2
    ):
        """
        Initialize lightweight vision encoder.

        Args:
            feature_dim: Output feature dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Simple CNN architecture
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Projection to desired feature dim
        self.projection = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"Initialized LightweightVisionEncoder (feature_dim={feature_dim})")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Input images [batch_size, 3, H, W]

        Returns:
            Visual embeddings [batch_size, feature_dim]
        """
        # Extract features
        features = self.conv_layers(images)  # [batch_size, 256, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch_size, 256]

        # Project to desired dimension
        features = self.projection(features)  # [batch_size, feature_dim]

        # Apply dropout
        features = self.dropout(features)

        return features

    def get_feature_dim(self) -> int:
        """Return the dimensionality of the output features."""
        return self.feature_dim


if __name__ == "__main__":
    # Test the ResNet encoder
    logger.info("Testing ResNetVisionEncoder...")

    # Create dummy input (batch of 4 images, 224x224)
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    # Test with frozen ResNet
    encoder = ResNetVisionEncoder(freeze=True, pretrained=False)  # pretrained=False for faster testing
    params = encoder.count_parameters()
    print(f"\nParameters: Total={params['total']:,}, Trainable={params['trainable']:,}, Frozen={params['frozen']:,}")

    # Forward pass
    features = encoder(dummy_images)
    print(f"Output shape: {features.shape}")
    assert features.shape == (batch_size, 2048), "Unexpected output shape"

    # Test unfreezing
    encoder.unfreeze_resnet(num_blocks=1)
    params = encoder.count_parameters()
    print(f"\nAfter unfreezing 1 block:")
    print(f"Parameters: Total={params['total']:,}, Trainable={params['trainable']:,}, Frozen={params['frozen']:,}")

    # Test lightweight encoder
    logger.info("\nTesting LightweightVisionEncoder...")
    lightweight = LightweightVisionEncoder(feature_dim=512)
    features_light = lightweight(dummy_images)
    print(f"Lightweight output shape: {features_light.shape}")

    logger.info("\nAll tests passed!")
