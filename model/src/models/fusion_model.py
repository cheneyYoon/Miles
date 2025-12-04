"""
Multimodal virality prediction model that fuses text, vision, and scalar features.

From implementation_plan.md lines 176-223:
- Text: BERT-base (768-dim)
- Vision: ResNet-50 (2048-dim)
- Scalars: Engagement metrics (~10-dim)
- Fusion: 3-layer MLP (2826 → 1024 → 256)
- Outputs: Binary classification + Engagement velocity regression
"""

import logging
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from .text_encoder import BERTTextEncoder
from .vision_encoder import ResNetVisionEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalViralityPredictor(nn.Module):
    """
    Multimodal model for viral video prediction.

    Combines:
    - Text features (BERT): 768-dim
    - Visual features (ResNet-50): 2048-dim
    - Scalar features: num_scalar_features-dim

    Architecture matches implementation_plan.md lines 176-223.
    """

    def __init__(
        self,
        num_scalar_features: int = 10,
        freeze_encoders: bool = True,
        fusion_hidden_dims: Tuple[int, int] = (1024, 256),
        dropout_rates: Tuple[float, float] = (0.3, 0.2),
        use_text: bool = True,
        use_vision: bool = True,
    ):
        """
        Initialize the multimodal virality predictor.

        Args:
            num_scalar_features: Number of scalar input features
            freeze_encoders: Whether to freeze BERT and ResNet initially
            fusion_hidden_dims: Hidden dimensions for fusion MLP (layer1, layer2)
            dropout_rates: Dropout rates for fusion layers
            use_text: Whether to use text modality
            use_vision: Whether to use vision modality
        """
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.use_text = use_text
        self.use_vision = use_vision

        # Text encoder (BERT)
        if self.use_text:
            self.text_encoder = BERTTextEncoder(
                model_name='bert-base-uncased',
                freeze=freeze_encoders
            )
            text_dim = self.text_encoder.get_embedding_dim()  # 768
        else:
            self.text_encoder = None
            text_dim = 0

        # Vision encoder (ResNet-50)
        if self.use_vision:
            self.vision_encoder = ResNetVisionEncoder(
                freeze=freeze_encoders,
                pretrained=True
            )
            vision_dim = self.vision_encoder.get_feature_dim()  # 2048
        else:
            self.vision_encoder = None
            vision_dim = 0

        # Calculate total input dimension for fusion layer
        fusion_input_dim = text_dim + vision_dim + num_scalar_features

        # Fusion layers (3-layer MLP)
        # fusion_input_dim (2826) → fusion_hidden_dims[0] (1024) → fusion_hidden_dims[1] (256)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),

            nn.Linear(fusion_hidden_dims[0], fusion_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
        )

        # Task-specific heads
        # Classification head (binary: viral or not)
        self.classifier = nn.Linear(fusion_hidden_dims[1], 2)

        # Regression head (engagement velocity)
        self.regressor = nn.Linear(fusion_hidden_dims[1], 1)

        logger.info(f"Initialized MultimodalViralityPredictor")
        logger.info(f"  Text encoder: {'BERT (768-dim)' if use_text else 'Disabled'}")
        logger.info(f"  Vision encoder: {'ResNet-50 (2048-dim)' if use_vision else 'Disabled'}")
        logger.info(f"  Scalar features: {num_scalar_features}-dim")
        logger.info(f"  Fusion input: {fusion_input_dim}-dim")
        logger.info(f"  Encoders frozen: {freeze_encoders}")

    def forward(
        self,
        text_input: Optional[Dict[str, torch.Tensor]] = None,
        image_input: Optional[torch.Tensor] = None,
        scalar_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multimodal model.

        Args:
            text_input: Dictionary with 'input_ids', 'attention_mask', 'token_type_ids'
                       Shape: [batch_size, seq_length]
            image_input: Image tensor [batch_size, 3, H, W]
            scalar_features: Scalar features [batch_size, num_scalar_features]

        Returns:
            Tuple of:
            - classification_logits: [batch_size, 2]
            - regression_output: [batch_size, 1]
        """
        features = []

        # 1. Text encoding
        if self.use_text and text_input is not None:
            text_emb = self.text_encoder(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                token_type_ids=text_input.get('token_type_ids')
            )  # [batch_size, 768]
            features.append(text_emb)

        # 2. Image encoding
        if self.use_vision and image_input is not None:
            img_emb = self.vision_encoder(image_input)  # [batch_size, 2048]
            features.append(img_emb)

        # 3. Scalar features
        if scalar_features is not None:
            features.append(scalar_features)  # [batch_size, num_scalar_features]

        # 4. Concatenate all features
        if not features:
            raise ValueError("At least one modality must be provided")

        combined = torch.cat(features, dim=1)  # [batch_size, fusion_input_dim]

        # 5. Fusion
        fused = self.fusion(combined)  # [batch_size, 256]

        # 6. Task-specific predictions
        classification_logits = self.classifier(fused)  # [batch_size, 2]
        regression_output = self.regressor(fused)  # [batch_size, 1]

        return classification_logits, regression_output

    def freeze_encoders(self):
        """Freeze both text and vision encoders."""
        if self.text_encoder:
            self.text_encoder.freeze_bert()
        if self.vision_encoder:
            self.vision_encoder.freeze_resnet()
        logger.info("Encoders frozen")

    def unfreeze_encoders(self, text_layers: Optional[int] = None, vision_blocks: Optional[int] = None):
        """
        Unfreeze encoders for fine-tuning.

        Args:
            text_layers: Number of BERT layers to unfreeze (None = all)
            vision_blocks: Number of ResNet blocks to unfreeze (None = all)
        """
        if self.text_encoder:
            self.text_encoder.unfreeze_bert(num_layers=text_layers)
        if self.vision_encoder:
            self.vision_encoder.unfreeze_resnet(num_blocks=vision_blocks)
        logger.info(f"Encoders unfrozen (text_layers={text_layers}, vision_blocks={vision_blocks})")

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result = {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
        }

        # Per-component counts
        if self.text_encoder:
            text_params = self.text_encoder.count_parameters()
            result['text_encoder'] = text_params

        if self.vision_encoder:
            vision_params = self.vision_encoder.count_parameters()
            result['vision_encoder'] = vision_params

        # Fusion + heads
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        regressor_params = sum(p.numel() for p in self.regressor.parameters())

        result['fusion'] = fusion_params
        result['classifier'] = classifier_params
        result['regressor'] = regressor_params

        return result


class TextOnlyViralityPredictor(nn.Module):
    """
    Simplified model using only text features.
    Useful for ablation studies or when images are not available.
    """

    def __init__(
        self,
        num_scalar_features: int = 10,
        freeze_encoder: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize text-only model.

        Args:
            num_scalar_features: Number of scalar features
            freeze_encoder: Whether to freeze BERT
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout rate
        """
        super().__init__()

        self.text_encoder = BERTTextEncoder(freeze=freeze_encoder)
        text_dim = self.text_encoder.get_embedding_dim()  # 768

        input_dim = text_dim + num_scalar_features

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, 2)
        self.regressor = nn.Linear(hidden_dim, 1)

        logger.info(f"Initialized TextOnlyViralityPredictor (input_dim={input_dim})")

    def forward(
        self,
        text_input: Dict[str, torch.Tensor],
        scalar_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        text_emb = self.text_encoder(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            token_type_ids=text_input.get('token_type_ids')
        )

        combined = torch.cat([text_emb, scalar_features], dim=1)
        hidden = self.mlp(combined)

        classification = self.classifier(hidden)
        regression = self.regressor(hidden)

        return classification, regression


if __name__ == "__main__":
    # Test the multimodal model
    logger.info("Testing MultimodalViralityPredictor...")

    batch_size = 4

    # Create dummy inputs
    dummy_text = {
        'input_ids': torch.randint(0, 30522, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_scalars = torch.randn(batch_size, 10)

    # Initialize model with frozen encoders
    model = MultimodalViralityPredictor(
        num_scalar_features=10,
        freeze_encoders=True
    )

    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    print(f"  Text encoder: {params['text_encoder']['total']:,}")
    print(f"  Vision encoder: {params['vision_encoder']['total']:,}")
    print(f"  Fusion: {params['fusion']:,}")

    # Forward pass
    classification, regression = model(dummy_text, dummy_images, dummy_scalars)

    print(f"\nOutput shapes:")
    print(f"  Classification: {classification.shape} (expected: [{batch_size}, 2])")
    print(f"  Regression: {regression.shape} (expected: [{batch_size}, 1])")

    assert classification.shape == (batch_size, 2), "Classification shape mismatch"
    assert regression.shape == (batch_size, 1), "Regression shape mismatch"

    # Test text-only model
    logger.info("\nTesting TextOnlyViralityPredictor...")
    text_only_model = TextOnlyViralityPredictor(num_scalar_features=10)
    classification_t, regression_t = text_only_model(dummy_text, dummy_scalars)
    print(f"Text-only output shapes: {classification_t.shape}, {regression_t.shape}")

    logger.info("\nAll tests passed!")
