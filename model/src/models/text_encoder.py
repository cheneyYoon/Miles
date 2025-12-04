"""
BERT text encoder for extracting semantic embeddings from video titles and descriptions.

From implementation_plan.md: "Text Encoder: BERT-base-uncased (110M params, 768-dim embeddings)"
"""

import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTTextEncoder(nn.Module):
    """
    BERT-based text encoder that extracts 768-dimensional embeddings.

    The model can be used with frozen weights initially and fine-tuned later.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        freeze: bool = True,
        dropout: float = 0.1,
        pooling_strategy: str = 'cls'
    ):
        """
        Initialize the BERT text encoder.

        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze BERT weights initially
            dropout: Dropout probability for the output
            pooling_strategy: How to pool BERT outputs ('cls', 'mean', or 'max')
        """
        super().__init__()

        self.model_name = model_name
        self.pooling_strategy = pooling_strategy

        # Load pretrained BERT
        logger.info(f"Loading BERT model: {model_name}")
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config

        # Freeze parameters if requested
        if freeze:
            self.freeze_bert()

        # Output dimension
        self.hidden_size = self.config.hidden_size  # 768 for bert-base

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        logger.info(f"Initialized BERTTextEncoder (hidden_size={self.hidden_size}, frozen={freeze})")

    def freeze_bert(self):
        """Freeze all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info("BERT parameters frozen")

    def unfreeze_bert(self, num_layers: Optional[int] = None):
        """
        Unfreeze BERT parameters for fine-tuning.

        Args:
            num_layers: Number of top layers to unfreeze (None = unfreeze all)
        """
        if num_layers is None:
            # Unfreeze all parameters
            for param in self.bert.parameters():
                param.requires_grad = True
            logger.info("All BERT parameters unfrozen")
        else:
            # Unfreeze only the top N layers
            # BERT-base has 12 layers (layers 0-11)
            layers_to_unfreeze = list(range(12 - num_layers, 12))

            for layer_idx in layers_to_unfreeze:
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True

            # Also unfreeze pooler
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

            logger.info(f"Unfroze top {num_layers} BERT layers: {layers_to_unfreeze}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through BERT encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length] (optional)

        Returns:
            Text embeddings [batch_size, hidden_size]
        """
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Extract embeddings based on pooling strategy
        if self.pooling_strategy == 'cls':
            # Use [CLS] token embedding (pooler output)
            embeddings = outputs.pooler_output  # [batch_size, hidden_size]

        elif self.pooling_strategy == 'mean':
            # Mean pooling over all tokens
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
            # Mask out padding tokens
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        elif self.pooling_strategy == 'max':
            # Max pooling over all tokens
            last_hidden_state = outputs.last_hidden_state
            # Set padding tokens to large negative value
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state = last_hidden_state.masked_fill(~attention_mask_expanded.bool(), -1e9)
            embeddings = torch.max(last_hidden_state, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the output embeddings."""
        return self.hidden_size

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class LightweightTextEncoder(nn.Module):
    """
    Lightweight alternative to BERT using just an embedding layer + LSTM.
    Useful for faster experimentation or when BERT is too large.
    """

    def __init__(
        self,
        vocab_size: int = 30522,  # BERT vocab size
        embedding_dim: int = 256,
        hidden_dim: int = 384,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize lightweight text encoder.

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Dimension of token embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Output dimension
        self.hidden_size = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"Initialized LightweightTextEncoder (hidden_size={self.hidden_size})")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask (ignored for now)

        Returns:
            Text embeddings [batch_size, hidden_size]
        """
        # Embed tokens
        embedded = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]

        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        # Apply dropout
        embeddings = self.dropout(hidden)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the output embeddings."""
        return self.hidden_size


if __name__ == "__main__":
    # Test the BERT encoder
    logger.info("Testing BERTTextEncoder...")

    # Create dummy input
    batch_size = 4
    seq_length = 128
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)

    # Test with frozen BERT
    encoder = BERTTextEncoder(freeze=True)
    params = encoder.count_parameters()
    print(f"\nParameters: Total={params['total']:,}, Trainable={params['trainable']:,}, Frozen={params['frozen']:,}")

    # Forward pass
    embeddings = encoder(dummy_input_ids, dummy_attention_mask)
    print(f"Output shape: {embeddings.shape}")
    assert embeddings.shape == (batch_size, 768), "Unexpected output shape"

    # Test unfreezing
    encoder.unfreeze_bert(num_layers=2)
    params = encoder.count_parameters()
    print(f"\nAfter unfreezing 2 layers:")
    print(f"Parameters: Total={params['total']:,}, Trainable={params['trainable']:,}, Frozen={params['frozen']:,}")

    # Test lightweight encoder
    logger.info("\nTesting LightweightTextEncoder...")
    lightweight = LightweightTextEncoder()
    embeddings_light = lightweight(dummy_input_ids)
    print(f"Lightweight output shape: {embeddings_light.shape}")

    logger.info("\nAll tests passed!")
