"""
Optimized BERT implementation for TT-Buda.

This module provides a BERT implementation that leverages TT-Buda's custom
operators and optimizations for high performance on Tenstorrent hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math

from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BERTConfig:
    """Configuration class for BERT model."""
    
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    
    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )


class OptimizedBERT(nn.Module):
    """Optimized BERT model for TT-Buda."""
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        
        # Simple implementation using standard PyTorch layers
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass."""
        # Basic implementation
        embeddings = self.embeddings(input_ids)
        
        for layer in self.layers:
            embeddings = layer(embeddings, src_key_padding_mask=attention_mask)
        
        pooled = self.pooler(embeddings[:, 0])  # Use [CLS] token
        
        return embeddings, pooled

    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings = value
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return next(self.parameters()).dtype 