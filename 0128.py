"""
Project 128: Advanced Attention Mechanism Implementation
=======================================================

This project implements modern attention mechanisms including:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Self-Attention and Cross-Attention
- Interactive visualizations and demonstrations

The attention mechanism is the core innovation behind transformers, enabling
models to focus on specific parts of input when generating output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import json
import random
from dataclasses import dataclass
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention implementation
    
    This is the core attention mechanism used in transformers.
    It computes attention weights by taking the dot product of queries and keys,
    scaling by the square root of the dimension, and applying softmax.
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of scaled dot-product attention
        
        Args:
            Q: Query tensor of shape (batch_size, seq_len, d_k)
            K: Key tensor of shape (batch_size, seq_len, d_k)
            V: Value tensor of shape (batch_size, seq_len, d_v)
            mask: Optional mask tensor
            
        Returns:
            output: Weighted sum of values
            attention_weights: Attention weights matrix
        """
        # Step 1: Calculate raw attention scores
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Step 2: Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 3: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Step 4: Weighted sum of values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation
    
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask tensor
            
        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attn_output)
        
        return output, attention_weights

def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """Create padding mask for attention"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size: int) -> torch.Tensor:
    """Create look-ahead mask for decoder self-attention"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

def generate_sample_data(num_samples: int = 10) -> list:
    """Generate sample text data for demonstration"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Attention is all you need for transformer models",
        "Machine learning algorithms are revolutionizing technology",
        "Natural language processing enables computers to understand text",
        "Deep learning models require large amounts of training data",
        "Neural networks can learn complex patterns from data",
        "Transformers have revolutionized natural language processing",
        "Self-attention mechanisms allow models to focus on relevant information",
        "Multi-head attention provides different representation subspaces",
        "Scaled dot-product attention is computationally efficient"
    ]
    
    return sample_texts[:num_samples]

def visualize_attention_weights(attention_weights: torch.Tensor, 
                              tokens: list, 
                              title: str = "Attention Weights Heatmap"):
    """Visualize attention weights as a heatmap"""
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy and average across heads if multi-head
    if attention_weights.dim() == 4:  # Multi-head attention
        weights = attention_weights.mean(dim=1).squeeze().detach().numpy()
    else:  # Single-head attention
        weights = attention_weights.squeeze().detach().numpy()
    
    # Create heatmap
    sns.heatmap(weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues', 
                annot=True, 
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Key Tokens', fontsize=12)
    plt.ylabel('Query Tokens', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def demonstrate_attention_mechanisms():
    """Demonstrate different attention mechanisms"""
    print("🚀 Attention Mechanism Demonstration")
    print("=" * 50)
    
    # Configuration
    config = AttentionConfig()
    device = torch.device(config.device)
    
    # Generate sample data
    sample_texts = generate_sample_data(5)
    print(f"📝 Sample texts: {len(sample_texts)}")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1}. {text}")
    
    # Create simple tokenized representation
    tokens = ["The", "quick", "brown", "fox", "jumps"]
    seq_len = len(tokens)
    d_model = 64
    
    # Create sample embeddings (in practice, these would come from an embedding layer)
    torch.manual_seed(42)
    embeddings = torch.randn(1, seq_len, d_model, device=device)
    
    print(f"\n🔧 Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Device: {device}")
    
    # 1. Scaled Dot-Product Attention
    print(f"\n1️⃣ Scaled Dot-Product Attention")
    print("-" * 30)
    
    attention_layer = ScaledDotProductAttention().to(device)
    output, weights = attention_layer(embeddings, embeddings, embeddings)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # 2. Multi-Head Attention
    print(f"\n2️⃣ Multi-Head Attention")
    print("-" * 30)
    
    num_heads = 8
    multihead_attn = MultiHeadAttention(d_model, num_heads).to(device)
    mh_output, mh_weights = multihead_attn(embeddings, embeddings, embeddings)
    
    print(f"Number of heads: {num_heads}")
    print(f"Output shape: {mh_output.shape}")
    print(f"Attention weights shape: {mh_weights.shape}")
    
    # Visualize attention weights
    print(f"\n📊 Visualizing Attention Weights...")
    visualize_attention_weights(weights, tokens, "Scaled Dot-Product Attention")
    visualize_attention_weights(mh_weights, tokens, "Multi-Head Attention (Averaged)")
    
    return {
        'single_head': (output, weights),
        'multi_head': (mh_output, mh_weights),
        'tokens': tokens,
        'config': config
    }

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_attention_mechanisms()
    
    print(f"\n✅ Demonstration completed successfully!")
    print(f"🎯 Key insights:")
    print(f"  - Attention weights show which tokens attend to each other")
    print(f"  - Multi-head attention provides richer representations")
    print(f"  - The mechanism enables parallel processing of sequences")