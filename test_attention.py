"""
Unit tests for attention mechanism implementations
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main module (renamed to avoid import issues)
import importlib.util
spec = importlib.util.spec_from_file_location("attention_module", "0128.py")
attention_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention_module)

ScaledDotProductAttention = attention_module.ScaledDotProductAttention
MultiHeadAttention = attention_module.MultiHeadAttention
AttentionConfig = attention_module.AttentionConfig

class TestScaledDotProductAttention:
    """Test cases for ScaledDotProductAttention"""
    
    def test_attention_shape(self):
        """Test that attention output has correct shape"""
        batch_size, seq_len, d_model = 2, 5, 64
        attention = ScaledDotProductAttention()
        
        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along the last dimension"""
        attention = ScaledDotProductAttention()
        
        Q = torch.randn(1, 3, 4)
        K = torch.randn(1, 3, 4)
        V = torch.randn(1, 3, 4)
        
        _, weights = attention(Q, K, V)
        
        # Check that weights sum to 1 for each query
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    
    def test_attention_with_mask(self):
        """Test attention with masking"""
        attention = ScaledDotProductAttention()
        
        Q = torch.randn(1, 3, 4)
        K = torch.randn(1, 3, 4)
        V = torch.randn(1, 3, 4)
        
        # Create mask that zeros out the last position
        mask = torch.ones(1, 3, 3)
        mask[0, :, 2] = 0  # Mask last position
        
        output, weights = attention(Q, K, V, mask)
        
        # Check that masked positions have zero attention weights
        assert torch.allclose(weights[0, :, 2], torch.zeros(3), atol=1e-6)

class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention"""
    
    def test_multihead_attention_shape(self):
        """Test that multi-head attention output has correct shape"""
        batch_size, seq_len, d_model, num_heads = 2, 5, 64, 8
        attention = MultiHeadAttention(d_model, num_heads)
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_multihead_attention_heads(self):
        """Test that multi-head attention uses correct number of heads"""
        d_model, num_heads = 64, 8
        attention = MultiHeadAttention(d_model, num_heads)
        
        query = torch.randn(1, 5, d_model)
        key = torch.randn(1, 5, d_model)
        value = torch.randn(1, 5, d_model)
        
        _, weights = attention(query, key, value)
        
        assert weights.shape[1] == num_heads  # Second dimension should be num_heads
    
    def test_multihead_attention_consistency(self):
        """Test that multi-head attention is consistent across runs"""
        d_model, num_heads = 32, 4
        attention = MultiHeadAttention(d_model, num_heads)
        
        query = torch.randn(1, 3, d_model)
        key = torch.randn(1, 3, d_model)
        value = torch.randn(1, 3, d_model)
        
        # Set model to eval mode for consistency
        attention.eval()
        
        with torch.no_grad():
            output1, weights1 = attention(query, key, value)
            output2, weights2 = attention(query, key, value)
        
        assert torch.allclose(output1, output2, atol=1e-6)
        assert torch.allclose(weights1, weights2, atol=1e-6)

class TestAttentionConfig:
    """Test cases for AttentionConfig"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = AttentionConfig()
        
        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.dropout == 0.1
        assert config.max_seq_len == 1000
        assert config.device in ["cpu", "cuda"]
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = AttentionConfig(
            d_model=256,
            num_heads=4,
            dropout=0.2,
            max_seq_len=500
        )
        
        assert config.d_model == 256
        assert config.num_heads == 4
        assert config.dropout == 0.2
        assert config.max_seq_len == 500

if __name__ == "__main__":
    pytest.main([__file__])
