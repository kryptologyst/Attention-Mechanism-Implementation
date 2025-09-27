# API Reference

## Core Classes

### AttentionConfig

Configuration dataclass for attention mechanisms.

```python
@dataclass
class AttentionConfig:
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

**Parameters:**
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `dropout`: Dropout rate
- `max_seq_len`: Maximum sequence length
- `device`: Device to run on

### ScaledDotProductAttention

Core attention mechanism implementation.

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Parameters:**
- `dropout`: Dropout rate for attention weights

**Forward Parameters:**
- `Q`: Query tensor of shape (batch_size, seq_len, d_k)
- `K`: Key tensor of shape (batch_size, seq_len, d_k)
- `V`: Value tensor of shape (batch_size, seq_len, d_v)
- `mask`: Optional mask tensor

**Returns:**
- `output`: Weighted sum of values
- `attention_weights`: Attention weights matrix

### MultiHeadAttention

Multi-head attention implementation.

```python
class MultiHeadAttention(nn.Module):
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
```

**Parameters:**
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `dropout`: Dropout rate

**Forward Parameters:**
- `query`: Query tensor
- `key`: Key tensor
- `value`: Value tensor
- `mask`: Optional mask tensor

**Returns:**
- `output`: Multi-head attention output
- `attention_weights`: Attention weights

## Utility Functions

### create_padding_mask

Create padding mask for attention.

```python
def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
```

**Parameters:**
- `seq`: Input sequence tensor
- `pad_token`: Padding token ID

**Returns:**
- `mask`: Padding mask tensor

### create_look_ahead_mask

Create look-ahead mask for decoder self-attention.

```python
def create_look_ahead_mask(size: int) -> torch.Tensor:
```

**Parameters:**
- `size`: Sequence length

**Returns:**
- `mask`: Look-ahead mask tensor

### generate_sample_data

Generate sample text data for demonstration.

```python
def generate_sample_data(num_samples: int = 10) -> list:
```

**Parameters:**
- `num_samples`: Number of sample texts to generate

**Returns:**
- `sample_texts`: List of sample text strings

### visualize_attention_weights

Visualize attention weights as a heatmap.

```python
def visualize_attention_weights(attention_weights: torch.Tensor, 
                              tokens: list, 
                              title: str = "Attention Weights Heatmap"):
```

**Parameters:**
- `attention_weights`: Attention weights tensor
- `tokens`: List of token strings
- `title`: Plot title

### demonstrate_attention_mechanisms

Demonstrate different attention mechanisms.

```python
def demonstrate_attention_mechanisms():
```

**Returns:**
- Dictionary containing results and configuration

## Database Classes

### MockTextDatabase

Mock database containing text data for demonstrations.

```python
class MockTextDatabase:
    def __init__(self, db_path: str = "attention_demo.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_documents(self, category: str = None) -> List[Dict[str, Any]]:
    def get_sentences(self, document_id: int = None) -> List[Dict[str, Any]]:
    def get_attention_examples(self) -> List[Dict[str, Any]]:
    def get_random_sentence_pairs(self, num_pairs: int = 5) -> List[Dict[str, str]]:
    def search_documents(self, query: str) -> List[Dict[str, Any]]:
```

**Methods:**
- `get_documents()`: Retrieve documents by category
- `get_sentences()`: Retrieve sentences by document ID
- `get_attention_examples()`: Get attention examples
- `get_random_sentence_pairs()`: Get random sentence pairs
- `search_documents()`: Search documents by content

## Example Usage

### Basic Attention

```python
import torch
from 0128 import ScaledDotProductAttention

# Create attention layer
attention = ScaledDotProductAttention(dropout=0.1)

# Create sample tensors
Q = torch.randn(1, 5, 64)  # batch_size=1, seq_len=5, d_model=64
K = torch.randn(1, 5, 64)
V = torch.randn(1, 5, 64)

# Apply attention
output, weights = attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")
```

### Multi-Head Attention

```python
from 0128 import MultiHeadAttention

# Create multi-head attention
multihead_attn = MultiHeadAttention(d_model=512, num_heads=8)

# Create sample tensors
query = torch.randn(1, 10, 512)
key = torch.randn(1, 10, 512)
value = torch.randn(1, 10, 512)

# Apply multi-head attention
output, weights = multihead_attn(query, key, value)
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")
```

### Using the Database

```python
from mock_database import MockTextDatabase

# Initialize database
db = MockTextDatabase()

# Get documents
documents = db.get_documents(category="technical")
print(f"Found {len(documents)} technical documents")

# Get sentences
sentences = db.get_sentences(document_id=1)
for sent in sentences:
    print(f"Sentence: {sent['sentence']}")

# Search documents
results = db.search_documents("attention")
print(f"Found {len(results)} documents containing 'attention'")
```

### Visualization

```python
from 0128 import visualize_attention_weights
import matplotlib.pyplot as plt

# Create attention weights and tokens
attention_weights = torch.randn(1, 5, 5)
tokens = ["The", "quick", "brown", "fox", "jumps"]

# Visualize
visualize_attention_weights(attention_weights, tokens, "Sample Attention")
plt.show()
```
