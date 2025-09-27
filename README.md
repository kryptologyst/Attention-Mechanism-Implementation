# Attention Mechanism Implementation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-green)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A comprehensive implementation of attention mechanisms in PyTorch, featuring modern transformer architectures, interactive visualizations, and educational demonstrations.

## Features

- **Scaled Dot-Product Attention**: Core attention mechanism implementation
- **Multi-Head Attention**: Parallel attention heads for richer representations
- **Interactive Web UI**: Streamlit-based interface for exploration
- **Comprehensive Visualizations**: Heatmaps, flow diagrams, and statistical analysis
- **Mock Database**: Realistic text data for demonstrations
- **Unit Tests**: Complete test coverage for all components
- **Educational Content**: Detailed explanations and examples

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/attention-mechanism-implementation.git
cd attention-mechanism-implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Command Line Interface
```bash
python 0128.py
```

#### Interactive Web UI
```bash
streamlit run app.py
```

#### Run Tests
```bash
python -m pytest test_attention.py -v
```

## Project Structure

```
attention-mechanism-implementation/
├── 0128.py                 # Main attention mechanism implementation
├── app.py                  # Streamlit web application
├── mock_database.py        # Mock database with sample data
├── test_attention.py       # Unit tests
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
└── docs/                  # Additional documentation
    ├── attention_theory.md
    └── api_reference.md
```

## 🔧 Core Components

### ScaledDotProductAttention

The fundamental attention mechanism that computes attention weights through:
1. Dot product of queries and keys
2. Scaling by square root of dimension
3. Softmax normalization
4. Weighted sum of values

```python
from 0128 import ScaledDotProductAttention

attention = ScaledDotProductAttention(dropout=0.1)
output, weights = attention(Q, K, V)
```

### MultiHeadAttention

Extends single-head attention with multiple parallel attention heads:

```python
from 0128 import MultiHeadAttention

multihead_attn = MultiHeadAttention(d_model=512, num_heads=8)
output, weights = multihead_attn(query, key, value)
```

### MockTextDatabase

Provides realistic text data for demonstrations:

```python
from mock_database import MockTextDatabase

db = MockTextDatabase()
documents = db.get_documents(category="technical")
sentences = db.get_sentences(document_id=1)
```

## Key Concepts

### Attention Mechanism

The attention mechanism allows models to focus on relevant parts of the input when generating output. It's the core innovation behind transformer architectures.

**Mathematical Formulation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q: Query matrix
- K: Key matrix  
- V: Value matrix
- d_k: Dimension of key vectors

### Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
```

Where each head is:
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

## Visualizations

The application provides several visualization types:

1. **Attention Heatmaps**: Show attention weights between all token pairs
2. **Flow Diagrams**: Visualize attention connections between tokens
3. **Statistical Analysis**: Distribution and summary statistics
4. **Individual Head Analysis**: Examine specific attention heads

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest test_attention.py -v

# Run specific test class
python -m pytest test_attention.py::TestScaledDotProductAttention -v

# Run with coverage
python -m pytest test_attention.py --cov=0128 --cov-report=html
```

## Educational Resources

### Understanding Attention Mechanisms

1. **"Attention Is All You Need"** - Vaswani et al. (2017)
2. **"The Illustrated Transformer"** - Jay Alammar
3. **"Attention? Attention!"** - Lilian Weng

### Interactive Learning

Use the web interface to:
- Experiment with different text inputs
- Visualize attention patterns
- Compare single-head vs multi-head attention
- Explore attention statistics

## Advanced Usage

### Custom Attention Implementations

Extend the base classes for custom attention mechanisms:

```python
class CustomAttention(ScaledDotProductAttention):
    def forward(self, Q, K, V, mask=None):
        # Custom attention logic
        return super().forward(Q, K, V, mask)
```

### Integration with Transformers

Use with Hugging Face Transformers:

```python
from transformers import AutoTokenizer, AutoModel
from 0128 import MultiHeadAttention

# Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Extract embeddings and apply attention
embeddings = model.embeddings.word_embeddings(input_ids)
attention_output, weights = multihead_attn(embeddings, embeddings, embeddings)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original Transformer paper by Vaswani et al.
- PyTorch team for the excellent deep learning framework
- Streamlit team for the interactive web framework
- The open-source community for inspiration and contributions

## Related Projects

- [Transformer Implementation](https://github.com/example/transformer)
- [NLP Toolkit](https://github.com/example/nlp-toolkit)
- [Deep Learning Tutorials](https://github.com/example/dl-tutorials)


# Attention-Mechanism-Implementation
