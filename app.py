"""
Interactive Web UI for Attention Mechanism Demonstration
Built with Streamlit for easy exploration and visualization
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple
import json

# Import our attention mechanisms
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

from mock_database import MockTextDatabase, create_sample_embeddings

# Page configuration
st.set_page_config(
    page_title="Attention Mechanism Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def create_attention_heatmap(attention_weights: torch.Tensor, 
                           tokens: List[str], 
                           title: str = "Attention Weights") -> go.Figure:
    """Create interactive attention heatmap using Plotly"""
    
    # Convert to numpy and handle multi-head attention
    if attention_weights.dim() == 4:  # Multi-head attention
        weights = attention_weights.mean(dim=1).squeeze().detach().numpy()
    else:  # Single-head attention
        weights = attention_weights.squeeze().detach().numpy()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=weights,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        hoverongaps=False,
        text=np.round(weights, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Attention Weight")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Key Tokens",
        yaxis_title="Query Tokens",
        width=600,
        height=500,
        font=dict(size=12)
    )
    
    return fig

def create_attention_flow_diagram(tokens: List[str], attention_weights: torch.Tensor) -> go.Figure:
    """Create attention flow diagram showing connections between tokens"""
    
    if attention_weights.dim() == 4:
        weights = attention_weights.mean(dim=1).squeeze().detach().numpy()
    else:
        weights = attention_weights.squeeze().detach().numpy()
    
    # Create nodes and edges
    nodes = []
    edges = []
    
    for i, token in enumerate(tokens):
        nodes.append({
            'id': i,
            'label': token,
            'x': i,
            'y': 0
        })
    
    # Add edges for significant attention weights
    threshold = 0.1
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if weights[i, j] > threshold:
                edges.append({
                    'source': i,
                    'target': j,
                    'weight': weights[i, j],
                    'color': f'rgba(31, 119, 180, {weights[i, j]})'
                })
    
    fig = go.Figure()
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(size=50, color='lightblue'),
            text=node['label'],
            textposition="middle center",
            name=node['label'],
            showlegend=False
        ))
    
    # Add edges
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[edge['source'], edge['target']],
            y=[0, 0],
            mode='lines',
            line=dict(width=edge['weight']*10, color=edge['color']),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="Attention Flow Between Tokens",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=200,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Attention Mechanism Explorer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    d_model = st.sidebar.slider("Model Dimension", 32, 512, 64, 32)
    num_heads = st.sidebar.slider("Number of Heads", 1, 16, 8, 1)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.05)
    
    # Text input options
    st.sidebar.subheader("Text Input")
    input_mode = st.sidebar.radio(
        "Choose input mode:",
        ["Custom Text", "Sample from Database", "Random Generation"]
    )
    
    # Initialize database
    if 'db' not in st.session_state:
        st.session_state.db = MockTextDatabase()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">📝 Input Text</div>', unsafe_allow_html=True)
        
        if input_mode == "Custom Text":
            input_text = st.text_area(
                "Enter your text:",
                value="The quick brown fox jumps over the lazy dog",
                height=100
            )
        elif input_mode == "Sample from Database":
            documents = st.session_state.db.get_documents()
            doc_options = [f"{doc['title']} ({doc['category']})" for doc in documents]
            selected_doc = st.selectbox("Select a document:", doc_options)
            
            if selected_doc:
                doc_idx = doc_options.index(selected_doc)
                input_text = documents[doc_idx]['content']
            else:
                input_text = "The quick brown fox jumps over the lazy dog"
        else:  # Random Generation
            sentences = st.session_state.db.get_sentences()
            if st.button("Generate Random Text"):
                random_sentences = np.random.choice(sentences, size=3, replace=False)
                input_text = " ".join([sent['sentence'] for sent in random_sentences])
            else:
                input_text = "The quick brown fox jumps over the lazy dog"
        
        st.text_area("Current text:", value=input_text, height=100, disabled=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Text Statistics</div>', unsafe_allow_html=True)
        
        tokens = input_text.split()
        st.metric("Word Count", len(tokens))
        st.metric("Character Count", len(input_text))
        st.metric("Average Word Length", f"{np.mean([len(word) for word in tokens]):.1f}")
    
    # Process the text
    if st.button("🚀 Process with Attention Mechanisms", type="primary"):
        
        # Create embeddings
        embeddings, tokens = create_sample_embeddings(input_text, d_model)
        
        # Initialize attention mechanisms
        config = AttentionConfig(d_model=d_model, num_heads=num_heads, dropout=dropout)
        
        single_head_attn = ScaledDotProductAttention(dropout)
        multi_head_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Process with single-head attention
        with st.spinner("Processing with single-head attention..."):
            sh_output, sh_weights = single_head_attn(embeddings, embeddings, embeddings)
        
        # Process with multi-head attention
        with st.spinner("Processing with multi-head attention..."):
            mh_output, mh_weights = multi_head_attn(embeddings, embeddings, embeddings)
        
        # Display results
        st.markdown('<div class="section-header">🎯 Attention Results</div>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Heatmaps", "🔄 Flow Diagrams", "📈 Analysis", "🔍 Details"])
        
        with tab1:
            st.subheader("Single-Head Attention Heatmap")
            fig1 = create_attention_heatmap(sh_weights, tokens, "Single-Head Attention")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("Multi-Head Attention Heatmap (Averaged)")
            fig2 = create_attention_heatmap(mh_weights, tokens, "Multi-Head Attention")
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Single-Head Attention Flow")
            fig3 = create_attention_flow_diagram(tokens, sh_weights)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.subheader("Multi-Head Attention Flow")
            fig4 = create_attention_flow_diagram(tokens, mh_weights)
            st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Attention Statistics")
                
                # Calculate statistics
                sh_weights_np = sh_weights.squeeze().detach().numpy()
                mh_weights_np = mh_weights.mean(dim=1).squeeze().detach().numpy()
                
                stats_data = {
                    'Metric': ['Max Attention', 'Min Attention', 'Mean Attention', 'Std Attention'],
                    'Single-Head': [
                        f"{sh_weights_np.max():.3f}",
                        f"{sh_weights_np.min():.3f}",
                        f"{sh_weights_np.mean():.3f}",
                        f"{sh_weights_np.std():.3f}"
                    ],
                    'Multi-Head': [
                        f"{mh_weights_np.max():.3f}",
                        f"{mh_weights_np.min():.3f}",
                        f"{mh_weights_np.mean():.3f}",
                        f"{mh_weights_np.std():.3f}"
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.subheader("Attention Distribution")
                
                # Create histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=sh_weights_np.flatten(),
                    name='Single-Head',
                    opacity=0.7,
                    nbinsx=20
                ))
                fig_hist.add_trace(go.Histogram(
                    x=mh_weights_np.flatten(),
                    name='Multi-Head',
                    opacity=0.7,
                    nbinsx=20
                ))
                
                fig_hist.update_layout(
                    title="Attention Weight Distribution",
                    xaxis_title="Attention Weight",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab4:
            st.subheader("Detailed Attention Weights")
            
            # Single-head attention weights table
            st.write("**Single-Head Attention Weights:**")
            sh_df = pd.DataFrame(
                sh_weights_np,
                index=tokens,
                columns=tokens
            )
            st.dataframe(sh_df.round(3), use_container_width=True)
            
            # Multi-head attention weights (averaged)
            st.write("**Multi-Head Attention Weights (Averaged):**")
            mh_df = pd.DataFrame(
                mh_weights_np,
                index=tokens,
                columns=tokens
            )
            st.dataframe(mh_df.round(3), use_container_width=True)
            
            # Show individual heads
            if num_heads > 1:
                st.write("**Individual Head Attention Weights:**")
                head_idx = st.selectbox("Select head to view:", range(num_heads))
                
                head_weights = mh_weights[0, head_idx].detach().numpy()
                head_df = pd.DataFrame(
                    head_weights,
                    index=tokens,
                    columns=tokens
                )
                st.dataframe(head_df.round(3), use_container_width=True)
    
    # Educational section
    st.markdown('<div class="section-header">📚 Understanding Attention Mechanisms</div>', unsafe_allow_html=True)
    
    with st.expander("What is Attention?"):
        st.markdown("""
        **Attention mechanisms** allow neural networks to focus on specific parts of the input when generating each part of the output. 
        
        Key concepts:
        - **Query (Q)**: What the model is looking for
        - **Key (K)**: What the model is comparing against
        - **Value (V)**: The actual information being retrieved
        
        The attention mechanism computes a weighted sum of values, where the weights are determined by the compatibility between queries and keys.
        """)
    
    with st.expander("Single-Head vs Multi-Head Attention"):
        st.markdown("""
        **Single-Head Attention:**
        - Uses one attention function
        - Simpler but limited representation capacity
        
        **Multi-Head Attention:**
        - Uses multiple attention heads in parallel
        - Each head can focus on different aspects of the input
        - Provides richer representations
        - Outputs are concatenated and linearly transformed
        """)
    
    with st.expander("How to Read the Visualizations"):
        st.markdown("""
        **Heatmaps:**
        - Rows represent query tokens
        - Columns represent key tokens
        - Color intensity shows attention weight
        - Brighter colors = higher attention
        
        **Flow Diagrams:**
        - Shows connections between tokens
        - Line thickness indicates attention strength
        - Helps visualize which tokens attend to each other
        """)

if __name__ == "__main__":
    main()
