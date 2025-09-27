"""
Mock database for attention mechanism demonstrations
Contains realistic text data for various NLP tasks
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import random

class MockTextDatabase:
    """Mock database containing text data for attention mechanism demonstrations"""
    
    def __init__(self, db_path: str = "attention_demo.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                tokens TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                sentence TEXT NOT NULL,
                tokens TEXT,
                position INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attention_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                expected_attention TEXT,
                description TEXT
            )
        ''')
        
        # Insert sample data
        self._insert_sample_documents(cursor)
        self._insert_sample_sentences(cursor)
        self._insert_attention_examples(cursor)
        
        conn.commit()
        conn.close()
    
    def _insert_sample_documents(self, cursor):
        """Insert sample documents"""
        documents = [
            {
                "title": "Understanding Attention Mechanisms",
                "content": "Attention mechanisms have revolutionized natural language processing. They allow models to focus on relevant parts of input when generating output. The transformer architecture relies heavily on self-attention and multi-head attention mechanisms.",
                "category": "technical"
            },
            {
                "title": "Machine Learning Applications",
                "content": "Machine learning is transforming industries across the globe. From healthcare to finance, ML algorithms are solving complex problems. Deep learning models, especially transformers, are achieving state-of-the-art results in many tasks.",
                "category": "general"
            },
            {
                "title": "Neural Network Architectures",
                "content": "Neural networks come in many forms: feedforward, recurrent, convolutional, and transformer architectures. Each has unique strengths for different types of data and tasks. The choice of architecture significantly impacts model performance.",
                "category": "technical"
            },
            {
                "title": "Natural Language Processing",
                "content": "NLP enables computers to understand and process human language. Recent advances in transformer models have dramatically improved performance on tasks like translation, summarization, and question answering.",
                "category": "technical"
            },
            {
                "title": "The Future of AI",
                "content": "Artificial intelligence continues to evolve rapidly. New architectures and training methods are constantly being developed. The future holds promise for even more capable and efficient AI systems.",
                "category": "general"
            }
        ]
        
        for doc in documents:
            tokens = doc["content"].split()
            cursor.execute('''
                INSERT INTO documents (title, content, category, tokens)
                VALUES (?, ?, ?, ?)
            ''', (doc["title"], doc["content"], doc["category"], json.dumps(tokens)))
    
    def _insert_sample_sentences(self, cursor):
        """Insert sample sentences"""
        sentences = [
            (1, "Attention mechanisms have revolutionized natural language processing.", 0),
            (1, "They allow models to focus on relevant parts of input when generating output.", 1),
            (1, "The transformer architecture relies heavily on self-attention and multi-head attention mechanisms.", 2),
            (2, "Machine learning is transforming industries across the globe.", 0),
            (2, "From healthcare to finance, ML algorithms are solving complex problems.", 1),
            (2, "Deep learning models, especially transformers, are achieving state-of-the-art results in many tasks.", 2),
            (3, "Neural networks come in many forms: feedforward, recurrent, convolutional, and transformer architectures.", 0),
            (3, "Each has unique strengths for different types of data and tasks.", 1),
            (3, "The choice of architecture significantly impacts model performance.", 2),
            (4, "NLP enables computers to understand and process human language.", 0),
            (4, "Recent advances in transformer models have dramatically improved performance on tasks like translation, summarization, and question answering.", 1),
            (5, "Artificial intelligence continues to evolve rapidly.", 0),
            (5, "New architectures and training methods are constantly being developed.", 1),
            (5, "The future holds promise for even more capable and efficient AI systems.", 2)
        ]
        
        for doc_id, sentence, position in sentences:
            tokens = sentence.split()
            cursor.execute('''
                INSERT INTO sentences (document_id, sentence, tokens, position)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, sentence, json.dumps(tokens), position))
    
    def _insert_attention_examples(self, cursor):
        """Insert attention mechanism examples"""
        examples = [
            {
                "query": "What is attention?",
                "key": "Attention mechanisms focus on relevant input parts",
                "value": "Attention allows models to selectively focus on important information",
                "expected_attention": "high",
                "description": "Query about attention mechanism definition"
            },
            {
                "query": "How do transformers work?",
                "key": "Transformers use self-attention mechanisms",
                "value": "Transformers process sequences in parallel using attention",
                "expected_attention": "high",
                "description": "Query about transformer architecture"
            },
            {
                "query": "What is machine learning?",
                "key": "Machine learning algorithms learn from data",
                "value": "ML enables computers to learn patterns without explicit programming",
                "expected_attention": "medium",
                "description": "General ML query"
            }
        ]
        
        for example in examples:
            cursor.execute('''
                INSERT INTO attention_examples (query, key, value, expected_attention, description)
                VALUES (?, ?, ?, ?, ?)
            ''', (example["query"], example["key"], example["value"], 
                  example["expected_attention"], example["description"]))
    
    def get_documents(self, category: str = None) -> List[Dict[str, Any]]:
        """Retrieve documents from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('SELECT * FROM documents WHERE category = ?', (category,))
        else:
            cursor.execute('SELECT * FROM documents')
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_sentences(self, document_id: int = None) -> List[Dict[str, Any]]:
        """Retrieve sentences from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if document_id:
            cursor.execute('SELECT * FROM sentences WHERE document_id = ? ORDER BY position', (document_id,))
        else:
            cursor.execute('SELECT * FROM sentences ORDER BY document_id, position')
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_attention_examples(self) -> List[Dict[str, Any]]:
        """Retrieve attention examples from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM attention_examples')
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_random_sentence_pairs(self, num_pairs: int = 5) -> List[Dict[str, str]]:
        """Get random sentence pairs for attention demonstration"""
        sentences = self.get_sentences()
        pairs = []
        
        for _ in range(num_pairs):
            sent1 = random.choice(sentences)
            sent2 = random.choice(sentences)
            pairs.append({
                "sentence1": sent1["sentence"],
                "sentence2": sent2["sentence"],
                "tokens1": json.loads(sent1["tokens"]),
                "tokens2": json.loads(sent2["tokens"])
            })
        
        return pairs
    
    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search documents by content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM documents 
            WHERE content LIKE ? OR title LIKE ?
        ''', (f'%{query}%', f'%{query}%'))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

def create_sample_embeddings(text: str, d_model: int = 64) -> torch.Tensor:
    """Create sample embeddings for demonstration purposes"""
    import torch
    tokens = text.split()
    seq_len = len(tokens)
    
    # Create random embeddings (in practice, these would come from a trained model)
    torch.manual_seed(hash(text) % 2**32)  # Deterministic based on text
    embeddings = torch.randn(1, seq_len, d_model)
    
    return embeddings, tokens

if __name__ == "__main__":
    # Initialize database
    db = MockTextDatabase()
    
    print("📚 Mock Text Database Initialized")
    print("=" * 40)
    
    # Show sample data
    documents = db.get_documents()
    print(f"📄 Documents: {len(documents)}")
    for doc in documents[:3]:
        print(f"  - {doc['title']} ({doc['category']})")
    
    sentences = db.get_sentences()
    print(f"\n📝 Sentences: {len(sentences)}")
    for sent in sentences[:3]:
        print(f"  - {sent['sentence']}")
    
    examples = db.get_attention_examples()
    print(f"\n🎯 Attention Examples: {len(examples)}")
    for ex in examples:
        print(f"  - Q: {ex['query']}")
        print(f"    A: {ex['value']}")
    
    # Test random pairs
    pairs = db.get_random_sentence_pairs(2)
    print(f"\n🔀 Random Sentence Pairs:")
    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}:")
        print(f"    S1: {pair['sentence1']}")
        print(f"    S2: {pair['sentence2']}")
