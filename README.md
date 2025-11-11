# My LLM Journey

A hands-on learning repository for building Large Language Models (LLMs) from scratch using PyTorch. This project progressively explores the fundamental concepts and building blocks of transformer-based language models.

## Repository Structure

### Chapter 2: Text Preprocessing and Embeddings

#### `chap2/intro.py`
Introduction to tokenization with two approaches:
- **SimpleTokenizerV1**: A custom tokenizer built from scratch using regex-based splitting
  - Handles special tokens (`<|UNK|>`, `<|endoftext|>`)
  - Implements `encode()` and `decode()` methods
- **tiktoken**: Using OpenAI's BPE tokenizer (GPT-2 encoding)

#### `chap2/embeddings.py`
Building the embedding layer for language models:
- **GPTDatasetV1**: Custom PyTorch Dataset for creating training samples
  - Implements sliding window approach with configurable stride
  - Creates input-target pairs for next-token prediction
- **Token Embeddings**: Converting token IDs to dense vectors (vocab_size=50257, dim=256)
- **Position Embeddings**: Adding positional information to token embeddings
- **DataLoader**: Batch processing with configurable parameters

### Chapter 3: Attention Mechanisms

#### `chap3/basic-self-attention.py`
Simplified self-attention without trainable parameters:
- Computing attention scores using dot products between input vectors
- Applying softmax to get attention weights
- Computing context vectors as weighted sums
- Efficient matrix operations for all tokens simultaneously

#### `chap3/self-attention.py`
Full self-attention implementation with learnable weights:
- **Query, Key, Value matrices**: Trainable linear transformations
- **Scaled Dot-Product Attention**: Attention scores scaled by √(d_k)
- **SelfAttention Module**: PyTorch nn.Module implementation
  - Configurable input/output dimensions
  - Optional QKV bias parameters

#### `chap3/causal-attention.py`
Causal (masked) attention for autoregressive language modeling:
- **CausalAttention Module**: Prevents tokens from attending to future positions
- **Masking**: Upper triangular mask to enforce causality
- **Dropout**: Regularization on attention weights
- **Batch Processing**: Handles 3D tensors (batch_size, seq_len, d_model)

## Notes

- The examples use simple toy data for demonstration purposes
- Real implementations would require larger datasets and training loops
- This serves as educational code for understanding transformer architectures
