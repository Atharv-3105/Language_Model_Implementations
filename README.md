# Language Model Implementations

A repository containing from-scratch implementations of modern language-model architectures and their core building blocks. The purpose of this repository is to understand the architecture at implementation level instead of treating the model as a black box.

The implementations are written primarily in **PyTorch** and are organized as separate experiments/notebooks for individual architectures.

## What is implemented

### LLaMA 2

The `llama/` implementation follows the LLaMA-style decoder-only architecture and includes:

- Model configuration through a `ModelArg` dataclass
- Token embeddings and output projection
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Grouped-Query Attention (GQA) / configurable KV heads
- SwiGLU-style feed-forward network
- Residual Transformer blocks
- Key/Value caching for autoregressive inference
- KV-head repetition to match query heads

The implementation also contains the model paper and a `download.sh` helper alongside the model notebook.

### Gemma 3 270M

The `Gemma-3-270M/` implementation reconstructs the architecture in PyTorch and explicitly implements several of its important building blocks:

- Gated feed-forward network using GELU gating
- RMSNorm with Gemma-style scale parameterization
- Rotary Positional Embeddings
- Grouped-Query Attention
- Optional Query/Key normalization
- Configurable query pre-attention scaling
- Causal attention masking
- Transformer blocks
- Configurable base/instruct model selection
- Hugging Face Hub / tokenizer integration

The notebook also includes architecture diagrams and experiments around the implementation.

### Qwen 3 0.6B

The repository contains a dedicated `Qwen-3-0.6B/` implementation directory with the model notebook, tokenizer, requirements, model artifacts, and paper reference. These files use Git LFS in the repository.

### Small-LM

The `Small-LM/` directory contains a smaller language-model implementation and its supporting experiments. The repository includes:

- Transformer building blocks
- Multi-Head Attention
- Feed-forward network
- BPE tokenizer artifacts
- Training checkpoints
- Training notebook
- W&B experiment artifacts
- Model visualizations and intermediate artifacts

## Core building blocks explored

Across the implementations, the repository explores the components that repeatedly appear in modern language models:

- Tokenization and BPE
- Token embeddings
- Self-attention
- Multi-Head Attention (MHA)
- Grouped-Query Attention (GQA)
- Query/Key normalization
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU / gated feed-forward networks
- Residual connections
- Causal masking
- Key/Value caching
- Transformer blocks
- Autoregressive language-model structure

## LLaMA-style inference flow

For autoregressive generation, the implementation follows the general flow:

```text
Input tokens
    |
    v
Token Embeddings
    |
    v
Transformer Blocks
    |
    +--> RMSNorm
    |
    +--> Self Attention
    |      |
    |      +--> Q / K / V projections
    |      +--> RoPE
    |      +--> KV Cache
    |      +--> Attention
    |
    +--> RMSNorm
    |
    +--> SwiGLU Feed Forward
    |
    v
Final normalization / output projection
    |
    v
Next-token logits
```

## Why this repository exists

The main objective is not to train a production-scale LLM. The objective is to understand how modern language-model architectures are constructed internally and how individual architectural decisions affect the implementation.

Implementing the components manually makes details such as tensor shapes, attention-head layouts, positional encoding, normalization, KV caching, and feed-forward gating explicit.

## Repository structure

```text
Language_Model_Implementations/
│
├── Gemma-3-270M/
│   ├── model.ipynb
│   └── imgs/
│
├── Qwen-3-0.6B/
│   ├── model.ipynb
│   ├── requirements.txt
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── paper.pdf
│
├── Small-LM/
│   ├── model.ipynb
│   ├── small-lm_trained.ipynb
│   ├── saves/
│   └── wandb.zip
│
└── llama/
    ├── model.ipynb
    ├── Llama2_Paper.pdf
    └── download.sh
```

## Technology

- Python
- PyTorch
- Hugging Face Hub
- Hugging Face Tokenizers
- Jupyter Notebook
- Weights & Biases (for training experiments)

## Learning focus

This repository is intended as a reference for understanding modern LLM implementations from the inside out:

1. Understand the paper / architecture.
2. Break the architecture into independent components.
3. Implement each component in PyTorch.
4. Verify tensor shapes and data flow.
5. Assemble the Transformer blocks.
6. Add model-specific details such as RoPE, GQA, normalization and KV caching.
7. Run experiments and inspect the resulting model behaviour.

## Status

This is a research and learning repository. The implementations are primarily educational reproductions rather than production-ready replacements for the original model implementations.
