# Character-Level Transformer

A small decoder-only Transformer language model built from scratch in PyTorch.

The model is trained on text from *The Wonderful Wizard of Oz* and generates character-by-character continuations.

## Features

- Character-level tokenization
- Causal self-attention
- Transformer blocks
- Training and validation loss
- Temperature and top-k sampling
- CPU, CUDA, and Apple MPS support

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Place `wizard_of_oz.txt` in the project folder, then run:

```bash
python character_level_transformer.py
```

## Limitations

This is a small educational model. It learns text patterns and writing style but does not truly understand plot, facts, or reasoning.