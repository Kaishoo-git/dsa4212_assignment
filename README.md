# Basic Transformer For next character prediction

This repository contains a minimal character-level Transformer (decoder-only) implemented in JAX/Flax for next-character prediction. 

Repository structure
--------------------

Top-level layout:

- `transformer.ipynb` - Code reference provided.
- `transformer_expanded.ipynb` - Primary Jupyter notebook used for experimenting, training and generation. The notebook contains data loading, model initialization, training loop, and a JITted token generator cell.
- `models/` - Python package containing the Flax model implementation.
	- `models/models.py` - Minimal, decoder-only Transformer implementation (token & positional embeddings, DecoderBlocks, MLP, weight tying, causal attention).
- `data/` - a preprocessed `text8_dataset` used in the notebook.

Code additions are made only in `transformer_expanded.ipynb`, modules are untouched.
All runtime measurements were obtained using a Google Colab environment with an NVIDIA T4 GPU.

