# Basic Transformer For next character prediction

This repository contains a minimal character-level Transformer (decoder-only) implemented in JAX/Flax for next-character prediction. 
See GitHub repository at https://github.com/Kaishoo-git/dsa4212_assignment


Repository structure
--------------------

Top-level layout:

- `transformer.ipynb` - Code reference provided.
- `models.py` - Minimal, decoder-only Transformer implementation (token & positional embeddings, DecoderBlocks, MLP, weight tying, causal attention). Moved into root directory, remains unchanged.
- `text8_train/test.txt` a preprocessed `text8_dataset` used in the notebook. Moved into root directory, remains unchanged.

- `transformer_expanded.ipynb` - Primary Jupyter notebook used for experimenting, training and generation. The notebook contains data loading, model initialization, training loop, and a JITted token generator cell.
- `DSA4212_Group_Project.pdf` - Results and report. 

Code additions are made only in `transformer_expanded.ipynb`, modules are untouched.
All runtime measurements were obtained using a Google Colab environment with an NVIDIA T4 GPU.

All changes are made in `transformer_expanded.ipynb`. Refer to `DSA4212_Group_Project.pdf` for results.