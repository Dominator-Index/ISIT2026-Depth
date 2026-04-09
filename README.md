# Depth, Not Data: An Analysis of Hessian Spectral Bifurcation

This repository contains the official code for the paper [Depth, Not Data: An Analysis of Hessian Spectral Bifurcation](https://arxiv.org/abs/2602.00545).

## About

We prove that the Hessian spectral bifurcation (bulk-and-spike structure) in deep linear networks arises intrinsically from network depth, independent of data imbalance, with the dominant-to-bulk eigenvalue ratio scaling linearly with depth $L$.

## How to Run

```bash
cd code
python train.py
```

## Configuration (`code/config_linear.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `0.01` | SGD learning rate |
| `steps` | `10000` | Total training steps |
| `num_layer` | `2` | Network depth (L) |
| `input_dim` | `16` | Input dimension d₀ |
| `hidden_dim` | `32` | Hidden layer width |
| `output_dim` | `10` | Output dimension d_L |
| `rank` | `5` | Rank of the target matrix |
| `variance` | `0.01` | Initialization variance |
| `device` | `"cuda:3"` | GPU device |
| `target_type` | `"aligned_matrix"` | `"aligned_matrix"` or `"low_rank_identity"` |
| `method` | `"adaptive_threshold"` | Spectral space identification: `"adaptive_threshold"`, `"gap"`, `"log_gap"`, `"threshold"` |
| `record_steps` | every 100 steps | Steps at which eigenvalues are recorded |
| `hessian_analysis_steps` | every 50 steps (0–1000) | Steps at which full Hessian analysis is run |
| `use_activation` | `False` | Whether to use nonlinear activation |
| `activation_type` | `"relu"` | Activation function: `"relu"`, `"tanh"`, `"sigmoid"` |
| `top_k_pca_number` | `10000` | Number of top eigenvalues to compute (reduce if GPU memory is limited) |
