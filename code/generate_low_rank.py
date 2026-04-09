import torch
import numpy as np
def generate_low_rank_matrix_with_custom_rank(
    input_dim, output_dim, rank,
    target_type="low_rank_identity",
    initialization_matrix_A=None
):
    pass
    from config_linear import training_config
    device = training_config["device"]
    scale = training_config.get("scale", 1.0)
    effective_rank = min(rank, input_dim, output_dim)
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    if target_type == "low_rank_identity":
        matrix = torch.zeros((output_dim, input_dim))
        identity_matrix = torch.eye(effective_rank)
        matrix[:effective_rank, :effective_rank] = identity_matrix
        actual_rank = torch.linalg.matrix_rank(matrix).item()
        matrix = matrix.to(device) * scale
    elif target_type == "aligned_matrix":
        if initialization_matrix_A is None:
            raise ValueError("Invalid argument.")
        if isinstance(initialization_matrix_A, torch.Tensor):
            A_np = initialization_matrix_A.cpu().numpy()
        else:
            A_np = initialization_matrix_A
        from scipy.linalg import svd
        U, Sigma, Vt = svd(A_np, full_matrices=False)
        U_reduced = U[:, :effective_rank]
        Vt_reduced = Vt[:effective_rank, :]
        matrix_np = U_reduced @ Vt_reduced
        matrix = torch.from_numpy(matrix_np).float()
        actual_rank = torch.linalg.matrix_rank(matrix).item()
        matrix = matrix.to(device) * scale
    else:
        raise ValueError("Invalid argument.")
    print(f"{'='*80}\n")
    return matrix
