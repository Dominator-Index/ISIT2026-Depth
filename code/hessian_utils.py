import torch
import numpy as np
from config_linear import device
from pyhessian import hessian
def compute_hessian_eigen(loss, params, top_k=832, device=None, return_numpy=True):
    pass
    if return_numpy:
        eigenvalues_tensor, eigenvectors_tensor = compute_hessian_eigen_pure_cuda(loss, params, top_k, device)
        return eigenvalues_tensor.cpu().numpy(), eigenvectors_tensor.cpu().numpy()
    else:
        return compute_hessian_eigen_pure_cuda(loss, params, top_k, device)
def compute_hessian_eigen_pure_cuda(loss, params, top_k=832, device=None):
    pass
    import torch
    if device is None:
        device = loss.device
    params = list(params)
    params = [p.to(device) if p.device != device else p for p in params]
    grads_flat = []
    for p in params:
        g = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)[0]
        grads_flat.append(g.view(-1))
    grads_flat = torch.cat(grads_flat).to(device)
    total_params = grads_flat.size(0)
    hessian_matrix = torch.zeros((total_params, total_params),
                                device=device, dtype=torch.float32)
    for i, g in enumerate(grads_flat):
        if i % max(1, total_params // 10) == 0:
            pass
        hessian_row = torch.autograd.grad(
            outputs=g,
            inputs=params,
            retain_graph=True,
            allow_unused=True
        )
        hessian_row_flat = []
        for h, p in zip(hessian_row, params):
            if h is None:
                h_flat = torch.zeros_like(p, device=device).view(-1)
            else:
                h_flat = h.to(device).view(-1)
            hessian_row_flat.append(h_flat)
        hessian_matrix[i] = torch.cat(hessian_row_flat)
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian_matrix)
    sorted_indices = torch.argsort(-eigenvalues)
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    eigenvalues_topk = eigenvalues_sorted[:top_k] if top_k < len(eigenvalues_sorted) else eigenvalues_sorted
    eigenvectors_topk = eigenvectors_sorted[:, :top_k] if top_k < eigenvectors_sorted.shape[1] else eigenvectors_sorted
    return eigenvalues_topk, eigenvectors_topk
def flatten_eigenvector_list(eigenvector_list):
    pass
    flattened_parts = []
    for tensor_part in eigenvector_list:
        if isinstance(tensor_part, torch.Tensor):
            flattened_parts.append(tensor_part.view(-1))
        else:
            flattened_parts.append(torch.tensor(tensor_part).view(-1))
    return torch.cat(flattened_parts)
def process_pyhessian_eigenvectors(eigenvectors_matrix, eigenvalues_list):
    pass
    if isinstance(eigenvectors_matrix, list) and len(eigenvectors_matrix) > 0:
        first_eigenvector = eigenvectors_matrix[0]
        if isinstance(first_eigenvector, list):
            for i, part in enumerate(first_eigenvector):
                pass
            num_eigenvectors = len(eigenvectors_matrix)
            flattened_eigenvectors = []
            for i, eigenvector_list in enumerate(eigenvectors_matrix):
                flattened_vec = flatten_eigenvector_list(eigenvector_list)
                flattened_eigenvectors.append(flattened_vec)
            eigenvectors_tensor = torch.stack(flattened_eigenvectors, dim=1)
        elif isinstance(first_eigenvector, torch.Tensor):
            eigenvectors_tensor = torch.stack(eigenvectors_matrix, dim=1)
        else:
            eigenvectors_tensor = torch.tensor(eigenvectors_matrix, dtype=torch.float32)
    else:
        if isinstance(eigenvectors_matrix, torch.Tensor):
            eigenvectors_tensor = eigenvectors_matrix.clone().detach()
        elif isinstance(eigenvectors_matrix, np.ndarray):
            eigenvectors_tensor = torch.from_numpy(eigenvectors_matrix)
        else:
            eigenvectors_tensor = torch.tensor(eigenvectors_matrix, dtype=torch.float32)
    if isinstance(eigenvalues_list, torch.Tensor):
        eigenvalues_tensor = eigenvalues_list.clone().detach()
    elif isinstance(eigenvalues_list, (list, np.ndarray)):
        eigenvalues_tensor = torch.tensor(eigenvalues_list, dtype=torch.float32)
    else:
        eigenvalues_tensor = torch.tensor(eigenvalues_list, dtype=torch.float32)
    print(f"   eigenvalues_tensor: {eigenvalues_tensor.shape}")
    print(f"   eigenvectors_tensor: {eigenvectors_tensor.shape}")
    return eigenvalues_tensor, eigenvectors_tensor
def compute_hessian_eigen_pyhessian(model, criterion, data_loader, top_k=5, device=device):
    pass
    hessian_computer = hessian(model=model, criterion=criterion, data=data_loader, cuda='cuda')
    hessian_eigen = hessian_computer.eigenvalues(top_n=top_k)
    eigenvalues = np.array(hessian_eigen[0])
    eigenvectors = np.array(hessian_eigen[1])
    return eigenvalues, eigenvectors
def compute_layer_weight_eigenvalues(model, top_k=None):
    pass
    layer_eigenvalues = {}
    layer_eigenvectors = {}
    for name, param in model.named_parameters():
        if name in ['W1', 'W2', 'W3'] and param.dim() == 2:
            weight_matrix = param.detach().cpu().numpy()
            if weight_matrix.shape[0] != weight_matrix.shape[1]:
                gram_matrix = np.dot(weight_matrix, weight_matrix.T)
                eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
            else:
                eigenvals, eigenvecs = np.linalg.eigh(weight_matrix)
            sorted_indices = np.argsort(-eigenvals)
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            if top_k is not None:
                eigenvals = eigenvals[:top_k]
                eigenvecs = eigenvecs[:, :top_k]
            layer_eigenvalues[name] = eigenvals
            layer_eigenvectors[name] = eigenvecs
    return layer_eigenvalues, layer_eigenvectors
def compute_hessian_eigenvalues_pyhessian_fixed(model, criterion, data_loader, top_k=5, device='cuda'):
    pass
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        actual_top_k = min(top_k, total_params)
        if actual_top_k != top_k:
            pass
        hessian_computer = hessian(model=model,
                                  criterion=criterion,
                                  dataloader=data_loader,
                                  cuda=device.type=='cuda' if hasattr(device, 'type') else 'cuda' in str(device),
                                  device=device)
        result = hessian_computer.eigenvalues(
            maxIter=400,
            tol=1e-6,
            top_n=actual_top_k
        )
        if isinstance(result, tuple) and len(result) == 2:
            eigenvalues_list, eigenvectors_matrix = result
        else:
            raise ValueError("Invalid argument.")
        print(f"   eigenvalues_list: {type(eigenvalues_list)}")
        print(f"   eigenvectors_matrix: {type(eigenvectors_matrix)}")
        eigenvalues_tensor, eigenvectors_tensor = process_pyhessian_eigenvectors(
            eigenvectors_matrix, eigenvalues_list
        )
        eigenvalues_tensor = eigenvalues_tensor.to(device)
        eigenvectors_tensor = eigenvectors_tensor.to(device)
        eigenvalues_sorted, sort_indices = torch.sort(eigenvalues_tensor, descending=True)
        eigenvectors_sorted = eigenvectors_tensor[:, sort_indices]
        return eigenvalues_sorted, eigenvectors_sorted
    except Exception as e:
        import traceback
        raise e
def debug_model_parameters(model):
    pass
    total_params = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
    return total_params
def compute_dominant_projection_matrix(top_eigenvectors, k):
    pass
    p = top_eigenvectors.shape[0]
    P_k = np.zeros((p, p))
    for i in range(k):
        u_i = top_eigenvectors[:, i].reshape(-1, 1)
        P_k += np.dot(u_i, u_i.T)
    return torch.from_numpy(P_k).float().to(device)
