import torch
from hessian_utils import compute_hessian_eigenvalues_pyhessian_fixed, compute_hessian_eigen
from Top_k_Dom_search import search_top_k_dominant_bulk_space
def pre_compute_and_analyze_eigenvalues(model, loss_function, single_loader, top_k, device, config, step):
    pass
    try:
        eigenvalues, eigenvectors = compute_hessian_eigenvalues_pyhessian_fixed(
            model=model,
            criterion=loss_function,
            data_loader=single_loader,
            top_k=top_k,
            device=device
        )
        is_descending = torch.all(eigenvalues[:-1] >= eigenvalues[1:])
        eigenvalues_cpu_list = eigenvalues.cpu().tolist()
        if config["method"] in ['gap', 'adaptive_threshold']:
            result = search_top_k_dominant_bulk_space(eigenvalues_cpu_list, method=config["method"])
            dominant_dim = result['dominant_k']
            gaps = result['gaps']
            if result.get('second_max_gap_index') is not None:
                pass
        else:
            dominant_dim, gaps = search_top_k_dominant_bulk_space(eigenvalues_cpu_list, method=config["method"])
            result = {'dominant_k': dominant_dim, 'gaps': gaps}
        eigenvectors_dominant = eigenvectors[:, :dominant_dim]
        eigenvalues_dominant = eigenvalues[:dominant_dim]
        return eigenvalues, eigenvectors_dominant, dominant_dim, gaps, True
    except Exception as e:
        import traceback
        return None, None, None, None, False
def compute_and_analyze_eigenvalues(model, loss_function, single_loader, top_k, device, config, step):
    pass
    try:
        for data, target in single_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = 0.5 * loss_function(output, target)
            break
        eigenvalues, eigenvectors = compute_hessian_eigen(
            loss=loss,
            params=model.parameters(),
            top_k=top_k,
            device=device,
            return_numpy=False
        )
        is_descending = torch.all(eigenvalues[:-1] >= eigenvalues[1:])
        eigenvalues_cpu_list = eigenvalues.cpu().tolist()
        if config["method"] in ['gap', 'adaptive_threshold']:
            result = search_top_k_dominant_bulk_space(eigenvalues_cpu_list, method=config["method"])
            dominant_dim = result['dominant_k']
            gaps = result['gaps']
            if result.get('second_max_gap_index') is not None:
                pass
        else:
            dominant_dim, gaps = search_top_k_dominant_bulk_space(eigenvalues_cpu_list, method=config["method"])
            result = {'dominant_k': dominant_dim, 'gaps': gaps}
        eigenvectors_dominant = eigenvectors[:, :dominant_dim]
        eigenvalues_dominant = eigenvalues[:dominant_dim]
        return eigenvalues, eigenvectors, dominant_dim, gaps, True
    except Exception as e:
        import traceback
        return None, None, None, None, False
def compute_and_analyze_eigenvalues_with_dom_bulk(model, loss_function, single_loader, top_k, device, config, step):
    pass
    try:
        eigenvalues, eigenvectors = compute_hessian_eigenvalues_pyhessian_fixed(
            model=model,
            criterion=loss_function,
            data_loader=single_loader,
            top_k=top_k,
            device=device
        )
        is_descending = torch.all(eigenvalues[:-1] >= eigenvalues[1:])
        eigenvalues_cpu_list = eigenvalues.cpu().tolist()
        result = search_top_k_dominant_bulk_space(eigenvalues_cpu_list, method=config["method"])
        dominant_dim = result['dominant_k']
        bulk_start = result['bulk_start']
        bulk_end = result['bulk_end']
        gaps = result['gaps']
        eigenvalues_dominant = eigenvalues[:dominant_dim]
        eigenvectors_dominant = eigenvectors[:, :dominant_dim]
        if bulk_start < bulk_end:
            eigenvalues_bulk = eigenvalues[bulk_start:bulk_end]
            eigenvectors_bulk = eigenvectors[:, bulk_start:bulk_end]
            bulk_dim = bulk_end - bulk_start
        else:
            eigenvalues_bulk = torch.tensor([], device=device)
            eigenvectors_bulk = torch.tensor([], device=device).reshape(eigenvectors.shape[0], 0)
            bulk_dim = 0
        if result.get('second_max_gap_value') is not None:
            pass
        return {
            'eigenvalues_all': eigenvalues,
            'eigenvectors_all': eigenvectors,
            'eigenvalues_dominant': eigenvalues_dominant,
            'eigenvectors_dominant': eigenvectors_dominant,
            'eigenvalues_bulk': eigenvalues_bulk,
            'eigenvectors_bulk': eigenvectors_bulk,
            'dominant_dim': dominant_dim,
            'bulk_dim': bulk_dim,
            'bulk_start': bulk_start,
            'bulk_end': bulk_end,
            'gaps': gaps,
            'space_analysis': result,
            'success': True
        }
    except Exception as e:
        import traceback
        return {
            'eigenvalues_all': None,
            'eigenvectors_all': None,
            'eigenvalues_dominant': None,
            'eigenvectors_dominant': None,
            'eigenvalues_bulk': None,
            'eigenvectors_bulk': None,
            'dominant_dim': None,
            'bulk_dim': None,
            'bulk_start': None,
            'bulk_end': None,
            'gaps': None,
            'space_analysis': None,
            'success': False
        }
def compute_and_analyze_eigenvalues_manual(model, loss, top_k, device, config, step):
    pass
    for i, eigenval in enumerate(eigenvalues):
        if isinstance(eigenval, torch.Tensor):
            raw_eigenval = eigenval.cpu().item()
        else:
            raw_eigenval = float(eigenval)
        eigenvalue_history[f"top_{i+1}"].append(raw_eigenval)
def prepare_swanlab_log(eigenvalues, dominant_dim):
    pass
    hessian_eigenvals = {}
    for i, eigenval in enumerate(eigenvalues):
        if isinstance(eigenval, torch.Tensor):
            raw_eigenval = eigenval.cpu().item()
        else:
            raw_eigenval = float(eigenval)
        hessian_eigenvals[f"Raw_Hessian_Eigenvalues/Top_{i+1}"] = raw_eigenval
    swanlab_data = {
        "Dominant_Dimension": dominant_dim
    }
    swanlab_data.update(hessian_eigenvals)
    return swanlab_data
def collect_space_analysis_data(space_history, space_analysis):
    pass
    space_history['dominant_dims'].append(space_analysis['dominant_k'])
    space_history['bulk_starts'].append(space_analysis['bulk_start'])
    space_history['bulk_ends'].append(space_analysis['bulk_end'])
    space_history['bulk_sizes'].append(space_analysis['bulk_end'] - space_analysis['bulk_start'])
    space_history['max_gap_values'].append(space_analysis['max_gap_value'])
    space_history['max_gap_indices'].append(space_analysis['max_gap_index'])
    if space_analysis.get('second_max_gap_value') is not None:
        space_history['second_max_gap_values'].append(space_analysis['second_max_gap_value'])
        space_history['second_max_gap_indices'].append(space_analysis['second_max_gap_index'])
    else:
        space_history['second_max_gap_values'].append(None)
        space_history['second_max_gap_indices'].append(None)
