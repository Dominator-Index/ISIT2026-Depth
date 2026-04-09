import torch
import torch.nn as nn
import numpy as np
import swanlab as wandb
import time
import csv
from datetime import datetime
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, DataLoader
from config_linear import training_config as config
from config_linear import device
from model_linear import LinearNetwork
from generate_low_rank import generate_low_rank_matrix_with_custom_rank
from training_utils import (
    set_seed, format_time, generate_eigenvalue_steps,
    prepare_data_dimensions, print_training_info
)
from eigenvalue_analysis import (
    compute_and_analyze_eigenvalues, collect_eigenvalue_data,
    prepare_swanlab_log
)
from plotting import save_eigenvalue_csv, get_activation_suffix
from hessian_fnorm import compute_hessian_fnorm_exact
from hessian_analysis_plugin import (
    run_complete_hessian_analysis, create_layerwise_optimizer,
    create_analysis_directories
)
from directory_manager import (
    initialize_experiment_directories, create_step_subdirectories,
    get_save_path
)
from weight_matrix_tracker import (
    compute_weight_product_singular_values, collect_weight_matrix_data,
    prepare_weight_swanlab_log, save_weight_matrix_csv
)
def main():
    experiment_dirs = initialize_experiment_directories(config)
    model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], config["num_layer"], config["variance"], device).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    if config.get("use_layerwise_lr", False):
        optimizer = create_layerwise_optimizer(model, config)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    method = "SGD"
    lr = config["learning_rate"]
    var = model.var
    top_k = config["top_k_pca_number"]
    rank = config.get("rank", 5)
    num_layer = config.get("num_layer", 3)
    output_dim = config.get("output_dim", 10)
    eigenvalue_interval = config.get("eigenvalue_interval", 25)
    steps = config["steps"]
    seed = 12138
    enable_hessian_analysis = config.get("enable_hessian_analysis", True)
    hessian_analysis_steps = config.get("hessian_analysis_steps", [])
    print(f"Learning rate: {lr}")
    print(f"Variance: {var}")
    print(f"Computing top {top_k} Hessian eigenvalues")
    print(f"Rank: {rank}")
    if enable_hessian_analysis:
        pass
    selected_steps = config["record_steps"]
    gap_method = config["method"]
    wandb.init(project=config["swanlab_project_name"],
               name=f"Linear{num_layer}+lr{config['layer_learning_rates']['layer_0']}+{method}+outputdim{output_dim}+lr{lr}+var{var:.6f}_rank{rank}_top{top_k}",
               api_key="your_swanlab_key")
    wandb.config.update(config)
    wandb.config.update({
        "optimizer": method,
        "eigenvalue_interval": eigenvalue_interval,
        "target_rank": rank,
        "enable_hessian_analysis": enable_hessian_analysis
    })
    loss_function = nn.MSELoss(reduction='sum')
    total_start_time = time.time()
    print_training_info(steps, [seed], selected_steps)
    set_seed(seed)
    eigenvalue_history = {f"top_{i+1}": [] for i in range(top_k)}
    step_history = []
    dominant_dim_history = []
    eigenvalue_step_history = []
    loss_history = []
    hessian_analysis_history = []
    weight_matrix_history = {
        'weight_singular_values': [],
        'weight_sigma_power': [],
    }
    weight_matrix_step_history = []
    dummmy_input = torch.rand(config["input_dim"], config["input_dim"]).to(device)
    target_type = config.get("target_type", "low_rank_identity")
    initialization_matrix_A = getattr(model, 'initialization_matrix_A', None)
    target_matrix = generate_low_rank_matrix_with_custom_rank(
        config["input_dim"],
        config["output_dim"],
        rank=config["rank"],
        target_type=target_type,
        initialization_matrix_A=initialization_matrix_A
    ).to(device)
    print(f"Dummy input shape: {dummmy_input.shape}")
    print(f"Target matrix shape: {target_matrix.shape}")
    dummmy_input = dummmy_input.unsqueeze(0)
    target_matrix = target_matrix.unsqueeze(0)
    single_loader = DataLoader(TensorDataset(dummmy_input, target_matrix), batch_size=1, shuffle=False)
    loss_csv_filename = os.path.join(
        experiment_dirs['visualizations'],
        f"linear{num_layer}_{gap_method}_training_loss_{method}_outputdim{output_dim}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}{get_activation_suffix(config)}.csv"
    )
    with open(loss_csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss'])
    progress_bar = tqdm(range(steps + 1),
                       desc="Training",
                       ncols=100)
    for step in progress_bar:
        step_start_time = time.time()
        output = model.forward(dummmy_input)
        loss = 0.5 * loss_function(output, target_matrix)
        step_history.append(step)
        loss_history.append(loss.item())
        with open(loss_csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, loss.item()])
        if hasattr(optimizer, 'zero_grad'):
            optimizer.zero_grad()
        else:
            optimizer.optimizer.zero_grad()
        loss.backward()
        if hasattr(optimizer, 'step'):
            optimizer.step()
        else:
            optimizer.optimizer.step()
        log_dict = {"Training Loss": loss.item()}
        hessian_time = 0
        if step in selected_steps:
            hessian_start = time.time()
            try:
                singular_values, sigma_power, weight_analysis = compute_weight_product_singular_values(
                    model, num_layer, top_k=20
                )
                collect_weight_matrix_data(weight_matrix_history, singular_values, sigma_power, top_k=20)
                weight_matrix_step_history.append(step)
                weight_swanlab_log = prepare_weight_swanlab_log(singular_values, sigma_power)
                print(f"   Sigma^(1/L) (top 5): {weight_analysis['sigma_power_top_10'][:5]}")
            except Exception as e:
                pass
            eigenvalues, eigenvectors, dominant_dim, gaps, success = compute_and_analyze_eigenvalues(
                model, loss_function, single_loader, top_k, device, config, step
            )
            fnorm, fnorm_squared = compute_hessian_fnorm_exact(
                    model, loss_function, single_loader, device
                )
            if success:
                eigenvalue_step_history.append(step)
                dominant_dim_history.append(dominant_dim)
                collect_eigenvalue_data(eigenvalue_history, eigenvalues)
                try:
                    save_eigenvalue_csv(eigenvalue_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, experiment_dirs['visualizations'])
                except Exception as e:
                    pass
                if enable_hessian_analysis and step in hessian_analysis_steps:
                    step_dirs = create_step_subdirectories(experiment_dirs, step)
                    if hasattr(optimizer, 'zero_grad'):
                        optimizer.zero_grad()
                    else:
                        optimizer.optimizer.zero_grad()
                    output = model.forward(dummmy_input)
                    analysis_loss = 0.5 * loss_function(output, target_matrix)
                    analysis_results = run_complete_hessian_analysis(
                                            model=model,
                                            loss=analysis_loss,
                                            config=config,
                                            step=step,
                                            eigenvalues=eigenvalues,
                                            eigenvectors=eigenvectors,
                                            dominant_dim=dominant_dim,
                                            fnorm_squared=fnorm_squared,
                                            base_save_dir=experiment_dirs['hessian_analysis'],
                                            device=device,
                                            wandb_logger=wandb
                                        )
                    hessian_analysis_history.append(analysis_results)
            hessian_time = time.time() - hessian_start
        wandb.log(log_dict, step=step)
        step_time = time.time() - step_start_time
        postfix_dict = {'Loss': f'{loss.item():.4f}'}
        if step in selected_steps:
            postfix_dict['EigenTime'] = f'{hessian_time:.1f}s'
        progress_bar.set_postfix(postfix_dict)
    progress_bar.close()
    try:
        if len(eigenvalue_step_history) > 0 and eigenvalue_history:
            save_eigenvalue_csv(eigenvalue_history, eigenvalue_step_history, method, lr, var, seed, rank, num_layer, config, experiment_dirs['visualizations'])
        if len(weight_matrix_step_history) > 0 and weight_matrix_history['weight_singular_values']:
            save_weight_matrix_csv(weight_matrix_history, weight_matrix_step_history,
                                  method, lr, var, seed, rank, num_layer, config,
                                  experiment_dirs['visualizations'])
    except Exception as e:
        import traceback
    if hessian_analysis_history:
        summary_data = []
        for result in hessian_analysis_history:
            trace_analysis = result['trace_analysis']
            energy_analysis = result['energy_analysis']
            summary_data.append({
                'step': result.get('step', 0),
                'trace': trace_analysis['trace'],
                'trace_ratio': trace_analysis['ratio'],
                'theoretical_ratio': trace_analysis['theoretical_ratio'],
                'energy_ratio': energy_analysis['energy_ratio']
            })
    total_time = time.time() - total_start_time
    wandb.finish()
if __name__ == "__main__":
    main()
