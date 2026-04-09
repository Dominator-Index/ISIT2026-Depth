import torch
import numpy as np
import os
os.environ["SWANLAB_API_KEY"] = "your_swanlab_key"
from datetime import datetime
if torch.cuda.is_available():
    device = "cuda:3"
else:
    device = torch.device("cpu")
np_seed = 12138
torch_seed = 12138
np.random.seed(np_seed)
torch.manual_seed(torch_seed)
num_layer = 2
training_config = {
    "np_seed": 12138,
    "torch_seed": 12138,
    "learning_rate": 0.01,
    "steps": 10000,
    "record_steps": list(range(0, 10001, 100)),
    "input_dim": 16,
    "hidden_dim": 32,
    "output_dim": 10,
    "variance": 0.01,
    "rank": 5,
    "top_k_pca_number": 10000,
    "swanlab_project_name": "Baseline_ReLU_Projection",
    "num_layer": num_layer,
    "eigenvalue_interval": 25,
    "hessian_viz_interval": 1000,
    "method": "adaptive_threshold",
    "eigenvalue_threshold": 0.0001,
    "device": str(device),
    "enable_hessian_analysis": True,
    "hessian_analysis_steps": list(range(0, 1001, 50)),
    "use_layerwise_lr": False,
    "layer_learning_rates": {
        "layer_0": 2.0,
        "layer_1": 1.0,
        "layer_2": 2.0,
    },
    "cleanup_old_experiments": False,
    "base_dir": ".",
    "use_activation": False,
    "activation_type": "relu",
    "with_num": False,
    "scale": 1.0,
    "target_type": "aligned_matrix",
    "swanlab_api_key": "your_swanlab_key",
}
def generate_run_name(config):
    pass
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    activation_str = f"{config['activation_type']}" if config['use_activation'] else "no_act"
    run_name = (
        f"DomBulk_Linear{config['num_layer']}L_"
        f"dim{config['input_dim']}x{config['hidden_dim']}x{config['output_dim']}_"
        f"lr{config['learning_rate']}_"
        f"var{config['variance']:.4f}_"
        f"rank{config['rank']}_"
        f"steps{config['steps']}_"
        f"topk{config['top_k_pca_number']}_"
        f"seed{config['torch_seed']}_"
        f"{activation_str}_"
        f"scale{config['scale']}_"
        f"{timestamp}_"
        f"{config['method']}_"
    )
    return run_name
training_config["swanlab_run_name"] = generate_run_name(training_config)
