import torch
import numpy as np
import random
import time
from datetime import datetime
def set_seed(seed):
    pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def format_time(seconds):
    pass
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"
def generate_eigenvalue_steps(steps, eigenvalue_interval):
    pass
    important_early_steps = [1, 2, 3, 4, 5]
    interval_steps = list(range(0, steps + 1, eigenvalue_interval))
    final_steps = [steps] if steps not in interval_steps else []
    selected_steps = sorted(set(important_early_steps + interval_steps + final_steps))
    return selected_steps
def prepare_data_dimensions(data, label, device):
    pass
    if data.dim() == 2:
        data = data.unsqueeze(0)
    if label.dim() == 2:
        label = label.unsqueeze(0)
    data = data.to(device)
    label = label.to(device)
    return data, label
def setup_optimizer(model, config, args):
    pass
    if args.use_optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
        method = "SGD"
        print("Using SGD optimizer")
    else:
        optimizer = None
        method = "Manual"
        print("Using manual gradient descent")
    return optimizer, method
def print_training_info(steps, seeds, selected_steps):
    pass
    print(f"\n{'='*60}")
    print(f"{'='*60}\n")
