import torch
import numpy as np
def compute_hessian_fnorm_exact(model, criterion, data_loader, device):
    pass
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        if output.shape != target.shape:
            if output.dim() == 2 and target.dim() == 3:
                output = output.unsqueeze(0)
            elif output.dim() == 3 and target.dim() == 2:
                target = target.unsqueeze(0)
            if output.shape != target.shape:
                output = output.view(target.shape)
        loss = 0.5 * criterion(output, target)
        break
    first_grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_grad = torch.cat([grad.view(-1) for grad in first_grads])
    fnorm_squared = 0.0
    progress_step = max(1, total_params // 20)
    for i in range(total_params):
        if i % progress_step == 0 or i == total_params - 1:
            pass
        second_grads = torch.autograd.grad(
            outputs=flat_grad[i],
            inputs=params,
            retain_graph=True,
            allow_unused=True
        )
        flat_second_grad = torch.cat([
            grad.view(-1) if grad is not None else torch.zeros_like(param).view(-1)
            for grad, param in zip(second_grads, params)
        ])
        fnorm_squared += torch.sum(flat_second_grad ** 2).item()
    fnorm = np.sqrt(fnorm_squared)
    print(f"   F-norm² = {fnorm_squared:.6e}")
    print(f"   F-norm = {fnorm:.6e}")
    return fnorm, fnorm_squared
