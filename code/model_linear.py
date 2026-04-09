import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd
from collections import OrderedDict
from config_linear import training_config
def get_activation_function(activation_type, params=None):
    pass
    if params is None:
        params = {}
    activation_functions = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(negative_slope=params.get("leaky_relu_negative_slope", 0.01)),
        "gelu": nn.GELU(),
        "swish": lambda x: x * torch.sigmoid(params.get("swish_beta", 1.0) * x),
        "none": nn.Identity()
    }
    return activation_functions.get(activation_type.lower(), nn.ReLU())
def balanced_init(input_dim, hidden_dim, output_dim, num_layer, device):
    pass
    dims = [input_dim]
    for _ in range(num_layer - 1):
        dims.append(hidden_dim)
    dims.append(output_dim)
    d0, dN = dims[0], dims[-1]
    min_d = min(d0, dN, hidden_dim)
    var = training_config["variance"]
    A = np.random.randn(dN, d0) * np.sqrt(var)
    U, Sigma, Vt = svd(A, full_matrices=False)
    Sigma_power = np.power(np.diag(Sigma[:min_d]), 1 / (len(dims) - 1))
    weights = []
    for i in range(len(dims) - 1):
        weight = np.zeros((dims[i + 1], dims[i]))
        if i == 0:
            weight[:min_d, :] = Sigma_power @ Vt[:min_d, :]
        elif i == len(dims) - 2:
            weight[:, :min_d] = U[:, :min_d] @ Sigma_power
        else:
            weight[:min_d, :min_d] = Sigma_power
        weights.append(torch.from_numpy(weight).float().to(device))
    return weights, A
class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, var, device):
        super(LinearNetwork, self).__init__()
        self.var = var
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.num_layer = num_layer
        self.use_activation = training_config.get("use_activation", False)
        self.activation_type = training_config.get("activation_type", "relu")
        self.activation_params = training_config.get("activation_params", {})
        weights, self.initialization_matrix_A = balanced_init(input_dim, hidden_dim, output_dim, num_layer, device)
        self.layers = nn.ModuleList()
        for i, weight in enumerate(weights):
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            layer.weight.data = weight
            self.layers.append(layer)
        if self.use_activation:
            self.activations = nn.ModuleList()
            for i in range(len(self.layers)):
                activation = get_activation_function(self.activation_type, self.activation_params)
                self.activations.append(activation)
    def _print_network_info(self):
        pass
        if self.use_activation:
            pass
        for i, layer in enumerate(self.layers):
            pass
    def forward(self, x):
        pass
        device = self.layers[0].weight.device
        x = torch.eye(self.input_dim).unsqueeze(0).to(device)
        batch_size = x.shape[0]
        x = x.squeeze(0)
        for i, layer in enumerate(self.layers):
            x = layer.weight @ x
            if self.use_activation and i < len(self.layers):
                if self.activation_type == "swish":
                    beta = self.activation_params.get("swish_beta", 1.0)
                    x = x * torch.sigmoid(beta * x)
                else:
                    x = self.activations[i](x)
        x = x.unsqueeze(0)
        return x
    def get_network_info(self):
        pass
        info = f"LinearNetwork-{self.num_layer}L"
        if self.use_activation:
            info += f"-{self.activation_type.upper()}"
        else:
            info += "-NoActivation"
        return info
    def count_parameters(self):
        pass
        return sum(p.numel() for p in self.parameters())
