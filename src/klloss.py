import torch

def kl_divergence(mu: torch.tensor, log_var: torch.tensor):
    return -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mu.pow(2))