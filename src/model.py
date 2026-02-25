from .layers import LinearLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.linear_layer_1 = LinearLayer(x_dim, hidden_dim) # (N, H)
        self.linear_layer_2 = LinearLayer(hidden_dim, latent_dim) # (H, L)
        self.linear_layer_3 = LinearLayer(hidden_dim, latent_dim) # (H, L)

    def forward(self, x: torch.tensor):
        # x.shape  = (B, N)
        h = torch.sigmoid(self.linear_layer_1(x)) # (B, N) @ (N, H) --> (B, H)
        mu = self.linear_layer_2(h) # (B, H) @ (H, L) --> (B, L)
        log_var = self.linear_layer_3(h) # (B, H) @ (H, L) --> (B, L)

        return mu, log_var
    
    def reparameterization(self, mu: torch.tensor, log_var: torch.tensor):
        # sample eps from Gaussian Standard Normal Distribution
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * log_var)
        z = mu + std * eps
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, x_dim: int):
        super().__init__()

        self.linear_layer_1 = LinearLayer(latent_dim, hidden_dim)
        self.linear_layer_2 = LinearLayer(hidden_dim, hidden_dim)
        self.linear_layer_3 = LinearLayer(hidden_dim, x_dim)

    def forward(self, x):
        x = F.relu(self.linear_layer_1(x))
        x = F.relu(self.linear_layer_2(x))
        x = self.linear_layer_3(x)

        return torch.sigmoid(x)
    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, x_dim: int, hidden_enc: int, hidden_dec: int, latent_dim: int):
        super().__init__()

        self.x_dim = x_dim
        self.hidden_enc = hidden_enc
        self.hidden_dec = hidden_dec
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(x_dim = x_dim, hidden_dim = hidden_enc, latent_dim = latent_dim)
        self.decoder = Decoder(latent_dim = latent_dim, hidden_dim = hidden_dec, x_dim = x_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterization(mu, log_var)
        return self.decoder(z), mu, log_var