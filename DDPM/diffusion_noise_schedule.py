import torch
import numpy as np
import matplotlib.pyplot as plt



class DiffusionNoiseScheduler:
    def __init__(self, n_steps, beta_init, beta_end, device):
        self.device = device

        self.beta_t = torch.linspace(beta_init, beta_end, n_steps, device=device)
        self.alpha_t = 1.0 - self.beta_t

        self.alpha_dash_t = torch.cumprod(self.alpha_t, dim=0)
        self.alpha_dash_sqrt = torch.sqrt(self.alpha_dash_t)
        self.one_minus_alpha_dash_sqrt = torch.sqrt(1.0 - self.alpha_dash_t)

    def add_noise(self, x_0, noise, t):
        B = x_0.shape[0]

        a = self.alpha_dash_sqrt[t].view(B, 1, 1, 1)
        b = self.one_minus_alpha_dash_sqrt[t].view(B, 1, 1, 1)

        return a * x_0 + b * noise

    def sample_prev_timestep(self, x_t, noise_pred, t):


        alpha_t = self.alpha_t[t]
        beta_t = self.beta_t[t]
        alpha_dash_t = self.alpha_dash_t[t]

        sqrt_alpha_dash = torch.sqrt(alpha_dash_t)
        sqrt_one_minus = torch.sqrt(1 - alpha_dash_t)

        # Predict x0
        x_0 = (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha_dash
        x_0 = torch.clamp(x_0, -1.0, 1.0)

        # Mean of posterior q(x_{t-1} | x_t, x_0)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / sqrt_one_minus) * noise_pred
        )

        if t == 0:
            return mean, x_0
        else:
            variance = beta_t * (1 - self.alpha_dash_t[t - 1]) / (1 - alpha_dash_t)
            sigma = torch.sqrt(variance)
            z = torch.randn_like(x_t)
            return mean + sigma * z, x_0
