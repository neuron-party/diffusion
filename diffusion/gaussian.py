import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Diffusion:
    def __init__(self, device, steps=1000, beta_start=1e-4, beta_end=0.02):
        self.device = device
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.beta = torch.linspace(start=beta_start, end=beta_end, steps=steps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)
        
    def forward_chain(self, x_0, t):
        # forward process can be parameterized by x_t = sqrt(a_hat_t) * x_0 + sqrt(1 - a_hat_t) * epsilon (standard gaussian noise)
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1).to(self.device)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1).to(self.device)
        epsilon = torch.randn(x_0.shape, device=self.device)
        out = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * epsilon
        return out, epsilon
    
    def reverse_chain(self, model):
        with torch.no_grad():
            x = torch.randn((1, 3, 128, 128), device=self.device)

            for i in reversed(range(1, 1000)):
                t = torch.ones(1, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * model(x, t)))) + (epsilon_t * random_noise)


        x = ((x.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
        return x