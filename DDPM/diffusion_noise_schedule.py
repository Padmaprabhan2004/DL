import torch
import numpy as np
import matplotlib.pyplot as plt


class DiffusionNoiseScheduler:
    def __init__(self,n_steps,beta_init,beta_end):
        self.beta_t = torch.linspace(beta_init,beta_end,n_steps)
        self.alpha_t = 1-self.beta_t

        #cumulative product terms
        self.alpha_dash_t = torch.cumprod(self.alpha_t,dim=0)
        self.alpha_dash_sqrt = torch.sqrt(self.alpha_dash_t)

        self.one_minus_alpha_dash_sqrt = torch.sqrt(1-self.alpha_dash_t)
    
    def add_noise(self,x_0,noise,t):
        original = x_0.shape
        batch_size = original[0]
        alpha_dash_sqrt = self.alpha_dash_sqrt[t].reshape(batch_size)
        one_minus_alpha_dash_sqrt =  self.one_minus_alpha_dash_sqrt[t].reshape(batch_size)

        for _ in range(len(original)-1):
            alpha_dash_sqrt = alpha_dash_sqrt.unsqueeze(-1)
            one_minus_alpha_dash_sqrt= one_minus_alpha_dash_sqrt.unsqueeze(-1)
        
        return alpha_dash_sqrt*x_0 + one_minus_alpha_dash_sqrt*noise
    

    def sample_previous_timestep(self,x_t,noise_pred,t):
        x_0 = (x_t- (self.one_minus_alpha_dash_sqrt[t]*noise_pred))/self.alpha_dash_sqrt
        x_0 = torch.clamp(x_0,-1,1)

        mean = x_t - ((self.beta[t]*noise_pred)/(self.one_minus_alpha_dash_sqrt[t]))
        mean = mean/torch.sqrt(self.alpha_t[t])

        if t==0:
            return mean,x_0
        else:
            variance = (1-self.alpha_dash_t[t-1])/(1-self.alpha_dash_t[t])
            variance = variance*(self.beta_t[t])
            sigma = variance**0.5
            z= torch.randn(x_t.shape).to(x_t.device)
            return mean +sigma*z,x_0
        
