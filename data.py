import torch
import gp


def _sample_uniform_inputs(num: int):
    x = (torch.rand((num,)) * (2.5 - (-2.5))) + (-2.5)
    return x

def get_data(n: int, noise: float = 0.0):
    x = _sample_uniform_inputs(num=n)
    gp_prior = gp.gp_prior(x)
    fs = gp.get_samples(gp_prior, n=1).squeeze()
    y = fs + (torch.randn_like(fs) * noise)
    return x, y