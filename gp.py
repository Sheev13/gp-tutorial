import torch 
from torch import nn
from typing import Optional

def covariance_matrix(x1: torch.Tensor, x2: torch.Tensor, cov: str = 'se', l: float = 1.0):
    d = torch.cdist(x1.unsqueeze(-1)/l, x2.unsqueeze(-1)/l)
    if cov == 'exp':
        return torch.exp(-d)
    elif cov == 'mat1.5':
        return (1 + torch.tensor(3.0).sqrt() * d) * torch.exp(-torch.tensor(3.0).sqrt() * d)
    elif cov == 'mat2.5':
        return (1 + torch.tensor(5.0).sqrt() * d + (5/3) * d.square()) * torch.exp(-torch.tensor(5.0).sqrt() * d)
    elif cov == 'se':
        return torch.exp(-0.5 * d.square())
    else:
        raise NotImplementedError(f"'{cov}' covariance function not recognised or implemented")

def gp_prior(X: torch.Tensor, cov: str = 'se', l: float = 1.0):
    K = covariance_matrix(X, X, cov=cov, l=l) + torch.eye(X.shape[0])*1e-8
    return torch.distributions.MultivariateNormal(loc=torch.zeros_like(X), covariance_matrix=K)

def gp_posterior(X: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor, cov: str = 'se', sigma_y: float = 0.0, l: float = 1.0, observation_noise: bool = False):
    K_nn = covariance_matrix(X, X, cov=cov, l=l) + (sigma_y**2 + 1e-8)*torch.eye(X.shape[0])
    chol = torch.linalg.cholesky(K_nn)
    inv = torch.cholesky_inverse(chol)

    K_tn = covariance_matrix(x_test, X, cov=cov, l=l)
    K_tt = covariance_matrix(x_test, x_test, cov=cov, l=l) + (1e-8)*torch.eye(x_test.shape[0]) 
    if observation_noise: # True if p(y_*|...), False if p(f_*|...)
        K_tt += sigma_y**2 * torch.eye(x_test.shape[0])

    mu = K_tn @ inv @ y
    covar = K_tt - (K_tn @ inv @ K_tn.T)
    return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=covar)

def get_mean_and_stds(gp: torch.distributions.MultivariateNormal):
    mu = gp.mean
    covar = gp.covariance_matrix
    return mu, covar.diagonal().sqrt()

def get_samples(gp: torch.distributions.MultivariateNormal, n: int = 10):
    samps = torch.zeros((n, gp.mean.shape[0]))
    for i in range(n):
        samps[i,:] = gp.sample()
    return samps

# The below class is for (visualisations to do with) training GPs.
class GP(nn.Module):
    def __init__(self, sigma_y: float = 0.0, learn_sigma_y: bool = False, l: float=1.0, learn_l: bool=False):
        super().__init__()
        self.log_sigma_y = nn.Parameter(torch.tensor(sigma_y).abs().log(), requires_grad=learn_sigma_y)
        self.log_l = nn.Parameter(torch.tensor(l).abs().log(), requires_grad=learn_l)
    
    @property
    def sigma_y(self):
        return self.log_sigma_y.exp()
    
    @property
    def l(self):
        return self.log_l.exp()

    def forward(self, x_test: torch.Tensor, X: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, cov: str ='se', observation_noise: bool = False):
        if X is None:
            return gp_prior(x_test, cov=cov, l=self.l)
        else:
            assert y is not None
            return gp_posterior(X, y, x_test, cov=cov, l=self.l, sigma_y=self.sigma_y, observation_noise=observation_noise)
        
    def log_marginal_likelihood(self, X: torch.Tensor, y: torch.Tensor, cov: str = 'se'):
        I = torch.eye(X.shape[0])
        K_nn = covariance_matrix(X, X, cov=cov, l=self.l) + I*1e-8
        chol = torch.linalg.cholesky(K_nn + self.sigma_y**2*I)
        inv = torch.cholesky_inverse(chol)

        a = -0.5 * y.T@inv@y
        b = -0.5 * torch.linalg.det(inv).pow(-1).log()
        c = - X.shape[0]/2 * (torch.tensor(2) * torch.pi).log()

        return a + b + c