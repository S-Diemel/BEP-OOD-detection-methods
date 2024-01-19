
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Gaussian Multivariate Layer for epistemic and aleatoric uncertainty as proposed for the DDU model
class Gauss_DDU(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features, gamma =5e-3):
        super(Gauss_DDU, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('classwise_mean_features', torch.zeros(out_features, in_features))
        self.register_buffer('classwise_cov_features', torch.eye(in_features).unsqueeze(0).repeat(out_features, 1, 1))
        self.gda = self.init_gda()  # class-wise multivatiate Gaussians, to be initialized with fit()
        self.mahalanobis = torch.distributions.multivariate_normal._batch_mahalanobis
        self.gamma = nn.Parameter(gamma*torch.ones(out_features))
        self.percentile = 1000
    
    def forward(self, D):
        L  = self.gda._unbroadcasted_scale_tril

        if L.device.type != 'cpu':
            gamma=self.gamma.to('cuda')
        else:
            gamma = self.gamma

        return -gamma * self.mahalanobis(L, D[:, None, :]-self.gda.loc).float()

    def forward2(self, D, scale):
        L  = scale*self.gda._unbroadcasted_scale_tril
        if L.device.type != 'cpu':
            gamma=self.gamma.to('cuda')
        else:
            gamma = self.gamma
        return -gamma * self.mahalanobis(L, D[:, None, :]-self.gda.loc).float()

    def get_log_probs(self,D):

        return self.gda.log_prob(D[:, None, :]).float()

    def conf(self,D):
        return self.conf_logits(self.forward(D))

    def conf_logits(self,logits):
        return torch.exp(logits)
    
    def prox(self):
        return
    
    def fit(self, embeddings, labels): #embeddings should be num_samples x dim_embedding
        with torch.no_grad():
            classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(self.out_features)])
            classwise_cov_features = torch.stack(
                [torch.cov(embeddings[labels == c].T) for c in range(self.out_features)])

            for jitter_eps in [0, torch.finfo(torch.float).tiny] + [10 ** exp for exp in range(-308, 0, 2)]:
                try:
                    jitter = jitter_eps * torch.eye(
                        classwise_cov_features.shape[1], device=classwise_cov_features.device,
                    ).unsqueeze(0)
                    self.classwise_mean_features = classwise_mean_features
                    self.classwise_cov_features = classwise_cov_features + jitter
                    self.init_gda()
                except RuntimeError as e:
                    continue
                except ValueError as e:
                    continue
                break
    
    def init_gda(self):
        self.gda = torch.distributions.MultivariateNormal(loc=self.classwise_mean_features, covariance_matrix=(self.classwise_cov_features))

# simple layer to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


#%%
