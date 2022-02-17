import torch
import geomloss
import numpy as np
import FrEIA.modules as Fm
import FrEIA.framework as Ff
from typing import Callable, Union

class _BaseCouplingBlock(Fm.InvertibleModule):
    '''Copied wholesale from FrEIA'''

    def __init__(self, dims_in, dims_c=[],
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN"):
        '''
        Additional args in docstring of base class.
        Args:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.channels // 2

        self.clamp = clamp

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = (lambda u: 2. * (torch.sigmoid(u) - 0.5))
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations

        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)

        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1, j1 = self._coupling1(x1, x2_c)

            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2, j2 = self._coupling2(x2, y1_c)
        else:
            # names of x and y are swapped for the reverse computation
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2, j2 = self._coupling2(x2, x1_c, rev=True)

            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1, j1 = self._coupling1(x1, y2_c, rev=True)

        return (torch.cat((y1, y2), 1),), j1 + j2

    def _coupling1(self, x1, u2, rev=False):
        '''The first/left coupling operation in a two-sided coupling block.
        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def _coupling2(self, x2, u1, rev=False):
        '''The second/right coupling operation in a two-sided coupling block.
        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j2 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims

class iConditionalAffineTransform(_BaseCouplingBlock):
    """Adapted from the FrEIA package. Makes the conditional affine block volume preserving."""
    def __init__(self, dims_in, dims_c=[],
        subnet_constructor: Callable = None,
        clamp: float = 2.,
        clamp_activation: Union[str, Callable] = "ATAN"):
        
        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        if not self.conditional:
            raise ValueError("ConditionalAffineTransform must have a condition")

        self.subnet = subnet_constructor(self.condition_length, 2 * self.channels)


    def forward(self, x, c=[], rev=False, jac=True):
        if len(c) > 1:
            cond = torch.cat(c, 1)
        else:
            cond = c[0]

        # notation:
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(cond)
        s, t = a[:, :self.channels], a[:, self.channels:]
        
        # make it volume preserving
        a[:,-1] = -a[:,:-1].sum(dim=1)
        
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y = (x[0] - t) * torch.exp(-s)
            return (y,), -j
        else:
            y = torch.exp(s) * x[0] + t
            return (y,), j

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class InvLocalAffines(_BaseCouplingBlock):
    def __init__(self, dims_in, num_nbhds, total_dim, device, dims_c=[],
        clamp: float = 2.,
        clamp_activation: Union[str, Callable] = "ATAN"):
        
        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.affines = [Fm.OrthogonalTransform([[total_dim]]) for _ in range(num_nbhds)]

        # Make it so our affines are registered as submodules
        for i, aff in enumerate(self.affines):
            self._modules[f'aff{i}'] = aff

    def forward(self, x, c=[], rev=False, jac=True):
        # input validation for sanity
        assert type(c[0]) is int and len(c) == 1
        idx = c[0]
        assert 0 <= idx and idx < len(self.affines)

        proj = self.affines[idx]
        return proj(x, rev=rev)
    
    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims

############################################
# Similarity Metrics for generative models #
############################################

def MSMD(sample, true_pts):
    dists = (sample.unsqueeze(1).repeat(1,len(true_pts), 1) - true_pts.unsqueeze(0).repeat(len(sample), 1, 1)).norm(dim=-1)
    min_dists = dists.min(dim=1)[0]
    return min_dists.pow(2).mean()

def MMD_old(s1, s2, alphas=[1.0]):
    def kernel(x, y, alphas):
        total = 0
        for alpha in alphas:
            total += (-alpha*(x-y).norm().pow(2)).exp()
        return total

    k_xx, k_yy, k_xy = 0, 0, 0

    for i, x in enumerate(s1):
        k_xx += kernel(x, s1[np.array([j for j in range(len(s1)) if not i == j])], alphas)
        k_xy += kernel(x, s2, alphas)
    k_xx /= len(s1)*(len(s1) - 1)
    k_xy /= len(s1)*len(s2)
    
    for i, y in enumerate(s2):
        k_yy += kernel(y, s2[np.array([j for j in range(len(s2)) if not i == j])], alphas)
    k_yy /= len(s2)*(len(s2) - 1)

    return k_xx + k_yy - 2*k_xy

def MMD(s1, s2):
    loss = geomloss.SamplesLoss("gaussian", blur=0.5)
    return loss(s1, s2)

def KL_divergence(s1, s2, k=1):
    n, m = len(s1), len(s2)
    D = torch.tensor(m / (n - 1), dtype=torch.float, device=s1.device).log()
    d = torch.tensor(s1.shape[1], dtype=torch.float, device=s1.device)
    
    for pt in s1:
        # estimate densities using nearest neighbors
        norms = (s2-pt).norm(dim=1).reshape(-1)
        nu = norms[~(norms == 0)].kthvalue(k=k)[0]

        norms = (s1-pt).norm(dim=1).reshape(-1)
        rho = norms[~(norms == 0)].kthvalue(k=k)[0]

        D += (d/n)*(nu/rho).log()
    return D

def Wasserstein1(s1, s2, blur=0.01):
    loss = geomloss.SamplesLoss('sinkhorn', p=1, blur=blur, debias=True)
    return loss(s1, s2)

def Wasserstein2(s1, s2, blur=0.01):
    loss = geomloss.SamplesLoss('sinkhorn', p=2, blur=blur, debias=True)
    return loss(s1, s2)