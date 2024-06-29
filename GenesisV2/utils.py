import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clamp_preserve_gradients(x, lower, upper):
    return x + (x.clamp(lower, upper) - x).detach()

def euclidian_norm(x):
    return clamp_preserve_gradients((x**2).sum(1), 1e-10, 1e10).sqrt()

def to_sigma(x):
    return F.softplus(x + 0.5) + 1e-8

def to_prior_sigma(x, simgoid_bias=4.0, eps=1e-4):
    return torch.sigmoid(x + simgoid_bias) + eps

def pixel_coords(img_size):
    g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, img_size, device=device),
                              torch.linspace(-1, 1, img_size, device=device))
    g_1 = g_1.view(1, 1, img_size, img_size)
    g_2 = g_2.view(1, 1, img_size, img_size)
    return torch.cat((g_1, g_2), dim=1)

def unflatten(x):
    return x.view(x.size(0), -1, 1, 1)

def euclidian_distance(embedA, embedB):
    if embedA.dim() == 4 or embedB.dim() == 4:
        if embedA.dim() == 2:
            embedA = unflatten(embedA)
        if embedB.dim() == 2:
            embedB = unflatten(embedB)
    return euclidian_norm(embedA - embedB)

def squared_distance(embedA, embedB):
    if embedA.dim() == 4 or embedB.dim() == 4:
        if embedA.dim() == 2:
            embedA = unflatten(embedA)
        if embedB.dim() == 2:
            embedB = unflatten(embedB)
    return ((embedA - embedB)**2).sum(1)

def x_loss(x, log_m_k, x_r_k, std, pixel_wise=False):
    p_xr_stack = Normal(torch.stack(x_r_k, dim=4), std)
    log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
    log_m_stack = torch.stack(log_m_k, dim=4)
    log_mx = log_m_stack + log_xr_stack
    err_ppc = -torch.log(log_mx.exp().sum(dim=4))
    if pixel_wise:
        return err_ppc
    else:
        return err_ppc.sum(dim=(1, 2, 3))
    
def mask_latent_loss(q_zm_0_k, zm_0_k, zm_k_k=None, ldj_k=None,
                        prior_lstm=None, prior_linear=None, debug=False):
    num_steps = len(zm_0_k)
    batch_size = zm_0_k[0].size(0)
    latent_dim = zm_0_k[0].size(1)
    if zm_k_k is None:
        zm_k_k = zm_0_k


    if prior_lstm is not None and prior_linear is not None:
        zm_seq = torch.cat(
            [zm.view(1, batch_size, -1) for zm in zm_k_k[:-1]], dim=0)
        lstm_out, _ = prior_lstm(zm_seq)
        linear_out = prior_linear(lstm_out)
        linear_out = torch.chunk(linear_out, 2, dim=2)
        mu_raw = torch.tanh(linear_out[0])
        sigma_raw = to_prior_sigma(linear_out[1])
        mu_k = torch.split(mu_raw, 1, dim=0)
        sigma_k = torch.split(sigma_raw, 1, dim=0)
        p_zm_k = [Normal(0, 1)]
        for mean, std in zip(mu_k, sigma_k):
            p_zm_k += [Normal(mean.view(batch_size, latent_dim),
                                std.view(batch_size, latent_dim))]
        if debug:
            assert zm_seq.size(0) == num_steps-1
    else:
        p_zm_k = num_steps*[Normal(0, 1)]

    kl_m_k = []
    for step, p_zm in enumerate(p_zm_k):
        log_q = q_zm_0_k[step].log_prob(zm_0_k[step]).sum(dim=1)
        log_p = p_zm.log_prob(zm_k_k[step]).sum(dim=1)
        kld = log_q - log_p
        if ldj_k is not None:
            ldj = ldj_k[step].sum(dim=1)
            kld = kld - ldj
        kl_m_k.append(kld)

    if debug:
        assert len(p_zm_k) == num_steps
        assert len(kl_m_k) == num_steps

    return kl_m_k, p_zm_k


def get_mask_recon_stack(m_r_logits_k, prior_mode, log):
    if prior_mode == 'softmax':
        if log:
            return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
    elif prior_mode == 'scope':
        log_m_r_k = []
        log_s = torch.zeros_like(m_r_logits_k[0], device=device)
        for step, logits in enumerate(m_r_logits_k):
            if step == len(m_r_logits_k) - 1:
                log_m_r_k.append(log_s)
            else:
                log_a = F.logsigmoid(logits)
                log_neg_a = F.logsigmoid(-logits)
                log_m_r_k.append(log_s + log_a)
                log_s = log_s +  log_neg_a
        log_m_r_stack = torch.stack(log_m_r_k, dim=4)
        return log_m_r_stack if log else log_m_r_stack.exp()
    else:
        raise ValueError("No valid prior mode.")

def kl_m_loss(log_m_k, log_m_r_k, debug=False):
    if debug:
        assert len(log_m_k) == len(log_m_r_k)
    batch_size = log_m_k[0].size(0)
    m_stack = torch.stack(log_m_k, dim=4).exp()
    m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
    m_stack = torch.max(m_stack, torch.tensor(1e-5, device=device))
    m_r_stack = torch.max(m_r_stack, torch.tensor(1e-5, device=device))
    q_m = Categorical(m_stack.view(-1, len(log_m_k)))
    p_m = Categorical(m_r_stack.view(-1, len(log_m_k)))
    kl_m_ppc = kl_divergence(q_m, p_m).view(batch_size, -1)
    return kl_m_ppc.sum(dim=1)

def check_log_masks(log_m_k):
    summed_masks = torch.stack(log_m_k, dim=4).exp().sum(dim=4)
    summed_masks = summed_masks.clone().data.cpu().numpy()
    flat = summed_masks.flatten()
    diff = flat - np.ones_like(flat)
    idx = np.argmax(diff)
    max_diff = diff[idx]
    if max_diff > 1e-3 or np.any(np.isnan(flat)):
        print("Max difference: {}".format(max_diff))
        for i, log_m in enumerate(log_m_k):
            mask_k = log_m.exp().data.cpu().numpy()
            print("Mask value at k={}: {}".format(i, mask_k.flatten()[idx]))
        raise ValueError("Masks do not sum to 1.0. Not close enough.")

class BroadcastLayer(nn.Module):
    def __init__(self, dim):
        super(BroadcastLayer, self).__init__()
        self.dim = dim
        self.pixel_coords = PixelCoords(dim)
    def forward(self, x):
        b_sz = x.size(0)
        if x.dim() == 2:
            x = x.view(b_sz, -1, 1, 1)
            x = x.expand(-1, -1, self.dim, self.dim)
        else:
            x = F.interpolate(x, self.dim)
        return self.pixel_coords(x)

class PixelCoords(nn.Module):
    def __init__(self, im_dim):
        super(PixelCoords, self).__init__()
        g_1, g_2 = torch.meshgrid(torch.linspace(-1, 1, im_dim, device=device),
                                  torch.linspace(-1, 1, im_dim, device=device))
        self.g_1 = g_1.view((1, 1) + g_1.shape)
        self.g_2 = g_2.view((1, 1) + g_2.shape)
    def forward(self, x):
        g_1 = self.g_1.expand(x.size(0), -1, -1, -1)
        g_2 = self.g_2.expand(x.size(0), -1, -1, -1)
        return torch.cat((x, g_1, g_2), dim=1)

class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True)
        )

class ConvReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.ReLU(inplace=True)
        )

class ConvINReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.ReLU(inplace=True)
        )

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ScalarGate(nn.Module):
    def __init__(self, init=0.0):
        super(ScalarGate, self).__init__()
        self.gate = nn.Parameter(torch.tensor(init, device=device))
    def forward(self, x):
        return self.gate * x
    
class SemiConv(nn.Module):
    def __init__(self, nin, nout, img_size):
        super(SemiConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1)
        self.gate = ScalarGate()
        coords = pixel_coords(img_size)
        zeros = torch.zeros(1, nout-2, img_size, img_size, device=device)
        self.uv = torch.cat((zeros, coords), dim=1)
    def forward(self, x):
        out = self.gate(self.conv(x))
        delta = out[:, -2:, :, :]
        return out + self.uv.to(out.device), delta
    
class GECO():

    def __init__(self, goal, step_size, alpha=0.99, beta_init=1.0,
                 beta_min=1e-10, speedup=None):
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.beta = torch.tensor(beta_init, device=device)
        self.beta_min = torch.tensor(beta_min, device=device)
        self.beta_max = torch.tensor(1e10, device=device)
        self.speedup = speedup

    def to_cuda(self):
        self.beta = self.beta.cuda()
        if self.err_ema is not None:
            self.err_ema = self.err_ema.cuda()

    def loss(self, err, kld):
        loss = err + self.beta * kld
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0-self.alpha)*err + self.alpha*self.err_ema
            constraint = (self.goal - self.err_ema)
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)
            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        return loss