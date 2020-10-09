from __future__ import print_function

import numpy as np
import torch
from torch.autograd import grad as torchgrad
from tqdm.auto import tqdm

from .hmc import accept_reject, hmc_trajectory
from .utils import log_normal, safe_repeat


def ais_trajectory(
    model,
    loader,
    schedule=np.linspace(0.0, 1.0, 500),
    n_sample=100,
    n_latent=10,
    is_exp1=False,
):
    """Compute annealed importance sampling trajectories for a batch of data. 
  Could be used for *both* forward and reverse chain in BDMC.

  Args:
    model (vae.VAE): VAE model
    loader (iterator): iterator that returns pairs, with first component
      being `x`, second would be `z` or label (will not be used)
    forward (boolean): indicate forward/backward chain
    schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
    n_sample (int): number of importance samples

  Returns:
      A list where each element is a torch.autograd.Variable that contains the 
      log importance weights for a single batch of data
  """

    # Don't want any gradients for the trained model parameters
    model = model.eval()

    prior_distribution = torch.distributions.Normal(
        torch.zeros(n_latent, device="cuda"), torch.ones(n_latent, device="cuda")
    )

    def log_f_i(z, data, t):
        """
            Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        log_prior = prior_distribution.log_prob(z).sum(-1)
        log_likelihood = model.log_px_z(data, z)
        return log_prior + t * log_likelihood

    logws = []
    zs = []
    for i, tensors in enumerate(tqdm(loader)):
        if is_exp1:
            tensors = [torch.stack(tensors, 0)]
        batch = tensors[0]
        B = batch.size(0) * n_sample
        batch = safe_repeat(batch, n_sample)

        tensors = [safe_repeat(tens, n_sample) for tens in tensors]
        with torch.no_grad():
            epsilon = torch.ones(B).cuda().mul_(0.01)
            accept_hist = torch.zeros(B).cuda()
            logw = torch.zeros(B).cuda()

        # initial sample of z
        current_z = torch.randn(B, n_latent).cuda()
        current_z = current_z.requires_grad_()

        for j, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:]), 1):
            # update log importance weight
            log_int_1 = log_f_i(current_z, tensors, t0)
            log_int_2 = log_f_i(current_z, tensors, t1)
            logw += log_int_2 - log_int_1

            # resample velocity
            current_v = torch.randn(current_z.size()).cuda()

            def U(z):
                return -log_f_i(z, tensors, t1)

            def grad_U(z):
                # grad w.r.t. outputs; mandatory in this case
                grad_outputs = torch.ones(B).cuda()
                # torch.autograd.grad default returns volatile
                grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
                # clip by norm
                max_ = B * n_latent * 100.0
                grad = torch.clamp(grad, -max_, max_)
                grad.requires_grad_()
                return grad

            def normalized_kinetic(v):
                zeros = torch.zeros(B, n_latent).cuda()
                return -log_normal(v, zeros, zeros)

            z, v = hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
            current_z, epsilon, accept_hist = accept_reject(
                current_z,
                current_v,
                z,
                v,
                epsilon,
                accept_hist,
                j,
                U,
                K=normalized_kinetic,
            )

        # logw = log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
        # logws.append(logw.data.cpu())
        # print("Last batch stats %.4f" % (logw.mean().cpu().data.numpy()))
        logws.append(logw.data.view(n_sample, -1).cpu())
        zs.append(current_z.data.view(n_sample, -1, n_latent).cpu())
    zs = torch.cat(zs, dim=1)
    logws = torch.cat(logws, dim=-1)
    return zs, logws


def ais_trajectory_sample(
    model, tensors, schedule=np.linspace(0.0, 1.0, 500), n_sample=100, n_latent=10
):
    """Compute annealed importance sampling trajectories for a batch of data. 
  Could be used for *both* forward and reverse chain in BDMC.

  Args:
    model (vae.VAE): VAE model
    loader (iterator): iterator that returns pairs, with first component
      being `x`, second would be `z` or label (will not be used)
    forward (boolean): indicate forward/backward chain
    schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
    n_sample (int): number of importance samples

  Returns:
      A list where each element is a torch.autograd.Variable that contains the 
      log importance weights for a single batch of data
  """

    # Don't want any gradients for the trained model parameters
    model = model.eval()

    prior_distribution = torch.distributions.Normal(
        torch.zeros(n_latent, device="cuda"), torch.ones(n_latent, device="cuda")
    )

    def log_f_i(z, data, t):
        """
            Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        log_prior = prior_distribution.log_prob(z).sum(-1)
        log_likelihood = model.log_px_z(data, z)
        return log_prior + t * log_likelihood

    batch = tensors[0]
    B = batch.size(0) * n_sample
    batch = safe_repeat(batch, n_sample)

    tensors = [safe_repeat(tens, n_sample) for tens in tensors]
    with torch.no_grad():
        epsilon = torch.ones(B).cuda().mul_(0.01)
        accept_hist = torch.zeros(B).cuda()
        logw = torch.zeros(B).cuda()

    # initial sample of z
    current_z = torch.randn(B, n_latent).cuda()
    current_z = current_z.requires_grad_()

    for j, (t0, t1) in enumerate(zip(tqdm(schedule[:-1]), schedule[1:]), 1):
        # update log importance weight
        log_int_1 = log_f_i(current_z, tensors, t0)
        log_int_2 = log_f_i(current_z, tensors, t1)
        logw += log_int_2 - log_int_1

        # resample velocity
        current_v = torch.randn(current_z.size()).cuda()

        def U(z):
            return -log_f_i(z, tensors, t1)

        def grad_U(z):
            # grad w.r.t. outputs; mandatory in this case
            grad_outputs = torch.ones(B).cuda()
            # torch.autograd.grad default returns volatile
            grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
            # clip by norm
            max_ = B * n_latent * 100.0
            grad = torch.clamp(grad, -max_, max_)
            grad.requires_grad_()
            return grad

        def normalized_kinetic(v):
            zeros = torch.zeros(B, n_latent).cuda()
            return -log_normal(v, zeros, zeros)

        z, v = hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
        current_z, epsilon, accept_hist = accept_reject(
            current_z, current_v, z, v, epsilon, accept_hist, j, U, K=normalized_kinetic
        )
    logws = logw.data.view(n_sample, -1).cpu()
    zs = current_z.data.view(n_sample, -1, n_latent).cpu()
    return zs, logws
