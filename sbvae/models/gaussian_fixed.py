# -*- coding: utf-8 -*-
"""Main module."""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from torch.distributions import MultivariateNormal, Normal

from sbvae.models.modules import Encoder
from sbvae.models.regular_modules import LinearEncoder

torch.backends.cudnn.benchmark = True


# Linear Gaussian with fixed variance model
class LinearGaussian(nn.Module):
    r"""Variational encoder model. Support full covariances as long as it is not learned.

    :param n_input: Number of input genes
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder NN
    :param dropout_rate: Dropout rate for neural networks

    """

    def __init__(
        self,
        A_param,
        pxz_log_det,
        pxz_inv_sqrt,
        gamma,
        n_input: int,
        learn_var: bool = False,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        linear_encoder: bool = False,
    ):
        super().__init__()

        self.n_latent = n_latent
        self.linear_encoder = linear_encoder
        if not linear_encoder:
            self.encoder = Encoder(
                n_input,
                n_latent,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
            )
        else:
            self.encoder = LinearEncoder(n_input, n_latent)

        self.A = torch.from_numpy(np.array(A_param, dtype=np.float32)).cuda()

        self.log_det_pxz = torch.tensor(
            pxz_log_det, requires_grad=False, dtype=torch.float
        ).cuda()
        self.inv_sqrt_pxz = torch.from_numpy(
            np.array(pxz_inv_sqrt, dtype=np.float32)
        ).cuda()

        self.learn_var = learn_var
        self.px_log_diag_var = torch.nn.Parameter(torch.randn(1, n_input))

        self.gamma = torch.from_numpy(np.array(np.diag(gamma), dtype=np.float32)).cuda()
        log_det = np.log(np.linalg.det(gamma))
        self.log_det_px_z = torch.tensor(
            log_det, requires_grad=False, dtype=torch.float
        ).cuda()
        inv_sqrt = sqrtm(np.linalg.inv(gamma))
        self.inv_sqrt_px_z = torch.from_numpy(
            np.array(inv_sqrt, dtype=np.float32)
        ).cuda()

    def get_std(self):
        return torch.sqrt(torch.exp(self.px_log_diag_var))

    def get_latents(self, x):
        r""" returns the result of ``sample_from_posterior`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior(x)]

    def sample_from_posterior(self, x, give_mean=False):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        qz_vars = self.encoder(x, None)  # y only used in VAEC
        qz_m = qz_vars["q_m"]
        z = qz_vars["latent"]
        if give_mean:
            z = qz_m
        return z

    def inference(self, x, n_samples=1, reparam: bool = True):
        # Sampling
        if not self.linear_encoder:
            qz_vars = self.encoder(x, None)
            qz_m = qz_vars["q_m"]
            qz_v = qz_vars["q_v"]
            z = qz_vars["latent"]

            if n_samples > 1:
                qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
                qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))

                if reparam:
                    z = Normal(qz_m, qz_v.sqrt()).rsample()
                else:
                    z = Normal(qz_m, qz_v.sqrt()).sample()
        else:
            qz_vars = self.encoder(x, n_samples=n_samples, reparam=reparam)
            qz_m = qz_vars["q_m"]
            qz_v = qz_vars["q_v"]
            z = qz_vars["latent"]

        px_mean = torch.matmul(z, torch.transpose(self.A, 0, 1))
        return px_mean, torch.exp(self.px_log_diag_var), qz_m, qz_v, z

    def log_ratio(self, x, px_mean, px_var, qz_m, qz_v, z, return_full=False):
        if self.learn_var:
            log_px_z = (
                Normal(px_mean, torch.sqrt(px_var))
                .log_prob(x.repeat(px_mean.shape[0], 1, 1))
                .sum(dim=-1)
                .view((px_mean.shape[0], -1))
            )
            log_pz = (
                Normal(torch.zeros_like(qz_m), torch.ones_like(qz_m))
                .log_prob(z)
                .sum(dim=-1)
            )
            if not self.linear_encoder:
                log_qz_given_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            else:
                log_qz_given_x = MultivariateNormal(qz_m, qz_v).log_prob(
                    z
                )  # No need to sum over latent dim
            log_pxz = log_px_z + log_pz

        else:
            zx = torch.cat([z, x.repeat(px_mean.shape[0], 1, 1)], dim=-1)
            reshape_dim = x.shape[-1] + z.shape[-1]

            if not self.linear_encoder:
                log_qz_given_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            else:
                log_qz_given_x = MultivariateNormal(qz_m, qz_v).log_prob(z)
            log_pxz = self.log_normal_full(
                zx.view((-1, reshape_dim)),
                torch.zeros_like(zx.view((-1, reshape_dim))),
                self.log_det_pxz,
                self.inv_sqrt_pxz,
            ).view((px_mean.shape[0], -1))

        log_ratio = log_pxz - log_qz_given_x
        if return_full:
            return log_ratio, log_pxz, log_qz_given_x
        else:
            return log_ratio

    def neg_iwelbo(self, x, n_samples_mc):
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(
            x, n_samples=n_samples_mc, reparam=True
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)
        iwelbo = torch.logsumexp(log_ratio, dim=0) - np.log(n_samples_mc)
        return -iwelbo

    def neg_iwelbo_grad(self, x, n_samples_mc):
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(
            x, n_samples=n_samples_mc, reparam=True
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)
        iwelbo = torch.softmax(log_ratio, dim=0).detach() * log_ratio
        return -iwelbo.sum(dim=0)

    def neg_elbo(self, x):
        px_mean, px_var, qz_m, qz_v, z = self.inference(x, n_samples=2, reparam=True)
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)
        neg_elbo = -log_ratio.mean(dim=0)
        return neg_elbo

    def cubo(self, x, n_samples_mc):
        # computes the naive cubo from chi2 upper bound
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(
            x, n_samples=n_samples_mc, reparam=False
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)
        cubo = torch.logsumexp(2 * log_ratio, dim=0) - np.log(n_samples_mc)
        return 0.5 * cubo

    def cubo_grad(self, x, n_samples_mc):
        # computes the importance sampled objective for reverse KL EP (revisited reweighted wake-sleep)
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(
            x, n_samples=n_samples_mc, reparam=True
        )
        log_ratio, _, log_qz_given_x = self.log_ratio(
            x, px_mean, px_var, qz_m, qz_v, z, return_full=True
        )
        ws = torch.softmax(2 * log_ratio, dim=0)
        cubo = ws.detach() * (-1) * log_ratio
        # print(ws[:, 0])
        return cubo.sum(dim=0)

    def iwrevkl_obj(self, x, n_samples_mc):
        # computes the importance sampled objective for reverse KL EP (revisited reweighted wake-sleep)
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(
            x, n_samples=n_samples_mc, reparam=False
        )
        log_ratio, _, log_qz_given_x = self.log_ratio(
            x, px_mean, px_var, qz_m, qz_v, z, return_full=True
        )
        ws = torch.softmax(log_ratio, dim=0)
        rev_kl = ws.detach() * (-1) * log_qz_given_x
        return rev_kl.sum(dim=0)

    def vr_max(self, x, n_samples_mc):
        # computes the naive MC from VR-max bound
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z = self.inference(
            x, n_samples=n_samples_mc, reparam=True
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)
        return -log_ratio.max(dim=0)[0]

    def generate_prior_data(self):
        shape_z = (128, self.n_latent)
        z = Normal(torch.zeros(shape_z).cuda(), torch.ones(shape_z).cuda()).sample()
        px_mean = torch.matmul(z, torch.transpose(self.A, 0, 1))
        if self.learn_var:
            px_var = torch.exp(self.px_log_diag_var)
        else:
            px_var = self.gamma
        x = Normal(px_mean, torch.sqrt(px_var)).sample()
        return x, z

    def sleep_kl(self, x, z):
        x = x.detach()
        z = z.detach()
        vars = self.encoder(x, None)
        qz_m = vars["q_m"]
        qz_v = vars["q_v"]
        log_qz_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
        return -log_qz_x

    def forward(self, x, param):
        if param == "ELBO":
            return self.neg_elbo(x)
        if param == "CUBO":
            return self.cubo_grad(x, n_samples_mc=10)
        if param == "REVKL":
            return self.iwrevkl_obj(x, n_samples_mc=20)
        if param == "IWELBO":
            return self.neg_iwelbo(x, n_samples_mc=80)
        if param == "VRMAX":
            return self.vr_max(x, n_samples_mc=20)
        if param == "SLEEPKL":
            # unpack and compute KL
            x_data, z_data = x
            return self.sleep_kl(x_data, z_data)

    def joint_log_likelihood(self, xz, pxz_mean, pxz_var):
        if self.learn_var:
            return Normal(pxz_mean, torch.sqrt(pxz_var)).log_prob(xz).sum(dim=1)
        else:
            return self.log_normal_full(
                xz, pxz_mean, self.log_det_pxz, self.inv_sqrt_pxz
            )

    @staticmethod
    def log_normal_full(x, mean, log_det, inv_sqrt):
        # copying code from NUMPY
        d = x.shape[1]

        log_lik = torch.zeros((x.shape[0],), dtype=torch.float).cuda()
        log_lik += d * np.log(2 * np.array(np.pi, dtype=np.float32))
        log_lik += log_det
        vec_ = torch.matmul(x - mean, inv_sqrt)
        log_lik += torch.mul(vec_, vec_).sum(dim=-1)
        return -0.5 * log_lik

    @torch.no_grad()
    def prob_event(self, x, n_samples_mc, nu: Union[float, list, np.ndarray] = 0.0):
        px_mean, px_var, qz_m, qz_v, z = self.inference(x, n_samples=n_samples_mc)

        # compute for importance sampling
        log_ratio = self.log_ratio(x, px_mean, px_var, qz_m, qz_v, z)
        ratio = torch.exp(log_ratio - torch.max(log_ratio, dim=0)[0])

        # get SNIPS estimator
        if hasattr(nu, "__len__"):
            res = []
            for nu_item in nu:
                res_item = torch.sum(
                    ratio * (z[:, :, 0] <= nu_item).float(), dim=0
                ) / torch.sum(ratio, dim=0)
                res.append(res_item.view(-1, 1))
            res = torch.cat(res, dim=-1)
        else:
            res = torch.sum(ratio * (z[:, :, 0] <= nu).float(), dim=0) / torch.sum(
                ratio, dim=0
            )

        # get ESS
        ess = torch.sum(ratio, dim=0) ** 2 / torch.sum(ratio ** 2, dim=0)

        if not self.linear_encoder:
            mean_to_return = qz_m.mean(dim=0)
            variance_to_return = qz_v.mean(dim=0)
        else:
            mean_to_return = qz_m
            variance_to_return = qz_v
        return mean_to_return, variance_to_return, res, ess


class LinearGaussianDefensive(nn.Module):
    r"""Variational encoder model. Support full covariances as long as it is not learned.

    :param n_input: Number of input genes
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder NN
    :param dropout_rate: Dropout rate for neural networks

    """

    def __init__(
        self,
        A_param,
        pxz_log_det,
        pxz_inv_sqrt,
        gamma,
        n_input: int,
        learn_var: bool = False,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        linear_encoder: bool = False,
        multi_encoder_keys=["default"],
    ):
        super().__init__()

        self.n_latent = n_latent
        self.linear_encoder = linear_encoder
        if not linear_encoder:
            self.encoder = nn.ModuleDict(
                {
                    key: Encoder(
                        n_input,
                        n_latent,
                        n_layers=n_layers,
                        n_hidden=n_hidden,
                        dropout_rate=dropout_rate,
                    )
                    for key in multi_encoder_keys
                }
            )
        else:
            self.encoder = nn.ModuleDict(
                {key: LinearEncoder(n_input, n_latent) for key in multi_encoder_keys}
            )

        self.A = torch.from_numpy(np.array(A_param, dtype=np.float32)).cuda()

        self.log_det_pxz = torch.tensor(
            pxz_log_det, requires_grad=False, dtype=torch.float
        ).cuda()
        self.inv_sqrt_pxz = torch.from_numpy(
            np.array(pxz_inv_sqrt, dtype=np.float32)
        ).cuda()

        self.learn_var = learn_var
        self.px_log_diag_var = torch.nn.Parameter(torch.randn(1, n_input))

        self.gamma = torch.from_numpy(np.array(np.diag(gamma), dtype=np.float32)).cuda()
        log_det = np.log(np.linalg.det(gamma))
        self.log_det_px_z = torch.tensor(
            log_det, requires_grad=False, dtype=torch.float
        ).cuda()
        inv_sqrt = sqrtm(np.linalg.inv(gamma))
        self.inv_sqrt_px_z = torch.from_numpy(
            np.array(inv_sqrt, dtype=np.float32)
        ).cuda()

    def get_std(self):
        return torch.sqrt(torch.exp(self.px_log_diag_var))

    def get_latents(self, x):
        r""" returns the result of ``sample_from_posterior`` inside a list

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :return: one element list of tensor
        :rtype: list of :py:class:`torch.Tensor`
        """
        return [self.sample_from_posterior(x)]

    def sample_from_posterior(self, x, give_mean=False, encoder_key="default"):
        r""" samples the tensor of latent values from the posterior
        #doesn't really sample, returns the means of the posterior distribution

        :param x: tensor of values with shape ``(batch_size, n_input)``
        :param give_mean: is True when we want the mean of the posterior  distribution rather than sampling
        :return: tensor of shape ``(batch_size, n_latent)``
        :rtype: :py:class:`torch.Tensor`
        """
        qz_vars = self.encoder[encoder_key](x, None)  # y only used in VAEC
        qz_m = qz_vars["q_m"]
        z = qz_vars["latent"]
        if give_mean:
            z = qz_m
        return z

    def inference(
        self, x, n_samples=1, reparam: bool = True, encoder_key="default", counts=None
    ):
        assert not self.linear_encoder
        # Sampling
        if encoder_key == "defensive":
            batch_size = x.shape[0]
            device = x.device
            q_cubo = self.encoder["CUBO"](x, None)
            q_m_cubo = q_cubo["q_m"]
            q_v_cubo = q_cubo["q_v"]
            q_cubo = Normal(q_m_cubo, q_v_cubo.sqrt())
            z_cubo = q_cubo.sample((counts[0],))

            q_eubo = self.encoder["EUBO"](x, None)
            q_m_eubo = q_eubo["q_m"]
            q_v_eubo = q_eubo["q_v"]
            q_eubo = Normal(q_m_eubo, q_v_eubo.sqrt())
            z_eubo = q_eubo.sample((counts[1],))

            q_m_prior = torch.zeros(self.n_latent, device=device)
            q_v_prior = torch.ones(self.n_latent, device=device)
            q_prior = Normal(q_m_prior, q_v_prior)
            z_prior = q_prior.sample((counts[2], batch_size))

            z = torch.cat([z_cubo, z_eubo, z_prior])
            distribs = [q_cubo, q_eubo, q_prior]
            p_alpha = 1.0 * counts / counts.sum()
            log_p_alpha = p_alpha.log()

            log_contribs = []
            for (log_p_a, count, distrib) in zip(log_p_alpha, counts, distribs):
                if count >= 1:
                    log_contribs.append(
                        (distrib.log_prob(z).sum(-1) + log_p_a).unsqueeze(-1)
                    )
            log_contribs = torch.cat(log_contribs, dim=-1)
            # Those should never be used directly to compute Gaussian probabilities
            # As the mixture distribution is obviously not Gaussian
            qz_m = (
                p_alpha[0] * q_m_cubo + p_alpha[1] * q_m_eubo + p_alpha[2] * q_m_prior
            )
            qz_v = (
                p_alpha[0] * q_v_cubo + p_alpha[1] * q_v_eubo + p_alpha[2] * q_v_prior
            )
            log_qz_given_x = torch.logsumexp(log_contribs, dim=-1)
        else:
            # Case in which you consider a single variational distribution
            qz_vars = self.encoder[encoder_key](x, None)
            qz_m = qz_vars["q_m"]
            qz_v = qz_vars["q_v"]
            z = qz_vars["latent"]
            post_dist = Normal(qz_m, qz_v.sqrt())

            if n_samples > 1:
                if reparam:
                    z = post_dist.rsample((n_samples,))
                else:
                    z = post_dist.sample((n_samples,))
            log_qz_given_x = post_dist.log_prob(z).sum(-1)

        px_mean = torch.matmul(z, torch.transpose(self.A, 0, 1))
        return px_mean, torch.exp(self.px_log_diag_var), qz_m, qz_v, z, log_qz_given_x

    def log_px_z(self, tensors, z):
        """
            Usage reserved for Annealed Importance sampling
        """
        n_latent = z.shape[-1]
        z_prior = Normal(
            torch.zeros(n_latent, device="cuda"), torch.ones(n_latent, device="cuda")
        )
        log_pz = z_prior.log_prob(z).sum(-1)

        x = tensors[0]
        px_mean, px_var, qz_m, qz_v, _, log_qz_given_x = self.inference(
            x, n_samples=1, reparam=True
        )
        # Following step required to be consistent with below method
        z = z.unsqueeze(0)
        _, log_pxz, _ = self.log_ratio(
            x, px_mean, px_var, log_qz_given_x, z, return_full=True
        )
        log_pxz = log_pxz.squeeze()
        return log_pxz - log_pz

    def log_ratio(self, x, px_mean, px_var, log_qz_given_x, z, return_full=False):
        if self.learn_var:
            log_px_z = (
                Normal(px_mean, torch.sqrt(px_var))
                .log_prob(x.repeat(px_mean.shape[0], 1, 1))
                .sum(dim=-1)
                .view((px_mean.shape[0], -1))
            )
            log_pz = (
                Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
            )
            log_pxz = log_px_z + log_pz

        else:
            zx = torch.cat([z, x.repeat(z.shape[0], 1, 1)], dim=-1)
            reshape_dim = x.shape[-1] + z.shape[-1]

            log_pxz = self.log_normal_full(
                zx.view((-1, reshape_dim)),
                torch.zeros_like(zx.view((-1, reshape_dim))),
                self.log_det_pxz,
                self.inv_sqrt_pxz,
            ).view((z.shape[0], -1))

        log_ratio = log_pxz - log_qz_given_x
        if return_full:
            return log_ratio, log_pxz, log_qz_given_x
        else:
            return log_ratio

    def neg_iwelbo(self, x, n_samples_mc, encoder_key="default", counts=None):
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x,
            n_samples=n_samples_mc,
            reparam=True,
            encoder_key=encoder_key,
            counts=counts,
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, log_qz_given_x, z)
        iwelbo = torch.logsumexp(log_ratio, dim=0) - np.log(n_samples_mc)
        return -iwelbo

    def neg_iwelbo_grad(self, x, n_samples_mc, encoder_key="default", counts=None):
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x,
            n_samples=n_samples_mc,
            reparam=True,
            encoder_key=encoder_key,
            counts=counts,
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, log_qz_given_x, z)
        iwelbo = torch.softmax(log_ratio, dim=0).detach() * log_ratio
        return -iwelbo.sum(dim=0)

    def neg_elbo(self, x, encoder_key="default", counts=None):
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x, n_samples=2, reparam=True, encoder_key=encoder_key, counts=counts
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, log_qz_given_x, z)
        neg_elbo = -log_ratio.mean(dim=0)
        return neg_elbo

    def cubo(self, x, n_samples_mc, encoder_key="default", counts=None):
        # computes the naive cubo from chi2 upper bound
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x,
            n_samples=n_samples_mc,
            reparam=False,
            encoder_key=encoder_key,
            counts=counts,
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, log_qz_given_x, z)
        cubo = torch.logsumexp(2 * log_ratio, dim=0) - np.log(n_samples_mc)
        return 0.5 * cubo

    def cubo_grad(self, x, n_samples_mc, encoder_key="default", counts=None):
        # computes the importance sampled objective for reverse KL EP (revisited reweighted wake-sleep)
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x,
            n_samples=n_samples_mc,
            reparam=True,
            encoder_key=encoder_key,
            counts=counts,
        )
        log_ratio, _, log_qz_given_x = self.log_ratio(
            x, px_mean, px_var, log_qz_given_x, z, return_full=True
        )
        ws = torch.softmax(2 * log_ratio, dim=0)
        cubo = ws.detach() * (-1) * log_ratio
        # print(ws[:, 0])
        return cubo.sum(dim=0)

    def iwrevkl_obj(self, x, n_samples_mc, encoder_key="default", counts=None):
        # computes the importance sampled objective for reverse KL EP (revisited reweighted wake-sleep)
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x,
            n_samples=n_samples_mc,
            reparam=False,
            encoder_key=encoder_key,
            counts=counts,
        )
        log_ratio, _, log_qz_given_x = self.log_ratio(
            x, px_mean, px_var, log_qz_given_x, z, return_full=True
        )
        ws = torch.softmax(log_ratio, dim=0)
        rev_kl = ws.detach() * (-1) * log_qz_given_x
        return rev_kl.sum(dim=0)

    def vr_max(self, x, n_samples_mc, encoder_key="default", counts=None):
        # computes the naive MC from VR-max bound
        # tile vectors from (B, d) to (n_samples_mc, B, d)
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x,
            n_samples=n_samples_mc,
            reparam=True,
            encoder_key=encoder_key,
            counts=counts,
        )
        log_ratio = self.log_ratio(x, px_mean, px_var, log_qz_given_x, z)
        return -log_ratio.max(dim=0)[0]

    def generate_prior_data(self):
        shape_z = (128, self.n_latent)
        z = Normal(torch.zeros(shape_z).cuda(), torch.ones(shape_z).cuda()).sample()
        px_mean = torch.matmul(z, torch.transpose(self.A, 0, 1))
        if self.learn_var:
            px_var = torch.exp(self.px_log_diag_var)
        else:
            px_var = self.gamma
        x = Normal(px_mean, torch.sqrt(px_var)).sample()
        return x, z

    def forward(self, x, param, encoder_key="default", counts=None):
        if param == "ELBO":
            return self.neg_elbo(x, encoder_key=encoder_key, counts=counts)
        if param == "CUBO":
            return self.cubo_grad(
                x, n_samples_mc=10, encoder_key=encoder_key, counts=counts
            )
        if param == "REVKL":
            return self.iwrevkl_obj(
                x, n_samples_mc=20, encoder_key=encoder_key, counts=counts
            )
        if param == "IWELBO":
            return self.neg_iwelbo(
                x, n_samples_mc=80, encoder_key=encoder_key, counts=counts
            )
        if param == "VRMAX":
            return self.vr_max(
                x, n_samples_mc=20, encoder_key=encoder_key, counts=counts
            )

    def joint_log_likelihood(self, xz, pxz_mean, pxz_var):
        if self.learn_var:
            return Normal(pxz_mean, torch.sqrt(pxz_var)).log_prob(xz).sum(dim=1)
        else:
            return self.log_normal_full(
                xz, pxz_mean, self.log_det_pxz, self.inv_sqrt_pxz
            )

    @staticmethod
    def log_normal_full(x, mean, log_det, inv_sqrt):
        # copying code from NUMPY
        d = x.shape[1]

        log_lik = torch.zeros((x.shape[0],), dtype=torch.float).cuda()
        log_lik += d * np.log(2 * np.array(np.pi, dtype=np.float32))
        log_lik += log_det
        vec_ = torch.matmul(x - mean, inv_sqrt)
        log_lik += torch.mul(vec_, vec_).sum(dim=-1)
        return -0.5 * log_lik

    @torch.no_grad()
    def prob_event(
        self,
        x,
        n_samples_mc,
        nu: Union[float, list, np.ndarray] = 0.0,
        encoder_key="default",
        counts=None,
    ):
        px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.inference(
            x, n_samples=n_samples_mc, counts=counts, encoder_key=encoder_key
        )

        # compute for importance sampling
        log_ratio = self.log_ratio(x, px_mean, px_var, log_qz_given_x, z)
        ratio = torch.exp(log_ratio - torch.max(log_ratio, dim=0)[0])

        # get SNIPS estimator
        if hasattr(nu, "__len__"):
            res = []
            for nu_item in nu:
                res_item = torch.sum(
                    ratio * (z[:, :, 0] <= nu_item).float(), dim=0
                ) / torch.sum(ratio, dim=0)
                res.append(res_item.view(-1, 1))
            res = torch.cat(res, dim=-1)
        else:
            res = torch.sum(ratio * (z[:, :, 0] <= nu).float(), dim=0) / torch.sum(
                ratio, dim=0
            )

        # get ESS
        ess = torch.sum(ratio, dim=0) ** 2 / torch.sum(ratio ** 2, dim=0)

        # mean_to_return = qz_m.mean(dim=0)
        # variance_to_return = qz_v.mean(dim=0)
        mean_to_return = qz_m
        variance_to_return = qz_v
        return mean_to_return, variance_to_return, res, ess
