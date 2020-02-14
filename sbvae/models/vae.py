# -*- coding: utf-8 -*-
"""Main module."""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma, Normal, Poisson

from sbvae.models.log_likelihood import log_nb_positive, log_zinb_positive
from sbvae.models.modules import DecoderSCVI, Encoder, EncoderIAF
from sbvae.models.utils import one_hot

logger = logging.getLogger(__name__)


torch.backends.cudnn.benchmark = True


# VAE model
class VAE(nn.Module):
    r"""Variational auto-encoder model.

    :param n_input: Number of input genes
    :param n_batch: Number of batches
    :param n_labels: Number of labels
    :param n_hidden: Number of nodes per hidden layer
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param dropout_rate: Dropout rate for neural networks
    :param dispersion: One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell

    :param log_variational: Log variational distribution
    :param reconstruction_loss:  One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        iaf_t: int = 0,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        prevent_library_saturation: bool = False,
        prevent_library_saturation2: bool = False,
        multi_encoder_keys=["default"],
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_latent_layers = 1  # not sure what this is for, no usages?
        self.multi_encoder_keys = multi_encoder_keys
        do_multi_encoders = len(multi_encoder_keys) >= 2
        z_prior_mean = torch.zeros(n_latent, device="cuda")
        z_prior_std = torch.ones(n_latent, device="cuda")
        self.z_prior = Normal(z_prior_mean, z_prior_std)

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.do_iaf = iaf_t > 0
        if not self.do_iaf:
            logger.info("using MF encoder")
            self.z_encoder = nn.ModuleDict(
                {
                    key: Encoder(
                        n_input,
                        n_latent,
                        n_layers=n_layers,
                        n_hidden=n_hidden,
                        dropout_rate=dropout_rate,
                    )
                    for key in self.multi_encoder_keys
                }
            )
        else:
            logger.info("using IAF encoder")
            assert not do_multi_encoders
            self.z_encoder = EncoderIAF(
                n_in=n_input,
                n_latent=n_latent,
                n_cat_list=None,
                n_hidden=n_hidden,
                n_layers=n_layers,
                t=iaf_t,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
                do_h=True,
            )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            prevent_saturation=prevent_library_saturation,
            prevent_saturation2=prevent_library_saturation2,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

        assert not self.do_iaf

    @torch.no_grad()
    def z_defensive_sampling(self, x, counts):
        """
            Samples from q_alpha
            q_alpha = \alpha_0 q_CUBO + \alpha_1 q_EUBO + \alpha_2 prior
        """
        n_samples_total = counts.sum()
        n_batch, _ = x.shape
        # with torch.no_grad():
        post_cubo = self.z_encoder["CUBO"](
            x=x, n_samples=counts[0], reparam=False, squeeze=False
        )
        if counts[0] >= 1:
            z_cubo = post_cubo["latent"]
            q_cubo = Normal(post_cubo["q_m"][0], post_cubo["q_v"][0].sqrt())
        else:
            # Specific handling of counts=0 required for latter concatenation
            z_cubo = torch.tensor([], device="cuda")
            q_cubo = None

        post_eubo = self.z_encoder["EUBO"](
            x=x, n_samples=counts[1], reparam=False, squeeze=False
        )
        if counts[1] >= 1:
            z_eubo = post_eubo["latent"]
            q_eubo = Normal(post_eubo["q_m"][0], post_eubo["q_v"][0].sqrt())
        else:
            # Specific handling of counts=0 required for latter concatenation
            z_eubo = torch.tensor([], device="cuda")
            q_eubo = None

        z_prior = self.z_prior.sample((counts[2], n_batch))
        q_prior = self.z_prior

        z_all = torch.cat([z_cubo, z_eubo, z_prior], dim=0)
        distribs_all = [q_cubo, q_eubo, q_prior]
        # Mixture probability
        # q_alpha = sum p(alpha_j) * q_j(x)
        log_p_alpha = (1.0 * counts / counts.sum()).log()

        log_contribs = []
        for count, distrib, log_p_a in zip(counts, distribs_all, log_p_alpha):
            if count >= 1:
                contribution = distrib.log_prob(z_all).sum(-1) + log_p_a
                contribution = contribution.view(n_samples_total, n_batch, 1)
                log_contribs.append(contribution)

        log_q_alpha = torch.cat(log_contribs, dim=2)
        log_q_alpha = torch.logsumexp(log_q_alpha, dim=2)
        return dict(latent=z_all, posterior_density=log_q_alpha)

    def _reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        return reconst_loss

    def scale_from_z(self, sample_batch, fixed_batch):
        if self.log_variational:
            sample_batch = torch.log(1 + sample_batch)
        z = self.z_encoder(sample_batch)["latent"]
        batch_index = torch.cuda.IntTensor(sample_batch.shape[0], 1).fill_(fixed_batch)
        library = torch.cuda.FloatTensor(sample_batch.shape[0], 1).fill_(4)
        px_scale, _, _, _ = self.decoder("gene", z, library, batch_index)
        return px_scale

    def inference(
        self,
        x,
        batch_index=None,
        y=None,
        n_samples=1,
        reparam=True,
        observed_library=None,
        encoder_key: str = "default",
        counts: torch.Tensor = None,
    ):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Library sampling
        library_post = self.l_encoder(x_, n_samples=n_samples, reparam=reparam)
        library_variables = dict(
            ql_m=library_post["q_m"],
            ql_v=library_post["q_v"],
            library=library_post["latent"],
        )

        if observed_library is None:
            library = library_variables["library"]
        else:
            library = observed_library

        # Z sampling
        if encoder_key != "defensive":
            z_post = self.z_encoder[encoder_key](
                x_, y, n_samples=n_samples, reparam=reparam
            )
        else:
            z_post = self.z_defensive_sampling(x_, counts=counts)

        if self.do_iaf or encoder_key == "defensive":
            # IAF does not parametrize the means/covariances of the variational posterior
            z_variables = dict(
                qz_m=None,
                qz_v=None,
                z=z_post["latent"],
                log_qz_x=z_post["posterior_density"],
            )
        else:
            z_variables = dict(
                qz_m=z_post["q_m"],
                qz_v=z_post["q_v"],
                z=z_post["latent"],
                log_qz_x=None,
            )

        # Decoder pass
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z_post["latent"], library, batch_index, y
        )

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)
        decoder_variables = dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

        return {**decoder_variables, **library_variables, **z_variables}

    def from_variables_to_densities(
        self,
        x,
        local_l_mean,
        local_l_var,
        px_r,
        px_rate,
        px_dropout,
        z,
        library,
        px_scale=None,
        qz_m=None,
        qz_v=None,
        ql_m=None,
        ql_v=None,
        log_qz_x=None,
        log_ql_x=None,
    ):
        """
        Unifies VAE outputs to construct loss

        :param x:
        :param local_l_mean:
        :param local_l_var:
        :param px_r:
        :param px_rate:
        :param px_dropout:
        :param qz_m:
        :param qz_v:
        :param z:
        :param ql_m:
        :param ql_v:
        :param library:
        :param log_qz_x:
        :param log_ql_x:
        :return:
        """
        log_px_zl = (-1) * self._reconstruction_loss(x, px_rate, px_r, px_dropout)
        if log_qz_x is None:
            log_qz_x = Normal(qz_m, torch.sqrt(qz_v)).log_prob(z).sum(dim=-1)
        if log_ql_x is None:
            log_ql_x = Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)
        log_pz = Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
        log_pl = (
            Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library).sum(dim=-1)
        )

        a1_issue = torch.isnan(log_px_zl).any() or torch.isinf(log_px_zl).any()
        a2_issue = torch.isnan(log_pl).any() or torch.isinf(log_pl).any()
        a3_issue = torch.isnan(log_pz).any() or torch.isinf(log_pz).any()
        a4_issue = torch.isnan(log_qz_x).any() or torch.isinf(log_qz_x).any()
        a5_issue = torch.isnan(log_ql_x).any() or torch.isinf(log_ql_x).any()

        if a1_issue or a2_issue or a3_issue or a4_issue or a5_issue:
            print("aie")

        return dict(
            log_px_zl=log_px_zl,
            log_pl=log_pl,
            log_pz=log_pz,
            log_qz_x=log_qz_x,
            log_ql_x=log_ql_x,
        )

    def log_px_z(self, tensors, z):
        """
            Only works in the specific case where the library is observed and there are no batch indices
        """
        (x, _, _, batch_index, _) = tensors
        library = x.sum(1, keepdim=True)

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z, library, batch_index
        )
        if self.dispersion == "gene-label":
            raise ValueError
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)
        res = (-1) * self._reconstruction_loss(x, px_rate, px_r, px_dropout)
        return res

    @torch.no_grad()
    def generate_joint(
        self, x, local_l_mean, local_l_var, batch_index, y=None, zero_inflated=True
    ):
        """
        :param x: used only for shape match
        """
        n_batches, _ = x.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"
        z_mean = torch.zeros(n_batches, self.n_latent, device=device)
        z_std = torch.zeros(n_batches, self.n_latent, device=device)
        z_prior_dist = Normal(z_mean, z_std)
        z_sim = z_prior_dist.sample()

        l_prior_dist = Normal(local_l_mean, torch.sqrt(local_l_var))
        l_sim = l_prior_dist.sample()

        # Decoder pass
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z_sim, l_sim, batch_index, y
        )

        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        # Data generation
        p = px_rate / (px_rate + px_r)
        r = px_r
        # Important remark: Gamma is parametrized by the rate = 1/scale!
        l_train = Gamma(concentration=r, rate=(1 - p) / p).sample()

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(l_train, max=1e8)
        gene_expressions = Poisson(
            l_train
        ).sample()  # Shape : (n_samples, n_cells_batch, n_genes)
        if zero_inflated:
            p_zero = (1.0 + torch.exp(-px_dropout)).pow(-1)
            random_prob = torch.rand_like(p_zero)
            gene_expressions[random_prob <= p_zero] = 0

        return gene_expressions, z_sim, l_sim

    def forward(
        self,
        x,
        local_l_mean,
        local_l_var,
        batch_index=None,
        loss_type="ELBO",
        y=None,
        n_samples=1,
        reparam=True,
        do_observed_library=False,
        counts=None,
        encoder_key="default",
    ):
        r""" Returns the reconstruction loss and the Kullback divergences

        :param x: tensor of values with shape (batch_size, n_input)
        :param local_l_mean: tensor of means of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param local_l_var: tensor of variances of the prior distribution of latent variable l
         with shape (batch_size, 1)
        :param batch_index: array that indicates which batch the cells belong to with shape ``batch_size``
        :param y: tensor of cell-types labels with shape (batch_size, n_labels)
        :param loss_type:
        :param n_samples:
        :param reparam:
        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution
        observed_library = None
        if do_observed_library:
            observed_library = x.sum(1, keepdim=True)

        variables = self.inference(
            x,
            batch_index=batch_index,
            y=y,
            n_samples=n_samples,
            reparam=reparam,
            observed_library=observed_library,
            counts=counts,
            encoder_key=encoder_key,
        )
        op = self.from_variables_to_densities(
            x=x, local_l_mean=local_l_mean, local_l_var=local_l_var, **variables
        )

        if do_observed_library:
            log_ratio = op["log_px_zl"] + op["log_pz"] - op["log_qz_x"]
            sum_log_q = op["log_qz_x"]
        else:
            log_ratio = (
                op["log_px_zl"]
                + op["log_pz"]
                + op["log_pl"]
                - op["log_qz_x"]
                - op["log_ql_x"]
            )
            sum_log_q = op["log_qz_x"] + op["log_ql_x"]

        z_log_ratio = op["log_px_zl"] + op["log_pz"] - op["log_qz_x"]

        if loss_type == "ELBO":
            loss = -log_ratio.mean(dim=0)
        elif loss_type == "REVKL":
            loss = self.forward_kl(log_ratio=log_ratio, sum_log_q=sum_log_q)
        elif loss_type == "CUBO":
            loss = self.cubo(log_ratio=log_ratio)
        elif loss_type == "IWELBO":
            assert n_samples >= 2
            loss = self.iwelbo(log_ratio)
        else:
            cubo_loss = torch.logsumexp(2 * log_ratio, dim=0) - np.log(n_samples)
            cubo_loss = 0.5 * cubo_loss
            iwelbo_loss = torch.logsumexp(log_ratio, dim=0) - np.log(n_samples)
            return {
                "log_ratio": log_ratio,
                "z_log_ratio": z_log_ratio,
                "CUBO": cubo_loss,
                "IWELBO": iwelbo_loss,
                "debug_log_qz_x": op["log_qz_x"],
                **op,
                **variables,
            }

        return loss

    @staticmethod
    def iwelbo(log_ratio):
        return -(torch.softmax(log_ratio, dim=0).detach() * log_ratio).sum(dim=0)

    @staticmethod
    def cubo(log_ratio):
        """
        Algorithm 1 of
              nan,
        With reparameterized gradient!!!
        https://arxiv.org/pdf/1611.00328.pdf

        :param log_ratio:
        :return:
        """
        ws = torch.softmax(2 * log_ratio, dim=0)  # Corresponds to squaring
        cubo = ws.detach() * (-1) * log_ratio
        return cubo.sum(dim=0)

    @staticmethod
    def forward_kl(log_ratio, sum_log_q):
        ws = torch.softmax(log_ratio, dim=0)
        rev_kl = ws.detach() * (-1) * sum_log_q
        return rev_kl.sum(dim=0)
