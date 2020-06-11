import collections
import logging
from typing import Iterable, Union

import numpy as np
import torch
import torch.distributions as dist
from torch import nn as nn
from torch.distributions import Normal

from sbvae.models.utils import one_hot
from sbvae.models.distributions import EllipticalStudent

logger = logging.getLogger(__name__)


class FCLayers(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_cat_list: A list containing, for each category of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm=True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_in,)``
        :param cat_list: list of category membership(s) for this sample
        :return: tensor of shape ``(n_out,)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(
            cat_list
        ), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (
                n_cat and cat is None
            ), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                            # shape n_post_samples, n_batch, n_features
                            # x = layer(x.transpose(-1, -2)).transpose(-1, -2)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x


# Encoder
class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        prevent_saturation: bool = False,
        prevent_saturation2: bool = False,
    ):
        super().__init__()
        self.prevent_saturation = prevent_saturation
        self.prevent_saturation2 = prevent_saturation2

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def reparameterize(self, mu, var, reparam=True):
        if reparam:
            latent = Normal(mu, var.sqrt()).rsample()
        else:
            latent = Normal(mu, var.sqrt()).sample()
        return latent

    def forward(
        self, x: torch.Tensor, *cat_list: int, n_samples=1, reparam=True, squeeze=True
    ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)

        if self.prevent_saturation:
            q_m = 12.0 * nn.Tanh()(q_m)
            q_v = 3.0 * nn.Sigmoid()(q_v)
            # q_m = torch.clamp(q_m, min=,)
            # q_v = torch.clamp(q_v, min=,)
        if self.prevent_saturation2:
            q_m = torch.clamp(q_m, max=16.0)
            q_v = torch.clamp(q_v, min=-18.0, max=0.1)
            q_v = torch.exp(q_v)
        else:
            q_v = torch.clamp(q_v, min=-12.0, max=8.0)
            q_v = torch.exp(
                self.var_encoder(q)
            )  # (computational stability safeguard)torch.clamp(, -5, 5)
        if (n_samples > 1) or (not squeeze):
            q_m = q_m.unsqueeze(0).expand((n_samples, q_m.size(0), q_m.size(1)))
            q_v = q_v.unsqueeze(0).expand((n_samples, q_v.size(0), q_v.size(1)))
        dist = Normal(q_m, q_v.sqrt())
        latent = self.reparameterize(q_m, q_v, reparam=reparam)
        return dict(
            q_m=q_m,
            q_v=q_v,
            latent=latent,
            posterior_density=None,
            dist=dist,
            sum_last=True,
        )


# Encoder
class EncoderStudent(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        df: Union[float, str] = 1.0,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose
        if df == "learn":
            self.df_val = nn.Linear(n_hidden, 1)
            nn.init.zeros_(self.df_val.weight)
            nn.init.zeros_(self.df_val.bias)
            self.learn_df = True
        else:
            self.df_val = df
            self.learn_df = False
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    # @property
    def df(self, q: torch.Tensor):
        if self.learn_df:
            df_use = 1.0 + self.df_val(q).exp()
            df_use = torch.clamp(df_use, max=1e5)
            return df_use
        else:
            return torch.tensor(self.df_val, device="cuda")

    def reparameterize(self, dist, reparam=True):
        if reparam:
            latent = dist.rsample()
        else:
            latent = dist.sample()
        return latent

    def forward(
        self, x: torch.Tensor, *cat_list: int, n_samples=1, reparam=True, squeeze=True
    ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = 1e-16 + self.var_encoder(q)
        df_to_use = self.df(q)  # .squeeze()

        q_v = torch.clamp(q_v, min=-18.0, max=14.0)
        q_v = 1e-16 + torch.exp(
            self.var_encoder(q)
        )  # (computational stability safeguard)torch.clamp(, -5, 5)
        if (n_samples > 1) or (not squeeze):
            q_m = q_m.unsqueeze(0).expand((n_samples, q_m.size(0), q_m.size(1)))
            q_v = q_v.unsqueeze(0).expand((n_samples, q_v.size(0), q_v.size(1)))

        st_dist = EllipticalStudent(
            df=df_to_use, loc=q_m, scale=torch.sqrt(q_v), verbose=self.verbose
        )
        # st_dist = StudentT(df=df_to_use, loc=q_m, scale=torch.sqrt(q_v))
        latent = self.reparameterize(st_dist, reparam=reparam)
        log_density = st_dist.log_prob(latent)  # .sum(dim=-1)
        return dict(
            q_m=q_m,
            q_v=q_v,
            latent=latent,
            posterior_density=log_density,
            df=df_to_use,
            dist=st_dist,
            sum_last=False,
        )


# Decoder
class DecoderSCVI(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm=use_batch_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param dispersion: One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :param cat_list: list of category membership(s) for this sample
        :return: parameters for the ZINB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        library_c = torch.clamp(library, max=12)
        px_rate = torch.exp(library_c) * px_scale  # torch.clamp( , max=12)
        # px_rate = torch.clamp(px_rate, max=1e12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm=use_batch_norm,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        :param x: tensor with shape ``(n_input,)``
        :param cat_list: list of category membership(s) for this sample
        :return: Mean and variance tensors of shape ``(n_output,)``
        :rtype: 2-tuple of :py:class:`torch.Tensor`
        """

        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = self.var_decoder(p)

        p_v = 16.0 * nn.Tanh()(p_v)
        return p_m, p_v.exp()


class BernoulliDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.loc = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x, *cat_list: int):
        means = self.loc(x, *cat_list)
        means = nn.Sigmoid()(means)
        return means


class EncoderH(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_cat_list,
        n_layers,
        n_hidden,
        do_h,
        dropout_rate,
        use_batch_norm,
        do_sigmoid=True,
    ):
        """
        :param n_in:
        :param n_out:
        :param n_cat_list:
        :param n_layers:
        :param n_hidden:
        :param do_h:
        :param dropout_rate:
        :param use_batch_norm:
        """
        super().__init__()
        self.do_h = do_h
        self.do_sigmoid = do_sigmoid
        self.encoder = FCLayers(
            n_in=n_in,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        # with torch.no_grad():
        #     encoder0 =
        self.mu = nn.Linear(n_hidden, n_out)
        self.sigma = nn.Linear(n_hidden, n_out)
        if do_sigmoid:
            self.init_weights(self.sigma, bias_val=1.5)
        else:
            self.init_weights(self.sigma, bias_val=0.5)
        if do_h:
            self.h = nn.Linear(n_hidden, n_out)

        self.activation = nn.Sigmoid()

    def forward(self, x, *cat_list: int):
        """
        :param x:
        :param cat_list:
        :return:
        """
        z = self.encoder(x, *cat_list)
        mu = self.mu(z)
        sigma = self.sigma(z)
        sigma = torch.clamp(sigma, min=-50.0)
        if self.do_sigmoid:
            sigma = self.activation(sigma)
        else:
            # sigma = nn.ReLU()(sigma)
            sigma = sigma.exp()
        if (sigma.min() == 0).item():
            print("tada")
        if self.do_h:
            h = self.h(z)
            return mu, sigma, h
        return mu, sigma

    @staticmethod
    def init_weights(m, bias_val=1.5):
        torch.nn.init.normal_(m.weight, mean=0.0, std=1e-8)
        torch.nn.init.constant_(m.bias, val=bias_val)


class EncoderIAF(nn.Module):
    def __init__(
        self,
        n_in,
        n_latent,
        n_cat_list,
        n_hidden,
        n_layers,
        t,
        dropout_rate=0.05,
        use_batch_norm=True,
        do_h=True,
    ):
        """
        Encoder using h representation as described in IAF paper
        :param n_in:
        :param n_latent:
        :param n_cat_list:
        :param n_hidden:
        :param n_layers:
        :param t:
        :param dropout_rate:
        :param use_batch_norm:
        """
        super().__init__()
        self.do_h = do_h
        msg = "" if do_h else "Not "
        logger.info(msg="{}Using Hidden State".format(msg))
        self.n_latent = n_latent
        self.encoders = torch.nn.ModuleList()
        self.encoders.append(
            EncoderH(
                n_in=n_in,
                n_out=n_latent,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                do_h=do_h,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                do_sigmoid=False,
            )
        )

        n_in = 2 * n_latent if do_h else n_latent
        for _ in range(t - 1):
            self.encoders.append(
                EncoderH(
                    n_in=n_in,
                    n_out=n_latent,
                    n_cat_list=None,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    do_h=False,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    do_sigmoid=True,
                )
            )

        self.dist0 = dist.Normal(
            loc=torch.zeros(n_latent, device="cuda"),
            scale=torch.ones(n_latent, device="cuda"),
        )

    def forward(self, x, *cat_list: int, n_samples=1, reparam=True):
        """
        :param x:
        :param cat_list:
        :return:
        """
        if self.do_h:
            mu, sigma, h = self.encoders[0](x, *cat_list)
        else:
            mu, sigma = self.encoders[0](x, *cat_list)
            h = None

        # Big issue when x is 3d !!!
        # Should stay 2d!!
        sampler = self.dist0.rsample if reparam else self.dist0.sample
        if n_samples == 1:
            eps = sampler((len(x),))
            assert eps.shape == (len(x), self.n_latent)
        else:
            eps = sampler((n_samples, len(x)))
            assert eps.shape == (n_samples, len(x), self.n_latent)

        z = mu + eps * sigma
        qz_x = sigma.log() + 0.5 * (eps ** 2) + 0.5 * np.log(2.0 * np.pi)
        qz_x = -qz_x.sum(dim=-1)

        # z shape (n_samples, n_batch, n_latent)
        if (z.dim() == 3) & (h.dim() == 2):
            n_batches, n_hdim = h.shape
            h_it = h.reshape(1, n_batches, n_hdim).expand(n_samples, n_batches, n_hdim)
        else:
            h_it = h

        for ar_nn in self.encoders[1:]:
            if self.do_h:
                inp = torch.cat([z, h_it], dim=-1)
            else:
                inp = z
            mu_it, sigma_it = ar_nn(inp)
            z = sigma_it * z + (1.0 - sigma_it) * mu_it
            new_term = sigma_it.log()
            qz_x -= new_term.sum(dim=-1)

        if torch.isnan(qz_x).any() or torch.isinf(qz_x).any():
            print("ouille")
        return dict(latent=z, posterior_density=qz_x, last_inp=inp, q_m=None, q_v=None)
