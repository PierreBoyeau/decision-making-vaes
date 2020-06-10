import logging

import numpy as np
import torch
import torch.nn as nn

# from torch.distributions import MultivariateNormal, Normal
import torch.distributions as db

from sbvae.models.distributions import EllipticalStudent

logger = logging.getLogger(__name__)


class FCLayersA(nn.Module):
    def __init__(
        self, n_input, n_output, n_middle=None, dropout_rate=0.1, do_batch_norm=False
    ):
        super().__init__()
        n_middle = n_output if n_middle is None else n_middle
        self.to_hidden = nn.Linear(in_features=n_input, out_features=n_middle)
        self.do_batch_norm = do_batch_norm
        if do_batch_norm:
            logger.info("Performing batch normalization")
            self.batch_norm = nn.BatchNorm1d(num_features=n_middle)
        else:
            logger.info("No batch normalization")

        self.to_out = nn.Linear(in_features=n_middle, out_features=n_output)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.SELU()
        # self.activation = nn.ReLU()

    def forward(self, x):
        res = self.to_hidden(x)
        if self.do_batch_norm:
            if res.ndim == 4:
                n1, n2, n3, n4 = res.shape
                res = self.batch_norm(res.view(n1 * n2 * n3, n4))
                res = res.view(n1, n2, n3, n4)
            elif res.ndim == 3:
                n1, n2, n3 = res.shape
                res = self.batch_norm(res.view(n1 * n2, n3))
                res = res.view(n1, n2, n3)
            elif res.ndim == 2:
                res = self.batch_norm(res)
            else:
                raise ValueError("{} ndim not handled.".format(res.ndim))
        res = self.activation(res)
        res = self.dropout(res)
        res = self.to_out(res)
        return res


class EncoderA(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        super().__init__()
        logging.info("Using MF encoder")
        self.encoder = FCLayersA(
            n_input=n_input,
            n_output=n_hidden,
            dropout_rate=dropout_rate,
            do_batch_norm=do_batch_norm,
            n_middle=n_middle,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples, squeeze=True, reparam=True):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)

        # q_v = 16.0 * self.tanh(q_v)
        # q_v = torch.clamp(q_v, min=-17., max=14.)

        # PREVIOUS TO KEEP
        # q_m = torch.clamp(q_m, min=-1000, max=1000)

        # q_v = torch.clamp(q_v, min=-17.0, max=8.0)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        # q_v = 1e-16 + q_v.exp()

        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )


class EncoderB(nn.Module):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        # TODO: describe architecture and choice for people 
        super().__init__()
        logging.info("Using MF encoder with convolutions")
        self.encoder_cv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=64, out_channels=n_hidden, kernel_size=3),
            nn.SELU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.1),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.SELU(),
            nn.Dropout(p=0.1),
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    def forward(self, x, n_samples, squeeze=True, reparam=True):
        n_batch = len(x)
        x_reshape = x.view(n_batch, 1, 28, 28)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()
        variational_dist = db.Normal(loc=q_m, scale=q_v.sqrt())

        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=variational_dist, sum_last=True
        )


class EncoderBStudent(EncoderB):
    def __init__(
        self, n_input, n_output, n_hidden, dropout_rate, do_batch_norm, n_middle=None
    ):
        super().__init__(
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            do_batch_norm=do_batch_norm,
            n_middle=n_middle,
        )
        self.df_fn = nn.Linear(n_hidden, 1)

    def forward(self, x, n_samples, squeeze=True, reparam=True):
        n_batch = len(x)
        x_reshape = x.view(n_batch, 1, 28, 28)

        q = self.encoder_cv(x_reshape)
        q = q.view(n_batch, -1)

        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()

        # df = self.df_fn(q)
        df = 1.0 + self.df_fn(q).exp()
        df = torch.clamp(df, max=1e5)

        # print("df : ", df.shape)
        # print("q_m : ", q_m.shape)
        # print("q_v : ", q_v.shape)

        st_dist = EllipticalStudent(df=df, loc=q_m, scale=torch.sqrt(q_v))
        # if n_samples == 1 and squeeze:
        #     sample_shape = []
        # else:
        #     sample_shape = (n_samples,)
        # latent = self.reparameterize(
        #     st_dist, sample_shape=sample_shape, reparam=reparam
        # )
        # return dict(
        #     q_m=q_m, q_v=q_v, df=df, dist=st_dist, latent=latent, sum_last=False
        # )
        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = st_dist.rsample(sample_shape=sample_shape)
        else:
            latent = st_dist.sample(sample_shape=sample_shape)
        return dict(
            q_m=q_m, q_v=q_v, latent=latent, dist=st_dist, sum_last=False, df=df
        )


class EncoderAStudent(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        dropout_rate,
        do_batch_norm,
        n_middle=None,
        df="learn",
    ):
        super().__init__()
        logging.info("Using Student encoder")
        if df == "learn":
            self.df_val = nn.Linear(n_hidden, 1)
            nn.init.zeros_(self.df_val.weight)
            nn.init.zeros_(self.df_val.bias)
            self.learn_df = True
        else:
            self.df_val = df
            self.learn_df = False
        self.encoder = FCLayersA(
            n_input=n_input,
            n_output=n_hidden,
            dropout_rate=dropout_rate,
            do_batch_norm=do_batch_norm,
            n_middle=n_middle,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.tanh = nn.Tanh()

    # @property
    def df(self, q: torch.Tensor):
        if self.learn_df:
            df_use = 1.0 + self.df_val(q).exp()
            df_use = torch.clamp(df_use, max=1e5)
            return df_use
        else:
            return torch.tensor(self.df_val, device="cuda")

    def reparameterize(self, dist, sample_shape=torch.Size(), reparam=True):
        if reparam:
            latent = dist.rsample(sample_shape=sample_shape)
        else:
            latent = dist.sample(sample_shape=sample_shape)
        return latent

    def forward(self, x, n_samples, squeeze=True, reparam=True):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_encoder(q)
        q_v = torch.clamp(q_v, min=-17.0, max=10.0)
        q_v = q_v.exp()

        df_to_use = self.df(q)
        st_dist = EllipticalStudent(df=df_to_use, loc=q_m, scale=torch.sqrt(q_v))
        if n_samples == 1 and squeeze:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        latent = self.reparameterize(
            st_dist, sample_shape=sample_shape, reparam=reparam
        )
        return dict(
            q_m=q_m, q_v=q_v, df=df_to_use, dist=st_dist, latent=latent, sum_last=False
        )
        # return dict(q_m=q_m, q_v=q_v, latent=latent)


class LinearEncoder(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.mean_encoder = nn.Linear(n_input, n_output)
        self.n_output = n_output

        self.var_vals = nn.Parameter(
            0.1 * torch.rand(n_output, n_output), requires_grad=True
        )

    @property
    def l_mat_encoder(self):
        l_mat = torch.tril(self.var_vals)
        range_vals = np.arange(self.n_output)
        l_mat[range_vals, range_vals] = l_mat[range_vals, range_vals].exp()
        return l_mat

    @property
    def var_encoder(self):
        l_mat = self.l_mat_encoder
        return l_mat.matmul(l_mat.T)

    def forward(self, x, n_samples, reparam=True, squeeze=True):
        q_m = self.mean_encoder(x)
        l_mat = self.var_encoder
        q_v = l_mat.matmul(l_mat.T)

        variational_dist = db.MultivariateNormal(loc=q_m, scale_tril=l_mat)

        if squeeze and n_samples == 1:
            sample_shape = []
        else:
            sample_shape = (n_samples,)
        if reparam:
            latent = variational_dist.rsample(sample_shape=sample_shape)
        else:
            latent = variational_dist.sample(sample_shape=sample_shape)
        return dict(q_m=q_m, q_v=q_v, latent=latent)


# Decoder
class DecoderA(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int):
        super().__init__()
        self.decoder = FCLayersA(n_input=n_input, n_output=n_hidden, dropout_rate=0.0)

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

        self.tanh = nn.Tanh()

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

        # PREVIOUS TO KEEP
        # p_m = torch.clamp(p_m, min=-1000, max=1000)

        # p_v = torch.clamp(p_v, min=-17.0, max=10)
        p_v = torch.clamp(p_v, min=-17.0, max=8)

        # p_v = torch.clamp(p_v, min=-17.0, max=14.0)
        # p_v = 16. * self.tanh(p_v)
        return p_m, p_v.exp()


class ClassifierA(nn.Module):
    def __init__(self, n_input, n_output, dropout_rate=0.0, do_batch_norm=False):
        super().__init__()
        self.classifier = nn.Sequential(
            FCLayersA(
                n_input,
                n_output,
                dropout_rate=dropout_rate,
                do_batch_norm=do_batch_norm,
            ),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        probas = self.classifier(x)
        probas = probas + 1e-16
        probas = probas / probas.sum(-1, keepdim=True)
        return probas    


class BernoulliDecoderA(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        dropout_rate: float = 0.0,
        do_batch_norm: bool = True,
    ):
        super().__init__()
        self.loc = FCLayersA(
            n_input, n_output, dropout_rate=dropout_rate, do_batch_norm=do_batch_norm
        )

    def forward(self, x):
        means = self.loc(x)
        means = nn.Sigmoid()(means)
        return means


class EncoderH(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        # n_cat_list,
        # n_layers,
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
        self.encoder = FCLayersA(
            n_in=n_in,
            n_out=n_hidden,
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

        self.dist0 = db.Normal(
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
        return dict(latent=z, posterior_density=qz_x, last_inp=inp)
