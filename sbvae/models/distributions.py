from numbers import Number

import numpy as np
import torch
import torch.distributions as db
from torch.distributions import constraints


class EllipticalStudent(db.Distribution):
    support = constraints.real
    has_rsample = True
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "df": constraints.positive,
    }

    def __init__(
        self,
        df: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args=None,
        verbose: bool = False,
    ):
        self.loc = loc
        self.scale = scale
        self.df = df
        self.d = self.loc.shape[-1]
        self.verbose = verbose

        if isinstance(df, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.df.size()

        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def cov(self):
        eye = torch.eye(self.d, device=self.loc.device)
        return eye * (self.scale ** 2.0).unsqueeze(-1)

    def scale_to_cov(self, scales):
        assert scales.min() > 1e-32, scales.min()
        eye = torch.eye(self.d, device=self.loc.device)
        return eye * (scales ** 2.0).unsqueeze(-1)

    def sample(self, sample_shape=torch.Size()):
        # # Step 1: get spherical sample u
        # zeros = torch.zeros_like(self.loc)
        # ones = torch.ones_like(self.loc)
        # sphere_samp = db.Normal(zeros, ones).sample(sample_shape)
        # norm_r = (sphere_samp ** 2.0).sum(-1, keepdim=True).sqrt()

        # sphere_samp = sphere_samp / norm_r
        # local_loc, local_scale, sphere_samp = torch.broadcast_tensors(
        #     self.loc, self.scale, sphere_samp
        # )
        # local_cov = self.scale_to_cov(local_scale)
        # scale_mat = torch.cholesky(local_cov)

        # # Step 2: sample radius
        # batch_shape = self.loc.shape[:-1]

        # t2_dist = db.Chi2(
        #     df=torch.tensor(self.d, device=self.loc.device).expand(batch_shape)
        # )
        # t_samp = t2_dist.sample(sample_shape)
        # t_samp = t_samp.sqrt()
        # s2_dist = db.Chi2(df=self.df.expand(batch_shape))
        # s_samp = s2_dist.rsample(sample_shape)
        # s_samp = s_samp.sqrt()
        # radius = self.df.sqrt() * t_samp / s_samp
        # radius = radius.unsqueeze(-1)

        # u = torch.matmul(scale_mat, sphere_samp.unsqueeze(-1)).squeeze(-1)

        # assert radius.shape[:-1] == u.shape[:-1]
        # samp = local_loc + (radius * u)
        # if self.verbose:
        #     print(
        #         "z range: {0:.2f} / {1:.2f}".format(
        #             samp.min().item(), samp.max().item(),
        #         )
        #     )
        # return samp

        # Step 1: get spherical sample u
        zeros = torch.zeros_like(self.loc)
        ones = torch.ones_like(self.loc)
        sphere_samp = db.Normal(zeros, ones).sample(sample_shape)
        norm_r = (sphere_samp ** 2.0).sum(-1, keepdim=True).sqrt()

        sphere_samp = sphere_samp / norm_r
        local_loc, local_scale, sphere_samp = torch.broadcast_tensors(
            self.loc, self.scale, sphere_samp
        )
        scale_mat = local_scale

        # Step 2: sample radius
        batch_shape = list(self.loc.shape)
        batch_shape[-1] = 1  # Radius = 1 parameter

        t2_dist = db.Chi2(
            df=torch.tensor(self.d, device=self.loc.device).expand(batch_shape)
        )
        t_samp = t2_dist.sample(sample_shape)
        t_samp = t_samp.sqrt()
        s2_dist = db.Chi2(df=self.df.expand(batch_shape))
        s_samp = s2_dist.rsample(sample_shape)
        s_samp = s_samp.sqrt()
        radius = self.df.sqrt() * t_samp / s_samp
        # radius = radius.unsqueeze(-1)
        u = scale_mat * sphere_samp

        assert radius.shape[:-1] == u.shape[:-1]
        samp = local_loc + (radius * u)
        if self.verbose:
            print(
                "z range: {0:.2f} / {1:.2f}".format(
                    samp.min().item(), samp.max().item(),
                )
            )
        return samp

    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape=sample_shape)

    def log_prob(self, value):
        """

        """

        local_loc, local_scale, value = torch.broadcast_tensors(
            self.loc, self.scale, value
        )
        # local_cov = self.scale_to_cov(local_scale)
        # logdet = torch.logdet(local_cov)
        # diff = (value - local_loc).unsqueeze(-1)
        # isoval = torch.matmul(local_cov.inverse(), diff)
        # isoval = torch.matmul(diff.transpose(-1, -2), isoval)
        # isoval = isoval.squeeze(-1)
        # isoval = isoval.squeeze(-1)

        # central_term = (
        #     -0.5 * (self.d + self.df) * (1.0 + (1.0 / self.df * isoval)).log()
        # )
        # res = (
        #     torch.lgamma(0.5 * (self.df + self.d))
        #     - torch.lgamma(0.5 * self.df)
        #     - 0.5 * self.d * (self.df.log() + np.log(np.pi))
        #     - 0.5 * logdet
        #     + central_term
        # )

        local_cov = (local_scale ** 2.0).unsqueeze(-1)
        logdet = local_scale.log().sum(-1, keepdims=True)
        logdet = 2.0 * logdet
        diff = (value - local_loc).unsqueeze(-1)
        isoval = (1.0 / local_cov) * diff
        isoval = torch.matmul(diff.transpose(-1, -2), isoval)
        isoval = isoval.squeeze(-1)
        # isoval = isoval.squeeze(-1)

        central_term = (
            -0.5 * (self.d + self.df) * (1.0 + (1.0 / self.df * isoval)).log()
        )
        res = (
            torch.lgamma(0.5 * (self.df + self.d))
            - torch.lgamma(0.5 * self.df)
            - 0.5 * self.d * (self.df.log() + np.log(np.pi))
            - 0.5 * logdet
            + central_term
        )
        return res.squeeze(-1)
