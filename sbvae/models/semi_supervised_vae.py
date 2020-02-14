import logging

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from sbvae.models.regular_modules import (
    BernoulliDecoderA,
    ClassifierA,
    DecoderA,
    EncoderA,
)

logger = logging.getLogger(__name__)


class SemiSupervisedVAE(nn.Module):
    r"""

    """

    def __init__(
        self,
        n_input: int,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        y_prior=None,
        classifier_parameters: dict = dict(),
        prevent_saturation: bool = False,
        iaf_t=0,
        do_batch_norm=False,
        multi_encoder_keys=["default"],
        use_classifier=False,
    ):
        # TODO: change architecture so that it match something existing
        super().__init__()

        self.n_labels = n_labels
        self.n_latent = n_latent
        # Classifier takes n_latent as input
        self.classifier = nn.ModuleDict(
            {
                key: ClassifierA(
                    n_latent,
                    n_output=n_labels,
                    do_batch_norm=do_batch_norm,
                    dropout_rate=dropout_rate,
                )
                for key in multi_encoder_keys
            }
        )

        self.encoder_z1 = nn.ModuleDict(
            {
                key: EncoderA(
                    # n_input=n_input + n_labels,
                    n_input=n_input,
                    n_output=n_latent,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    do_batch_norm=do_batch_norm,
                )
                for key in multi_encoder_keys
            }
        )

        # q(z_2 \mid z_1, c)
        self.encoder_z2_z1 = nn.ModuleDict(
            {
                key: EncoderA(
                    n_input=n_latent + n_labels,
                    n_output=n_latent,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    do_batch_norm=do_batch_norm,
                )
                for key in multi_encoder_keys
            }
        )

        self.decoder_z1_z2 = DecoderA(
            n_input=n_latent + n_labels, n_output=n_latent, n_hidden=n_hidden
        )

        self.x_decoder = BernoulliDecoderA(
            n_input=n_latent, n_output=n_input, do_batch_norm=do_batch_norm
        )

        y_prior_probs = (
            y_prior
            if y_prior is not None
            else (1 / n_labels) * torch.ones(1, n_labels, device="cuda")
        )

        self.y_prior = Categorical(probs=y_prior_probs)

        self.encoder_params = filter(
            lambda p: p.requires_grad,
            list(self.classifier.parameters())
            + list(self.encoder_z1.parameters())
            + list(self.encoder_z2_z1.parameters()),
        )

        self.decoder_params = filter(
            lambda p: p.requires_grad,
            list(self.decoder_z1_z2.parameters()) + list(self.x_decoder.parameters()),
        )

        self.all_params = filter(lambda p: p.requires_grad, list(self.parameters()))
        self.use_classifier = use_classifier

    def classify(self, x, n_samples=1, encoder_key="default"):
        # n_cat = self.n_labels
        # n_batch = len(x)
        # inp = torch.cat((x, torch.zeros(n_batch, n_cat, device=x.device)), dim=-1)

        out = self.inference(x, n_samples=n_samples, encoder_key=encoder_key)
        w_y = out["qc_z1_all_probas"].squeeze()

        # inp = x
        # q_z1 = self.encoder_z1(inp, n_samples=n_samples)
        # z = q_z1["latent"]
        # w_y = self.classifier(z)
        return w_y

    def get_latents(self, x, y=None):
        pass

    def inference_defensive_sampling(self, x, y, counts):
        n_samples_total = counts.sum()
        n_batch, _ = x.shape
        n_latent = self.n_latent
        n_labels = self.n_labels
        sum_key = "sum_supervised" if y is not None else "sum_unsupervised"

        # z sampling
        ct = counts[0]
        if ct >= 1:
            post_cubo = self.inference(x, n_samples=ct, encoder_key="CUBO")
            z_cubo = (
                post_cubo["z1"]
                .view(1, counts[0], n_batch, n_latent)
                .expand(n_labels, counts[0], n_batch, n_latent)
            )
            u_cubo = post_cubo["z2"]
            ys_cubo = post_cubo["ys"]
        else:
            z_cubo = torch.tensor([], device="cuda")
            u_cubo = torch.tensor([], device="cuda")
            ys_cubo = torch.tensor([], device="cuda")

        ct = counts[1]
        if ct >= 1:
            post_eubo = self.inference(x, n_samples=ct, encoder_key="EUBO")
            z_eubo = (
                post_eubo["z1"]
                .view(1, counts[1], n_batch, n_latent)
                .expand(n_labels, counts[1], n_batch, n_latent)
            )
            u_eubo = post_eubo["z2"]
            ys_eubo = post_eubo["ys"]
        else:
            z_eubo = torch.tensor([], device="cuda")
            u_eubo = torch.tensor([], device="cuda")
            ys_eubo = torch.tensor([], device="cuda")
        # Sampling prior
        ct = counts[2]
        if ct >= 1:
            latents_prior = self.latent_prior_sample(n_batch=n_batch, n_samples=ct)

            z1_prior = latents_prior["z1"]
            z2_prior = latents_prior["z2"]
            ys_prior = latents_prior["ys"]
        else:
            z1_prior = torch.tensor([], device="cuda")
            z2_prior = torch.tensor([], device="cuda")
            ys_prior = torch.tensor([], device="cuda")

        # Concatenating latent variables
        z_all = torch.cat([z_cubo, z_eubo, z1_prior], dim=1)
        u_all = torch.cat([u_cubo, u_eubo, z2_prior], dim=1)
        y_all = torch.cat([ys_cubo, ys_eubo, ys_prior], dim=1)

        log_p_alpha = (1.0 * counts / counts.sum()).log()
        log_p_alpha = log_p_alpha
        counts = counts
        distribs_all = ["CUBO", "EUBO", "PRIOR"]

        log_contribs = []
        classifier_contribs = []
        for count, encoder_key, log_p_a in zip(counts, distribs_all, log_p_alpha):
            if encoder_key == "PRIOR":
                res = self.latent_prior_log_proba(z1=z_all, z2=u_all, y=y_all)
                log_proba_c = res["log_pc"]
                log_proba_prior = res["sum_unsupervised"]
            else:
                res = self.variational_log_proba(
                    z1=z_all, z2=u_all, y=y_all, x=x, encoder_key=encoder_key
                )
                log_proba_c = res["log_qc_z1"]
            if count >= 1:
                log_contribs.append(res[sum_key].unsqueeze(-1) + log_p_a)
                classifier_contribs.append(log_proba_c.unsqueeze(-1) + log_p_a)

        log_contribs = torch.cat(log_contribs, dim=-1)
        sum_log_q = torch.logsumexp(log_contribs, dim=-1)
        classifier_contribs = torch.cat(classifier_contribs, dim=-1)
        log_qc_z1 = torch.logsumexp(classifier_contribs, dim=-1)
        # n_cat, n_samples, n_batch
        qc_z1 = log_qc_z1.exp()
        qc_z1_all_probas = log_qc_z1.exp().permute(1, 2, 0)
        # Decoder part
        px_z_loc = self.x_decoder(z_all)
        log_px_z = Bernoulli(px_z_loc).log_prob(x).sum(-1)

        # Log ratio contruction
        log_ratio = log_px_z + log_proba_prior - sum_log_q
        # Shape n_cat, n_samples, n_batch
        if y is not None:
            one_hot = torch.cuda.FloatTensor(n_batch, n_labels).zero_()
            mask = (
                one_hot.scatter_(1, y.view(-1, 1), 1)
                .T.view(n_labels, 1, n_batch)
                .expand(n_labels, n_samples_total, n_batch)
            )
            log_ratio = (log_ratio * mask).sum(0)
            sum_log_q = (sum_log_q * mask).sum(0)
            log_px_z = (log_px_z * mask).sum(0)
            log_qc_z1 = (log_qc_z1 * mask).sum(0)
            qc_z1 = (qc_z1 * mask).sum(0)

        return dict(
            log_px_z=log_px_z,
            sum_log_q=sum_log_q,
            log_ratio=log_ratio,
            log_qc_z1=log_qc_z1,
            qc_z1=qc_z1,
            qc_z1_all_probas=qc_z1_all_probas,
            z1=z_all,
            z2=u_all,
        )

    def latent_prior_sample(self, n_batch, n_samples):
        n_cat = self.n_labels
        n_latent = self.n_latent

        u = (
            Normal(
                torch.zeros(n_latent, device="cuda"),
                torch.ones(n_latent, device="cuda"),
            )
            .sample((n_samples, n_batch))
            .view(1, n_samples, n_batch, n_latent)
            .expand(n_cat, n_samples, n_batch, n_latent)
        )

        ys = (
            torch.eye(n_cat, device="cuda")
            .view(n_cat, 1, 1, n_cat)
            .expand(n_cat, n_samples, n_batch, n_cat)
        )
        z2_y = torch.cat([u, ys], dim=-1)
        pz1_z2m, pz1_z2_v = self.decoder_z1_z2(z2_y)
        z = Normal(pz1_z2m, pz1_z2_v).sample()
        return dict(z1=z, z2=u, ys=ys)

    def latent_prior_log_proba(self, z1, z2, y):
        log_pc = -np.log(self.n_labels)
        log_pu = Normal(torch.zeros_like(z2), torch.ones_like(z2)).log_prob(z2).sum(-1)
        z2_y = torch.cat([z2, y], dim=-1)
        pz1_z2m, pz1_z2_v = self.decoder_z1_z2(z2_y)
        log_pz = Normal(pz1_z2m, pz1_z2_v.sqrt()).log_prob(z1).sum(-1)
        return dict(
            log_pc=log_pc * torch.ones_like(log_pz),
            log_pz1_z2=log_pz,
            log_pz2=log_pu,
            sum_unsupervised=log_pc + log_pu + log_pz,
            sum_supervised=log_pu + log_pz,
        )

    def variational_log_proba(self, z1, z2, y, x, encoder_key):
        n_cat, n_samples, n_batch, n_latent = z2.shape
        # Z
        post_z = self.encoder_z1[encoder_key](x, n_samples=1)
        log_qz = Normal(post_z["q_m"], post_z["q_v"].sqrt()).log_prob(z1).sum(-1)

        # C
        if self.use_classifier:
            # z1 all the same along categorical axis in this case
            probas_c = self.classifier[encoder_key](z1[0])
            q_c = probas_c.permute(2, 0, 1)
            # n_cat, n_samples, n_latent
            log_qc = q_c.log()
        else:
            probas_c = self.classifier[encoder_key](z1)
            mask = (
                torch.eye(n_cat, device="cuda")
                .view(n_cat, 1, 1, n_cat)
                .expand(n_cat, n_samples, n_batch, n_cat)
            )
            q_c = (probas_c * mask).sum(-1)
            log_qc = q_c.log()

        # U
        z1_y = torch.cat([z1, y], dim=-1)
        post_u = self.encoder_z2_z1[encoder_key](z1_y, n_samples=1)
        log_qu = Normal(post_u["q_m"], post_u["q_v"].sqrt()).log_prob(z2).sum(-1)
        return dict(
            log_qz1_x=log_qz,
            log_qc_z1=log_qc,
            log_qz2_z1=log_qu,
            sum_supervised=log_qz + log_qu,
            sum_unsupervised=log_qz + log_qc + log_qu,
        )

    def inference(
        self, x, y=None, n_samples=1, reparam=True, encoder_key="default", counts=None
    ):
        """
        Dimension choice
            (n_categories, n_is, n_batch, n_latent)

            log_q
            (n_categories, n_is, n_batch)
        """
        if encoder_key == "defensive":
            return self.inference_defensive_sampling(x=x, y=y, counts=counts)

        n_batch = len(x)
        n_cat = self.n_labels

        if y is None:
            # deal with the case that the latent factorization is not the same in M1 and M2
            ys = (
                torch.eye(n_cat, device=x.device)
                .view(n_cat, 1, 1, n_cat)
                .expand(n_cat, n_samples, n_batch, n_cat)
            )
            # inp = torch.cat((x, torch.zeros(n_batch, n_cat, device=x.device)), dim=-1)
        else:
            ys = torch.cuda.FloatTensor(n_batch, n_cat)
            ys.zero_()
            ys.scatter_(1, y.view(-1, 1), 1)
            ys = ys.view(1, n_batch, n_cat).expand(n_samples, n_batch, n_cat)
            # inp = torch.cat((x, ys[0, :]), dim=-1)
        inp = x
        q_z1 = self.encoder_z1[encoder_key](
            inp, n_samples=n_samples, reparam=reparam, squeeze=False
        )
        # if not self.do_iaf:
        qz1_m = q_z1["q_m"]
        qz1_v = q_z1["q_v"]
        z1 = q_z1["latent"]
        assert z1.dim() == 3

        log_qz1_x = Normal(qz1_m, qz1_v.sqrt()).log_prob(z1).sum(-1)

        # Broadcast labels if necessary
        if y is None:
            n_latent = z1.shape[-1]
            z1s = z1.view(1, n_samples, n_batch, n_latent).expand(
                n_cat, n_samples, n_batch, n_latent
            )
            y_int = ys.argmax(dim=-1)

            # Dealing with qc
            qc_z1 = self.classifier[encoder_key](z1).permute(2, 0, 1)

        else:
            z1s = z1
            y_int = y

            # Dealing with qc
            qc_z1 = self.classifier[encoder_key](z1)[ys.bool()].view(n_samples, n_batch)

            # qc_z1_check = (self.classifier(z1) * ys.byte()).max(-1).values
            # assert (qc_z1_check == qc_z1).all()

        qc_z1_all_probas = self.classifier[encoder_key](z1)
        log_qc_z1 = qc_z1.log()

        log_pc = self.y_prior.log_prob(y_int)
        pc = log_pc.exp()

        z1_y = torch.cat([z1s, ys], dim=-1)
        q_z2_z1 = self.encoder_z2_z1[encoder_key](z1_y, n_samples=1, reparam=reparam)
        z2 = q_z2_z1["latent"]
        qz2_z1_m = q_z2_z1["q_m"]
        qz2_z1_v = q_z2_z1["q_v"]
        log_qz2_z1 = Normal(q_z2_z1["q_m"], q_z2_z1["q_v"].sqrt()).log_prob(z2).sum(-1)

        z2_y = torch.cat([z2, ys], dim=-1)
        pz1_z2m, pz1_z2_v = self.decoder_z1_z2(z2_y)
        log_pz1_z2 = Normal(pz1_z2m, pz1_z2_v.sqrt()).log_prob(z1).sum(-1)

        log_pz2 = Normal(torch.zeros_like(z2), torch.ones_like(z2)).log_prob(z2).sum(-1)

        px_z_loc = self.x_decoder(z1)
        log_px_z = Bernoulli(px_z_loc).log_prob(x).sum(-1)

        generative_density = log_pz2 + log_pc + log_pz1_z2 + log_px_z
        variational_density = log_qz1_x + log_qz2_z1
        log_ratio = generative_density - variational_density

        variables = dict(
            z1=z1,
            ys=ys,
            z2=z2,
            qz1_m=qz1_m,
            qz1_v=qz1_v,
            qz2_z1_m=qz2_z1_m,
            qz2_z1_v=qz2_z1_v,
            pz1_z2m=pz1_z2m,
            pz1_z2_v=pz1_z2_v,
            px_z_m=px_z_loc,
            log_qz1_x=log_qz1_x,
            qc_z1=qc_z1,
            log_qc_z1=log_qc_z1,
            log_qz2_z1=log_qz2_z1,
            log_pz2=log_pz2,
            log_pc=log_pc,
            pc=pc,
            log_pz1_z2=log_pz1_z2,
            log_px_z=log_px_z,
            generative_density=generative_density,
            variational_density=variational_density,
            log_ratio=log_ratio,
            qc_z1_all_probas=qc_z1_all_probas,
        )
        return variables

    def forward(
        self,
        x,
        loss_type="ELBO",
        y=None,
        n_samples=1,
        reparam=True,
        encoder_key="default",
        counts=torch.tensor([8, 8, 2]),
    ):
        """

        n_categories, n_is, n_batch
        """
        is_labelled = False if y is None else True

        vars = self.inference(
            x=x,
            y=y,
            n_samples=n_samples,
            reparam=reparam,
            encoder_key=encoder_key,
            counts=counts,
        )
        if encoder_key == "defensive":
            log_ratio = vars["log_ratio"]
        else:
            log_ratio = (
                vars["generative_density"] - vars["log_qz1_x"] - vars["log_qz2_z1"]
            )
            if not is_labelled:
                # Unlabelled case: c latent variable
                log_ratio -= vars["log_qc_z1"]

        everything_ok = log_ratio.ndim == 2 if is_labelled else log_ratio.ndim == 3
        assert everything_ok

        if loss_type == "ELBO":
            loss = self.elbo(log_ratio, is_labelled=is_labelled, **vars)
        elif loss_type == "CUBO":
            loss = self.cubo(log_ratio, is_labelled=is_labelled, **vars)
        elif loss_type == "CUBOB":
            loss = self.cubob(log_ratio, is_labelled=is_labelled, **vars)
        elif loss_type == "REVKL":
            loss = self.forward_kl(log_ratio, is_labelled=is_labelled, **vars)
        elif loss_type == "IWELBO":
            loss = self.iwelbo(log_ratio, is_labelled=is_labelled, **vars)
        else:
            raise ValueError("Mode {} not recognized".format(loss_type))
        if torch.isnan(loss).any() or not torch.isfinite(loss).any():
            print("NaN loss")

        return loss

    @staticmethod
    def elbo(log_ratios, is_labelled, **kwargs):
        if is_labelled:
            loss = -log_ratios.mean()
        else:
            categorical_weights = kwargs["qc_z1"]
            loss = (categorical_weights * log_ratios).sum(0)
            loss = -loss.mean()
        return loss

    @staticmethod
    def iwelbo(log_ratios, is_labelled, evaluate=False, **kwargs):
        if is_labelled:
            # (n_samples, n_batch)
            assert not evaluate
            ws = torch.softmax(log_ratios, dim=0)
            loss = -(ws.detach() * log_ratios).sum(dim=0)
        else:
            if evaluate:
                log_q_c = kwargs["log_qc_z1"]
                n_cat, n_samples, n_batch = log_ratios.shape
                res = torch.logsumexp(
                    (log_ratios + log_q_c).view(n_cat * n_samples, n_batch),
                    dim=0,
                    keepdim=False,
                )
                res = res - np.log(n_samples)
                return res
            # loss =
            categorical_weights = kwargs["qc_z1"]
            # n_cat, n_samples, n_batch
            weights = torch.softmax(log_ratios, 1).detach()
            loss = categorical_weights * weights * log_ratios
            loss = loss.mean(dim=1)  # samples
            loss = loss.sum(dim=0)  # cats
            loss = -loss.mean()
        return loss

    @staticmethod
    def forward_kl(log_ratios, is_labelled, **kwargs):
        """
        IMPORTANT: Assumes non reparameterized latent samples
        # TODO: Reparameterized version?
        """

        # TODO Triple check
        if is_labelled:
            ws = torch.softmax(log_ratios, dim=0)
            sum_log_q = kwargs["log_qz1_x"] + kwargs["log_qz2_z1"]
            rev_kl = ws.detach() * (-1) * sum_log_q
            rev_kl = rev_kl.sum(dim=0)
            return rev_kl.mean()
        else:
            log_pz1z2x_c = kwargs["log_pz1_z2"] + kwargs["log_pz2"] + kwargs["log_px_z"]
            log_qz1z2_xc = kwargs["log_qz1_x"] + kwargs["log_qz2_z1"]
            log_q = log_qz1z2_xc + kwargs["log_qc_z1"]
            # Shape (n_cat, n_is, n_batch)
            importance_weights = torch.softmax(log_pz1z2x_c - log_qz1z2_xc, dim=1)

            # Reparameterized version
            # rev_kl = (importance_weights.detach() * log_ratios).sum(dim=1)
            # Reinforce version
            rev_kl = (-1.0) * importance_weights.detach() * log_q
            rev_kl = rev_kl.sum(dim=1)

            categorical_weights = kwargs["pc"].detach()
            # print(categorical_weights.shape) ==> 3d
            # assert (categorical_weights[:, 0] == categorical_weights[:, 1]).all()
            categorical_weights = categorical_weights.mean(1)
            rev_kl = (categorical_weights * rev_kl).sum(0)
            return rev_kl.mean()

    @staticmethod
    def cubo(log_ratios, is_labelled, evaluate=False, **kwargs):
        if is_labelled:
            assert not evaluate
            ws = torch.softmax(2 * log_ratios, dim=0)  # Corresponds to squaring
            cubo_loss = ws.detach() * (-1) * log_ratios
            return cubo_loss.mean()
        else:
            # Prefer to deal this case separately to avoid mistakes
            if evaluate:
                log_q_c = kwargs["log_qc_z1"]
                n_cat, n_samples, n_batch = log_ratios.shape
                res = torch.logsumexp(
                    (2 * log_ratios + log_q_c).view(n_cat * n_samples, n_batch),
                    dim=0,
                    keepdim=False,
                )
                res = res - np.log(n_samples)
                return res

            assert log_ratios.dim() == 3
            log_qc_z1 = kwargs["log_qc_z1"]
            log_ratios += 0.5 * log_qc_z1
            ws = torch.softmax(2 * log_ratios, dim=1)
            cubo_loss = ws.detach() * (-1) * log_ratios
            cubo_loss = cubo_loss.mean(dim=1)  # samples
            cubo_loss = cubo_loss.sum(dim=0)  # cats
            return cubo_loss.mean()

    @staticmethod
    def cubob(log_ratios, is_labelled, evaluate=False, **kwargs):
        if is_labelled:
            assert not evaluate
            ws = torch.softmax(2 * log_ratios, dim=0)  # Corresponds to squaring
            cubo_loss = ws.detach() * (-1) * log_ratios
            return cubo_loss.mean()
        else:
            assert log_ratios.dim() == 3
            log_qc_z1 = kwargs["log_qc_z1"]
            # n_cat, n_samples, n_batch
            qc_z1_d = kwargs["qc_z1"].detach()
            ws_d = torch.softmax(2 * log_ratios, dim=1).detach()

            cubo_loss = -(qc_z1_d * ws_d * (log_qc_z1 + 2.0 * log_ratios))
            cubo_loss = cubo_loss.mean(dim=1)  # samples
            cubo_loss = cubo_loss.sum(dim=0)  # cats
            return cubo_loss.mean()
