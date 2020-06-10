import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import (
    Bernoulli,
    Categorical,
    Normal,
    RelaxedOneHotCategorical,
    OneHotCategorical,
)

from sbvae.models.regular_modules import (
    BernoulliDecoderA,
    ClassifierA,
    DecoderA,
    EncoderA,
    EncoderAStudent,
    EncoderB,
    EncoderBStudent,
)

logger = logging.getLogger(__name__)


class RelaxedSVAE(nn.Module):
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
        y_prior: torch.Tensor = None,
        classifier_parameters: dict = dict(),
        do_batch_norm: bool = False,
        multi_encoder_keys: list = ["default"],
        use_classifier: bool = False,
        encoder_z1: nn.Module = None,
        encoder_z2_z1: nn.Module = None,
        vdist_map=None,
    ):
        if vdist_map is None:
            vdist_map = dict(
                REVKL="gaussian",
                CUBO="gaussian",
                ELBO="gaussian",
                IWELBO="gaussian",
                IWELBOC="gaussian",
                default="gaussian",
            )
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

        if encoder_z1 is None:
            z1_map = dict(gaussian=EncoderB, student=EncoderBStudent,)
            self.encoder_z1 = nn.ModuleDict(
                {
                    # key: EncoderA(
                    # key: EncoderB(
                    key: z1_map[vdist_map[key]](
                        n_input=n_input,
                        n_output=n_latent,
                        n_hidden=n_hidden,
                        dropout_rate=dropout_rate,
                        do_batch_norm=do_batch_norm,
                    )
                    for key in multi_encoder_keys
                }
            )
        else:
            self.encoder_z1 = encoder_z1

        # q(z_2 \mid z_1, c)
        if encoder_z2_z1 is None:
            z2_map = dict(gaussian=EncoderA, student=EncoderAStudent,)
            self.encoder_z2_z1 = nn.ModuleDict(
                {
                    # key: EncoderA(
                    key: z2_map[vdist_map[key]](
                        n_input=n_latent + n_labels,
                        n_output=n_latent,
                        n_hidden=n_hidden,
                        dropout_rate=dropout_rate,
                        do_batch_norm=do_batch_norm,
                    )
                    for key in multi_encoder_keys
                }
            )
        else:
            self.encoder_z2_z1 = encoder_z2_z1

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

    def classify(
        self,
        x,
        n_samples=1,
        mode="plugin",
        counts: pd.Series = None,
        encoder_key="default",
        outs=None,
    ):
        # if outs is None:
        # Temperature not used here
        # outs = self.inference(
        #     x,
        #     n_samples=n_samples,
        #     encoder_key=encoder_key,
        #     counts=counts,
        #     temperature=0.5,
        # )
        n_batch, _ = x.shape
        inference_kwargs = dict(
            n_samples=n_samples,
            encoder_key=encoder_key,
            counts=counts,
            temperature=0.5,
        )
        if mode == "plugin":
            outs = self.inference(x, **inference_kwargs)
            w_y = outs["qc_z1_all_probas"].mean(0)
        # if mode == "plugin":
        # w_y = outs["log_qc_z1"].exp().mean(1).transpose(-1, -2)
        elif mode == "is":
            pc_x = torch.zeros(n_batch, self.n_labels, device=x.device)
            for c_val in np.arange(self.n_labels):
                y_val = c_val * torch.ones(n_batch, device="cuda")
                y_val = y_val.long()
                outs = self.inference(x, y=y_val, **inference_kwargs)
                if counts is None:
                    log_generative = (
                        outs["log_pc"]
                        + outs["log_pz2"]
                        + outs["log_pz1_z2"]
                        + outs["log_px_z"]
                    )
                    log_predictive = (
                        outs["log_qz1_x"] + outs["log_qz2_z1"]  # + outs["log_qc_z1"]
                    )
                    log_w = log_generative - log_predictive
                else:
                    # log_qc_z = outs["log_qc_z1"]
                    log_w = outs["log_ratio"]
                    # log_w = log_w - log_qc_z
                # n_samples, n_batch
                log_p_c_i = torch.logsumexp(log_w, dim=0)
                pc_x[:, c_val] = log_p_c_i.exp()
            pc_x = pc_x / pc_x.sum(-1, keepdim=True)
            w_y = pc_x
            #     log_wtilde = nn.LogSoftmax(0)(log_w)
            #     log_pc1_x = log_qc_z + log_wtilde
            #     log_pc1_x = torch.logsumexp(log_pc1_x, 0)
            #     all_log_pc_x.append(log_pc1_x.unsqueeze(-1))
            # all_log_pc_x = torch.cat(all_log_pc_x, -1)
            # w_y = nn.Softmax(dim=-1)(all_log_pc_x)

        else:
            raise ValueError("Mode {} not recognize".format(mode))
        # inp = x
        # q_z1 = self.encoder_z1(inp, n_samples=n_samples)
        # z = q_z1["latent"]
        # w_y = self.classifier(z)
        return w_y

    def update_q(
        self, classifier: nn.Module, encoder_z1: nn.Module, encoder_z2_z1: nn.Module,
    ):
        logging.info("UPDATING ENCODERS ... ONLY MAKES SENSE WHEN USING EVAL ENCODER!")
        self.classifier = classifier
        self.encoder_z1 = encoder_z1
        self.encoder_z2_z1 = encoder_z2_z1

    def get_latents(self, x, y=None):
        pass

    def latent_prior_sample(self, y, n_batch, n_samples):
        n_cat = self.n_labels
        n_latent = self.n_latent

        u = Normal(
            torch.zeros(n_latent, device="cuda"), torch.ones(n_latent, device="cuda"),
        ).sample((n_samples, n_batch))

        if y is None:
            ys = OneHotCategorical(
                probs=(1.0 / n_cat) * torch.ones(n_cat, device="cuda")
            ).sample((n_samples, n_batch))
        else:
            ys = torch.cuda.FloatTensor(n_batch, n_cat)
            ys.zero_()
            ys.scatter_(1, y.view(-1, 1), 1)
            ys = ys.view(1, n_batch, n_cat).expand(n_samples, n_batch, n_cat)

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
        n_samples, n_batch, n_latent = z2.shape
        # Z
        post_z = self.encoder_z1[encoder_key](x, n_samples=1)
        z_dist = post_z["dist"]
        log_qz = z_dist.log_prob(z1)
        if post_z["sum_last"]:
            log_qz = log_qz.sum(-1)

        qc_z1 = self.classifier[encoder_key](z1)
        log_qc_z1 = qc_z1.log()
        y_int = y.argmax(-1)
        log_qc = torch.gather(log_qc_z1, dim=-1, index=y_int.unsqueeze(-1)).squeeze(-1)
        # U
        z1_y = torch.cat([z1, y], dim=-1)
        post_u = self.encoder_z2_z1[encoder_key](z1_y, n_samples=1)
        u_dist = post_u["dist"]
        log_qu = u_dist.log_prob(z2)
        if post_u["sum_last"]:
            log_qu = log_qu.sum(-1)

        assert log_qz.shape == log_qc.shape == log_qu.shape, (
            log_qz.shape,
            log_qc.shape,
            log_qu.shape,
        )
        return dict(
            log_qz1_x=log_qz,
            qc_all_probas=qc_z1,
            log_qc_z1=log_qc,
            log_qz2_z1=log_qu,
            sum_supervised=log_qz + log_qu,
            sum_unsupervised=log_qz + log_qc + log_qu,
        )

    def inference(
        self,
        x,
        y=None,
        temperature=None,
        n_samples=1,
        reparam=True,
        encoder_key="default",
        counts=None,
    ):
        """
        Dimension choice
            (n_categories, n_is, n_batch, n_latent)

            log_q
            (n_categories, n_is, n_batch)
        """
        if temperature is None:
            raise ValueError(
                "Please provide a temperature for the relaxed OneHot distribution"
            )

        if counts is not None:
            return self.inference_defensive_sampling(
                x=x, y=y, temperature=temperature, counts=counts
            )
        n_cat = self.n_labels
        n_batch = len(x)
        # Z | X
        inp = x
        q_z1 = self.encoder_z1[encoder_key](
            inp, n_samples=n_samples, reparam=reparam, squeeze=False
        )
        # if not self.do_iaf:
        qz1_m = q_z1["q_m"]
        qz1_v = q_z1["q_v"]
        z1 = q_z1["latent"]
        assert z1.dim() == 3
        # log_qz1_x = Normal(qz1_m, qz1_v.sqrt()).log_prob(z1).sum(-1)
        log_qz1_x = q_z1["dist"].log_prob(z1)
        dfs = q_z1.get("df", None)
        if q_z1["sum_last"]:
            log_qz1_x = log_qz1_x.sum(-1)
        z1s = z1
        # torch.cuda.synchronize()

        #  C | Z
        # Broadcast labels if necessary
        qc_z1 = self.classifier[encoder_key](z1)
        log_qc_z1 = qc_z1.log()
        qc_z1_all_probas = qc_z1
        # C
        if y is None:
            if reparam:
                cat_dist = RelaxedOneHotCategorical(
                    temperature=temperature, probs=qc_z1
                )
                ys_probs = cat_dist.rsample()
            else:
                cat_dist = OneHotCategorical(probs=qc_z1)
                ys_probs = cat_dist.sample()
            ys = (ys_probs == ys_probs.max(-1, keepdim=True).values).float()
            y_int = ys.argmax(-1)
        else:
            ys = torch.cuda.FloatTensor(n_batch, n_cat)
            ys.zero_()
            ys.scatter_(1, y.view(-1, 1), 1)
            ys = ys.view(1, n_batch, n_cat).expand(n_samples, n_batch, n_cat)
            y_int = y.view(1, -1).expand(n_samples, n_batch)
        log_pc = self.y_prior.log_prob(y_int)
        assert y_int.unsqueeze(-1).shape == (n_samples, n_batch, 1), y_int.shape
        log_qc_z1 = torch.gather(log_qc_z1, dim=-1, index=y_int.unsqueeze(-1)).squeeze(
            -1
        )
        qc_z1 = torch.gather(qc_z1, dim=-1, index=y_int.unsqueeze(-1)).squeeze(-1)
        assert qc_z1.shape == (n_samples, n_batch)
        pc = log_pc.exp()

        # U | Z1, C
        z1_y = torch.cat([z1s, ys], dim=-1)
        q_z2_z1 = self.encoder_z2_z1[encoder_key](z1_y, n_samples=1, reparam=reparam)
        z2 = q_z2_z1["latent"]
        qz2_z1_m = q_z2_z1["q_m"]
        qz2_z1_v = q_z2_z1["q_v"]
        # log_qz2_z1 = Normal(q_z2_z1["q_m"], q_z2_z1["q_v"].sqrt()).log_prob(z2).sum(-1)
        log_qz2_z1 = q_z2_z1["dist"].log_prob(z2)
        if q_z2_z1["sum_last"]:
            log_qz2_z1 = log_qz2_z1.sum(-1)
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
            df=dfs,
        )
        # torch.cuda.synchronize()
        return variables

    def inference_defensive_sampling(self, x, y, temperature, counts: pd.Series):
        n_samples_total = counts.sum()
        n_batch, _ = x.shape
        n_latent = self.n_latent
        n_labels = self.n_labels
        sum_key = "sum_supervised" if y is not None else "sum_unsupervised"

        # z sampling
        encoder_keys = counts.keys()
        z_all = []
        u_all = []
        y_all = []
        for key in encoder_keys:
            ct = counts[key]
            if key == "prior":
                _post = self.latent_prior_sample(y, n_batch=n_batch, n_samples=ct)
            else:
                _post = self.inference(
                    x,
                    y,
                    temperature=temperature,
                    n_samples=ct,
                    encoder_key=key,
                    reparam=False,
                )
            z_all.append(_post["z1"])
            u_all.append(_post["z2"])
            y_all.append(_post["ys"])
        z_all = torch.cat(z_all, dim=0)
        u_all = torch.cat(u_all, dim=0)
        y_all = torch.cat(y_all, dim=0)

        p_alpha = 1.0 * counts / counts.sum()
        log_p_alpha = p_alpha.apply(np.log)

        log_contribs = []
        classifier_contribs = []
        log_qc_all_contribs = []
        res_prior = self.latent_prior_log_proba(z1=z_all, z2=u_all, y=y_all)
        log_proba_c_prior = res_prior["log_pc"]
        log_proba_prior = res_prior["sum_unsupervised"]
        for encoder_key in encoder_keys:
            count = counts[encoder_key]
            log_p_a = log_p_alpha[encoder_key]
            if encoder_key == "prior":
                log_proba_c = log_proba_c_prior
                log_qc_all = (1.0 / n_labels) * torch.ones(
                    n_samples_total, n_batch, n_labels, device="cuda"
                )
                res = res_prior
            else:
                res = self.variational_log_proba(
                    z1=z_all, z2=u_all, y=y_all, x=x, encoder_key=encoder_key
                )
                log_qc_all = res["qc_all_probas"].log()
                log_proba_c = res["log_qc_z1"]
            log_contribs.append(res[sum_key].unsqueeze(-1) + log_p_a)
            log_qc_all_contribs.append(log_qc_all.unsqueeze(-1) + log_p_a)
            classifier_contribs.append(log_proba_c.unsqueeze(-1) + log_p_a)

        log_contribs = torch.cat(log_contribs, dim=-1)
        sum_log_q = torch.logsumexp(log_contribs, dim=-1)
        classifier_contribs = torch.cat(classifier_contribs, dim=-1)
        log_qc_z1 = torch.logsumexp(classifier_contribs, dim=-1)

        log_qc_all_contribs = torch.cat(log_qc_all_contribs, dim=-1)
        log_qc_all_contribs = torch.logsumexp(log_qc_all_contribs, dim=-1)
        qc_all = log_qc_all_contribs.exp()

        # n_cat, n_samples, n_batch
        qc_z1 = log_qc_z1.exp()
        # qc_z1_all_probas = log_qc_z1.exp().permute(1, 2, 0)
        # Decoder part
        px_z_loc = self.x_decoder(z_all)
        log_px_z = Bernoulli(px_z_loc).log_prob(x).sum(-1)

        # Log ratio contruction
        log_ratio = log_px_z + log_proba_prior - sum_log_q
        # Shape n_cat, n_samples, n_batch

        return dict(
            log_px_z=log_px_z,
            sum_log_q=sum_log_q,
            log_ratio=log_ratio,
            log_qc_z1=log_qc_z1,
            qc_z1=qc_z1,
            qc_z1_all_probas=qc_all,
            log_pc=res_prior["log_pc"],
            log_pz1_z2=res_prior["log_pz1_z2"],
            log_pz2=res_prior["log_pz2"],
            z1=z_all,
            z2=u_all,
        )

    def forward(
        self,
        x,
        temperature=0.5,
        loss_type="ELBO",
        y=None,
        n_samples=1,
        reparam=True,
        encoder_key="default",
        counts=torch.tensor([8, 8, 2]),
        return_outs: bool = False,
    ):
        """

        n_categories, n_is, n_batch
        """
        is_labelled = False if y is None else True

        vars = self.inference(
            x=x,
            temperature=temperature,
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
                log_ratio -= vars["log_qc_z1"]

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
            diagnostic = {
                "z1": (vars["z1"].min().item(), vars["z1"].max().item()),
                "z2": (vars["z2"].min().item(), vars["z2"].max().item()),
                "qz1_m": (vars["qz1_m"].min().item(), vars["qz1_m"].max().item()),
                "qz1_v": (vars["qz1_v"].min().item(), vars["qz1_v"].max().item()),
                "qz2_z1_m": (
                    vars["qz2_z1_m"].min().item(),
                    vars["qz2_z1_m"].max().item(),
                ),
                "qz2_z1_v": (
                    vars["qz2_z1_v"].min().item(),
                    vars["qz2_z1_v"].max().item(),
                ),
                "pz1_z2m": (vars["pz1_z2m"].min().item(), vars["pz1_z2m"].max().item()),
                "pz1_z2_v": (
                    vars["pz1_z2_v"].min().item(),
                    vars["pz1_z2_v"].max().item(),
                ),
                "px_z_m": (vars["px_z_m"].min().item(), vars["px_z_m"].max().item()),
                "log_qz1_x": (
                    vars["log_qz1_x"].min().item(),
                    vars["log_qz1_x"].max().item(),
                ),
                "qc_z1": (vars["qc_z1"].min().item(), vars["qc_z1"].max().item()),
                "log_qc_z1": (
                    vars["log_qc_z1"].min().item(),
                    vars["log_qc_z1"].max().item(),
                ),
                "log_qz2_z1": (
                    vars["log_qz2_z1"].min().item(),
                    vars["log_qz2_z1"].max().item(),
                ),
                "log_pz2": (vars["log_pz2"].min().item(), vars["log_pz2"].max().item()),
                "log_pc": (vars["log_pc"].min().item(), vars["log_pc"].max().item()),
                "log_pz1_z2": (
                    vars["log_pz1_z2"].min().item(),
                    vars["log_pz1_z2"].max().item(),
                ),
                "log_px_z": (
                    vars["log_px_z"].min().item(),
                    vars["log_px_z"].max().item(),
                ),
                "log_ratio": (
                    vars["log_ratio"].min().item(),
                    vars["log_ratio"].max().item(),
                ),
            }
            if vars["df"] is not None:
                diagnostic["df"] = (
                    vars["df"].min().item(),
                    vars["df"].max().item(),
                )
            raise ValueError(diagnostic)
        if return_outs:
            return loss, vars
        return loss

    @staticmethod
    def elbo(log_ratios, is_labelled, **kwargs):
        loss = -log_ratios.mean()
        return loss

    @staticmethod
    def iwelbo(log_ratios, is_labelled, evaluate=False, **kwargs):
        """ 
            log_ratios: if is_labelled:
                log p(x \mid z_i) p(z_i \mid u_i, c) p(u_i) p(c) 
                - log q(z_i \mid x) q(u_i \mid z_i, c)
                ==> shape (K, N)
            otherwise (unsupervised case)
                log p(x \mid z_i) p(z_i \mid u_i, c) p(u_i) p(c) 
                - log q(z_i \mid x) q(u_i \mid z_i, c) q(c \mid z_i)
                ==> shape (C, K, N)
        """
        # (n_samples, n_batch)
        assert not evaluate
        ws = torch.softmax(log_ratios, dim=0)
        loss = -(ws.detach() * log_ratios).sum(dim=0)
        return loss

    @staticmethod
    def forward_kl(log_ratios, is_labelled, **kwargs):
        """
        IMPORTANT: Assumes non reparameterized latent samples
        # TODO: Reparameterized version?
        """

        # TODO Triple check
        ws = torch.softmax(log_ratios, dim=0)
        if is_labelled:
            sum_log_q = kwargs["log_qz1_x"] + kwargs["log_qz2_z1"]
        else:
            sum_log_q = kwargs["log_qz1_x"] + kwargs["log_qz2_z1"] + kwargs["log_qc_z1"]
        rev_kl = ws.detach() * (-1) * sum_log_q
        return rev_kl.sum(dim=0)

    @staticmethod
    def cubo(log_ratios, is_labelled, evaluate=False, **kwargs):
        # if is_labelled:
        # assert not evaluate
        ws = torch.softmax(2 * log_ratios, dim=0)  # Corresponds to squaring
        cubo_loss = ws.detach() * (-1) * log_ratios
        return cubo_loss.mean()
