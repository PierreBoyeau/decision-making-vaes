import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import logging
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
import torch.distributions as db
from torch import nn
from arviz.stats import psislw

from sbvae.ais import ais_trajectory
from sbvae.dataset import GeneExpressionDataset


NUMS = 5
N_PICKS = 30
N_CELLS = 5
SEEDS = np.arange(500)
# option 3
n_genes = 100
DO_POISSON = True

import torch.distributions as distributions


class SignedGamma:
    def __init__(self, dim, proba_pos=0.75, shape=2, rate=4):
        self.proba_pos = proba_pos
        self.shape = shape
        self.rate = rate
        self.dim = dim

    def sample(self, size):
        if type(size) == int:
            sample_size = (size, self.dim)
        else:
            sample_size = list(size) + [self.dim]
        signs = 2.0 * distributions.Bernoulli(probs=0.75).sample(sample_size) - 1.0
        gammas = distributions.Gamma(concentration=self.shape, rate=self.rate).sample(
            sample_size
        )
        return signs * gammas


torch.manual_seed(42)
np.random.seed(42)

means = 10 + 50 * torch.rand(n_genes)
means = means.numpy()

means[means >= 1000] = 1000

lfc_sampler = SignedGamma(dim=2, proba_pos=0.5)
lfcs = lfc_sampler.sample(n_genes).numpy()
non_de_genes = np.random.choice(n_genes, size=300)
lfcs[non_de_genes, :] = 0.0

lfcs = torch.zeros(n_genes, 2)
non_de_genes = torch.rand(n_genes) <= 0.5
lfcs[non_de_genes, 1] = 0.16 * torch.randn_like(lfcs[non_de_genes, 1])
sign = 2.0 * (torch.rand_like(lfcs[~non_de_genes, 1]) > 0.5) - 1.0
lfcs[~non_de_genes, 1] = sign + 0.16 * torch.randn_like(lfcs[~non_de_genes, 1])

# Constructing sigma and mus
log2_mu0 = lfcs[:, 0] + np.log2(means)
log2_mu1 = lfcs[:, 1] + np.log2(means)

loge_mu0 = log2_mu0 / np.log2(np.e)
loge_mu1 = log2_mu1 / np.log2(np.e)

a = (2.0 * np.random.random(size=(n_genes, 1)) - 1).astype(float)
sigma = 2.0 * a.dot(a.T) + 0.5 * (
    1.0 + 0.5 * (2.0 * np.random.random(n_genes) - 1.0)
) * np.eye(n_genes)

sigma0 = 0.08 * sigma
sigma1 = sigma0
N_CELLS = 1000

# Poisson rates
h0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor(loge_mu0).float(), covariance_matrix=torch.tensor(sigma0).float()
).sample((N_CELLS // 2,))
h1 = torch.distributions.MultivariateNormal(
    loc=torch.tensor(loge_mu1).float(), covariance_matrix=torch.tensor(sigma1).float()
).sample((N_CELLS // 2,))
h = torch.cat([h0, h1])

# Data sampling
x_obs = torch.distributions.Poisson(rate=h.exp()).sample()
# Zero inflation
is_zi = torch.rand_like(x_obs) <= 0.3
# print("Added zeros: ", is_zi.mean())
x_obs = x_obs * (1.0 - is_zi.double())
labels = torch.zeros((N_CELLS, 1))
labels[N_CELLS // 2 :] = 1

not_null_cell = x_obs.sum(1) != 0
x_obs = x_obs[not_null_cell]
labels = labels[not_null_cell]
h0_bis, h1_bis = h0, h1
lfc_orig = h0_bis.exp().log2() - h1_bis.exp().log2()
lfc_gt = lfc_orig.mean(0)
lfc_gt = lfc_gt.numpy()
IS_SIGNIFICANT_DE = (np.abs(lfc_orig) >= 0.5).numpy().mean(0) >= 0.5
DATASET = GeneExpressionDataset(
    *GeneExpressionDataset.get_attributes_from_matrix(
        X=x_obs.numpy(), labels=labels.numpy().squeeze(),
    )
)

SAMPLE_IDX = np.random.choice(N_CELLS, 64)
Y = labels
X_U = torch.tensor(DATASET.X[SAMPLE_IDX]).to("cuda")
LOCAL_L_MEAN = torch.tensor(DATASET.local_vars[SAMPLE_IDX]).to("cuda")
LOCAL_L_VAR = torch.tensor(DATASET.local_means[SAMPLE_IDX]).to("cuda")
N_GENES = n_genes


def prauc(y, pred):
    prec, rec, thres = metrics.precision_recall_curve(y_true=y, probas_pred=pred)
    return metrics.auc(rec, prec)


def get_predictions(
    post,
    samp_a,
    samp_b,
    encoder_key,
    counts,
    n_post_samples=50,
    importance_sampling=True,
    encoder=None,
    do_observed_library=True,
):
    softmax = nn.Softmax(dim=0)

    my_multicounts = None
    if encoder_key == "defensive":
        factor_counts = n_post_samples // counts.sum()
        n_post_samples = factor_counts * counts.sum()
        my_multicounts = factor_counts * counts
    post_vals = post.getter(
        keys=["px_scale", "log_ratio"],
        n_samples=n_post_samples,
        do_observed_library=do_observed_library,
        encoder_key=encoder_key,
        counts=my_multicounts,
        z_encoder=encoder,
    )

    if importance_sampling:
        n_post, n_cells = post_vals["log_ratio"].shape
        # n_post, n_cells
        w_a = post_vals["log_ratio"][:, samp_a]
        h_a = post_vals["px_scale"][:, samp_a]
        n_post_effective = h_a.shape[0]
        w_a = softmax(w_a)
        w_ab = w_a.T
        dist_a = db.Categorical(probs=w_ab)
        iw_select_a = dist_a.sample(sample_shape=(400,))
        iw_select_a = iw_select_a.unsqueeze(1)  # index for original z shape
        iw_select_a = iw_select_a.unsqueeze(-1)  # index for number of genes
        iw_select_a = iw_select_a.expand((400, 1, len(samp_a), N_GENES))
        h_a_orig = h_a.unsqueeze(0).expand(
            (400, n_post_effective, len(samp_a), N_GENES)
        )
        h_a_final = torch.gather(h_a_orig, dim=1, index=iw_select_a).squeeze(1)

        w_b = post_vals["log_ratio"][:, samp_b]
        h_b = post_vals["px_scale"][:, samp_b]
        n_post_effective = h_b.shape[0]
        w_b = softmax(w_b)
        w_bb = w_b.T
        dist_b = db.Categorical(probs=w_bb)
        iw_select_b = dist_b.sample(sample_shape=(400,))
        iw_select_b = iw_select_b.unsqueeze(1)  # index for original z shape
        iw_select_b = iw_select_b.unsqueeze(-1)  # index for number of genes
        iw_select_b = iw_select_b.expand((400, 1, len(samp_b), N_GENES))
        h_b_orig = h_b.unsqueeze(0).expand(
            (400, n_post_effective, len(samp_b), N_GENES)
        )
        h_b_final = torch.gather(h_b_orig, dim=1, index=iw_select_b).squeeze(1)

        cells_a = np.random.choice(
            np.arange(h_a_final.shape[1]), size=2 * h_a_final.shape[1]
        )
        cells_b = np.random.choice(
            np.arange(h_b_final.shape[1]), size=2 * h_a_final.shape[1]
        )

        h_a_final = h_a_final[:, cells_a].log2()
        h_b_final = h_b_final[:, cells_b].log2()

        h_a_samp = torch.median(h_a_final, 1).values
        h_b_samp = torch.median(h_b_final, 1).values

        y_pred = h_a_samp - h_b_samp
        y_pred = (y_pred.abs() >= 0.5).float()
        y_pred = y_pred.mean(0)
        return y_pred
    else:
        #  n_samples_posterior, n_cells, n_genes
        h_a_samp = post_vals["px_scale"][:, samp_a].log2()
        h_b_samp = post_vals["px_scale"][:, samp_b].log2()

        h_a_samp = torch.median(h_a_samp, 1).values
        h_b_samp = torch.median(h_b_samp, 1).values

        y_pred = h_a_samp - h_b_samp
        y_pred = (y_pred.abs() >= 0.5).float()
        y_pred = y_pred.mean(0)
    return y_pred


def get_predictions_ais(post_a, post_b, model, schedule, n_latent, n_post_samples=200):
    """Predicts DEG using Annealing Importance Sampling

    """
    z_a, log_wa = ais_trajectory(
        model=model,
        schedule=schedule,
        loader=post_a,
        n_sample=n_post_samples,
        n_latent=n_latent,
    )
    z_b, log_wb = ais_trajectory(
        model=model,
        schedule=schedule,
        loader=post_b,
        n_sample=n_post_samples,
        n_latent=n_latent,
    )
    # Shapes n_samples, n_batch, n_latent
    # and n_samples, n_batch
    softmax = nn.Softmax(dim=0)

    n_batch = z_a.shape[1]
    with torch.no_grad():
        w_a = softmax(log_wa)
        log_h_a = (
            model.decoder(
                model.dispersion,
                z_a.cuda(),
                torch.ones(len(z_a), n_batch, 1, device="cuda"),
                None,
            )[0]
            .log2()
            .cpu()
            .view(n_post_samples, n_batch, -1)
        )
        w_ab = w_a.T
        n_post_effective = log_h_a.shape[0]
        dist_a = db.Categorical(probs=w_ab)
        iw_select_a = dist_a.sample(sample_shape=(400,))
        iw_select_a = iw_select_a.unsqueeze(1)  # index for original z shape
        iw_select_a = iw_select_a.unsqueeze(-1)  # index for number of genes
        iw_select_a = iw_select_a.expand((400, 1, n_batch, N_GENES))
        log_h_a_orig = log_h_a.unsqueeze(0).expand(
            (400, n_post_effective, n_batch, N_GENES)
        )
        log_h_a_final = torch.gather(log_h_a_orig, dim=1, index=iw_select_a).squeeze(1)

        w_b = softmax(log_wb)
        log_h_b = (
            model.decoder(
                model.dispersion,
                z_b.cuda(),
                torch.ones(len(z_b), n_batch, 1, device="cuda"),
                None,
            )[0]
            .log2()
            .cpu()
            .view(n_post_samples, n_batch, -1)
        )
        # torch.Size([200, 1, 500, 100]) torch.Size([200, 500])
        print(log_h_a.shape, log_wa.shape)
        # original n_samples, n_batch
        w_bb = w_b.T
        n_post_effective = log_h_b.shape[0]
        dist_b = db.Categorical(probs=w_bb)
        iw_select_b = dist_b.sample(sample_shape=(400,))
        iw_select_b = iw_select_b.unsqueeze(1)  # index for original z shape
        iw_select_b = iw_select_b.unsqueeze(-1)  # index for number of genes
        iw_select_b = iw_select_b.expand((400, 1, n_batch, N_GENES))
        log_h_b_orig = log_h_b.unsqueeze(0).expand(
            (400, n_post_effective, n_batch, N_GENES)
        )
        log_h_b_final = torch.gather(log_h_b_orig, dim=1, index=iw_select_b).squeeze(1)

        h_a_samp = torch.median(log_h_a_final, 1).values
        h_b_samp = torch.median(log_h_b_final, 1).values

        y_pred = h_a_samp - h_b_samp
        y_pred = (y_pred.abs() >= 0.5).float()
        y_pred = y_pred.mean(0)
    y_pred = y_pred.numpy()
    return y_pred


def fdr_score(y_true, y_pred):
    return 1.0 - precision_score(y_true, y_pred)


def tpr_score(y_true, y_pred):
    return recall_score(y_true, y_pred)


def true_fdr(y_true, y_pred):
    """
        Computes GT FDR
    """
    n_genes = len(y_true)
    probas_sorted = np.argsort(-y_pred)
    denom = np.arange(n_genes) + 1
    num = np.cumsum(y_true[probas_sorted])
    res = (denom - num) / denom
    return res


def posterior_expected_fdr(y_pred, fdr_target=0.05) -> tuple:
    """
        Computes posterior expected FDR
    """
    sorted_genes = np.argsort(-y_pred)
    sorted_pgs = y_pred[sorted_genes]
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))

    n_positive_genes = (cumulative_fdr <= fdr_target).sum() - 1
    pred_de_genes = sorted_genes[:n_positive_genes]
    is_pred_de = np.zeros_like(cumulative_fdr).astype(bool)
    is_pred_de[pred_de_genes] = True
    return cumulative_fdr, is_pred_de


def model_ais_evaluation_loop(
    trainer, n_latent, schedule_n=20, n_picks=N_PICKS, n_cells=N_CELLS,
):
    schedule = np.linspace(0, 1, schedule_n)
    mdl = trainer.model
    train_post = trainer.train_set.sequential(batch_size=128)
    test_post = trainer.test_set.sequential(batch_size=128)
    _, logwa = ais_trajectory(
        model=mdl, schedule=schedule, loader=test_post, n_sample=200, n_latent=n_latent,
    )
    iwelbo5000 = (torch.logsumexp(logwa, dim=0) - np.log(logwa.shape[0])).mean()

    logging.info("FDR/TPR ...")
    train_post = trainer.train_set.sequential()
    train_indices = train_post.indices
    y_train = Y[train_indices]

    decision_rule_fdr10 = np.zeros(n_picks)
    decision_rule_fdr05 = np.zeros(n_picks)
    decision_rule_fdr20 = np.zeros(n_picks)
    decision_rule_tpr10 = np.zeros(n_picks)
    fdr_gt = np.zeros((N_GENES, n_picks))
    pe_fdr = np.zeros((N_GENES, n_picks))
    fdr_gt_plugin = np.zeros((N_GENES, n_picks))
    y_preds_is = np.zeros((N_GENES, n_picks))
    y_gt = np.zeros((N_GENES, n_picks))

    np.random.seed(42)
    for ipick in range(n_picks):
        print(np.unique(y_train))
        if DO_POISSON:
            where_a = np.where(y_train == 0)[0]
            where_b = np.where(y_train == 1)[0]
        else:
            where_a = np.where(y_train == 1)[0]
            where_b = np.where(y_train == 2)[0]

        samples_a = np.random.choice(where_a, size=n_cells)
        samples_b = np.random.choice(where_b, size=n_cells)
        # Option 2
        samples_a_overall = train_indices[samples_a]
        samples_b_overall = train_indices[samples_b]

        h_a = h[samples_a_overall]
        h_b = h[samples_b_overall]
        lfc_loc = h_a - h_b
        is_significant_de_local = (lfc_loc.abs() >= 0.5).float().mean(0) >= 0.5
        is_significant_de_local = is_significant_de_local.numpy()

        logging.info("IS flavor ...")
        schedule = np.linspace(0.0, 1.0, 100)
        post_a = trainer.create_posterior(
            model=mdl, gene_dataset=DATASET, indices=samples_a_overall
        ).sequential(batch_size=32)

        post_b = trainer.create_posterior(
            model=mdl, gene_dataset=DATASET, indices=samples_b_overall
        ).sequential(batch_size=32)
        y_pred_ais = get_predictions_ais(
            post_a, post_b, mdl, schedule, n_latent=n_latent, n_post_samples=200,
        )
        y_pred_ais = y_pred_ais

        true_fdr_arr = true_fdr(y_true=is_significant_de_local, y_pred=y_pred_ais)
        pe_fdr_arr, y_decision_rule = posterior_expected_fdr(y_pred=y_pred_ais)
        # Fdr related
        fdr_gt[:, ipick] = true_fdr_arr
        pe_fdr[:, ipick] = pe_fdr_arr

        _, y_decision_rule10 = posterior_expected_fdr(y_pred=y_pred_ais, fdr_target=0.1)
        decision_rule_fdr10[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )
        decision_rule_tpr10[ipick] = tpr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )

        _, decision_rule_fdr05 = posterior_expected_fdr(
            y_pred=y_pred_ais, fdr_target=0.05
        )
        decision_rule_fdr05[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )
        _, decision_rule_fdr20 = posterior_expected_fdr(
            y_pred=y_pred_ais, fdr_target=0.2
        )
        decision_rule_fdr20[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )

        y_preds_is[:, ipick] = y_pred_ais
        y_gt[:, ipick] = is_significant_de_local

    prauc_is = np.array(
        [prauc(y=y_it, pred=y_pred) for (y_it, y_pred) in zip(y_gt.T, y_preds_is.T)]
    )
    all_fdr_gt = np.array(fdr_gt)
    all_pe_fdr = np.array(pe_fdr)
    fdr_gt_plugin = np.array(fdr_gt_plugin)
    fdr_diff = all_fdr_gt - all_pe_fdr
    return dict(
        iwelbo5000=np.array(iwelbo5000),
        iwelbo5000_train=None,
        pe_fdr_plugin=None,
        fdr_gt_plugin=fdr_gt_plugin,
        all_fdr_gt=all_fdr_gt,
        all_pe_fdr=all_pe_fdr,
        l1_fdr=np.linalg.norm(fdr_diff, axis=0, ord=1),
        l2_fdr=np.linalg.norm(fdr_diff, axis=0, ord=2),
        y_gt=y_gt,
        y_pred_is=y_preds_is,
        y_pred_plugin=None,
        prauc_plugin=None,
        prauc_is=prauc_is,
        fdr_controlled_fdr10=np.array(decision_rule_fdr10),
        fdr_controlled_fdr05=np.array(decision_rule_fdr05),
        fdr_controlled_fdr20=np.array(decision_rule_fdr20),
        fdr_controlled_tpr10=np.array(decision_rule_tpr10),
        fdr_controlled_fdr10_plugin=None,
        fdr_controlled_tpr10_plugin=None,
        khat_10000=None,
        lfc_gt=lfc_loc,
        ess=None,
    )


def model_evaluation_loop(
    trainer,
    eval_encoder,
    counts_eval,
    encoder_eval_name,
    n_iwsamples=5000,
    n_picks=N_PICKS,
    n_cells=N_CELLS,
    do_observed_library=True,
    n_samples_queries=200,
):
    test_post = trainer.test_set.sequential()
    mdl = trainer.model
    train_post = trainer.train_set.sequential()

    # *** IWELBO 5000
    logging.info("IWELBO 5K estimation...")
    multicounts_eval = None
    if counts_eval is not None:
        multicounts_eval = (n_iwsamples / counts_eval.sum()) * counts_eval
        multicounts_eval = multicounts_eval.astype(int)
        print(multicounts_eval)
    iwelbo5000_loss = (
        test_post.getter(
            keys=["IWELBO"],
            n_samples=n_iwsamples,
            batch_size=64,
            do_observed_library=do_observed_library,
            encoder_key=encoder_eval_name,
            counts=multicounts_eval,
            z_encoder=eval_encoder,
        )["IWELBO"]
        .cpu()
        .numpy()
    ).mean()

    iwelbo5000train_loss = (
        train_post.getter(
            keys=["IWELBO"],
            n_samples=n_iwsamples,
            batch_size=64,
            do_observed_library=do_observed_library,
            encoder_key=encoder_eval_name,
            counts=multicounts_eval,
            z_encoder=eval_encoder,
        )["IWELBO"]
        .cpu()
        .numpy()
    ).mean()

    # *** KHAT
    multicounts_eval = None
    if counts_eval is not None:
        multicounts_eval = (n_iwsamples / counts_eval.sum()) * counts_eval
        multicounts_eval = multicounts_eval.astype(int)
    log_ratios = []
    n_samples_total = 1e4
    n_samples_per_pass = (
        300 if encoder_eval_name == "default" else multicounts_eval.sum()
    )
    n_iter = int(n_samples_total / n_samples_per_pass)
    logging.info("Multicounts: {}".format(multicounts_eval))
    logging.info(
        "Khat computation using {} samples".format(n_samples_per_pass * n_iter)
    )
    for _ in tqdm(range(n_iter)):
        with torch.no_grad():
            out = mdl(
                X_U,
                LOCAL_L_MEAN,
                LOCAL_L_VAR,
                loss_type=None,
                n_samples=n_samples_per_pass,
                reparam=False,
                encoder_key=encoder_eval_name,
                counts=multicounts_eval,
                do_observed_library=do_observed_library,
                z_encoder=eval_encoder,
            )
        out = out["log_ratio"].cpu()
        log_ratios.append(out)

    log_ratios = torch.cat(log_ratios)
    wi = torch.softmax(log_ratios, 0)
    ess_here = 1.0 / (wi ** 2).sum(0)

    _, khats = psislw(log_ratios.T.clone())

    logging.info("FDR/TPR ...")
    train_indices = train_post.indices
    y_train = Y[train_indices]

    decision_rule_fdr10 = np.zeros(n_picks)
    decision_rule_fdr05 = np.zeros(n_picks)
    decision_rule_fdr20 = np.zeros(n_picks)
    decision_rule_tpr10 = np.zeros(n_picks)
    decision_rule_fdr10_plugin = np.zeros(n_picks)
    decision_rule_tpr10_plugin = np.zeros(n_picks)
    fdr_gt = np.zeros((N_GENES, n_picks))
    pe_fdr = np.zeros((N_GENES, n_picks))
    fdr_gt_plugin = np.zeros((N_GENES, n_picks))
    pe_fdr_plugin = np.zeros((N_GENES, n_picks))
    y_preds_is = np.zeros((N_GENES, n_picks))
    y_preds_plugin = np.zeros((N_GENES, n_picks))
    y_gt = np.zeros((N_GENES, n_picks))

    np.random.seed(42)
    for ipick in range(n_picks):
        print(np.unique(y_train))
        if DO_POISSON:
            where_a = np.where(y_train == 0)[0]
            where_b = np.where(y_train == 1)[0]
        else:
            where_a = np.where(y_train == 1)[0]
            where_b = np.where(y_train == 2)[0]

        samples_a = np.random.choice(where_a, size=n_cells)
        samples_b = np.random.choice(where_b, size=n_cells)

        samples_a_overall = train_indices[samples_a]
        samples_b_overall = train_indices[samples_b]

        h_a = h[samples_a_overall]
        h_b = h[samples_b_overall]
        lfc_loc = h_a - h_b
        is_significant_de_local = (lfc_loc.abs() >= 0.5).float().mean(0) >= 0.5
        is_significant_de_local = is_significant_de_local.numpy()

        logging.info("IS flavor ...")
        multicounts_eval = None
        if counts_eval is not None:
            multicounts_eval = (n_samples_queries / counts_eval.sum()) * counts_eval
            multicounts_eval = multicounts_eval.astype(int)
        y_pred_is = get_predictions(
            train_post,
            samples_a,
            samples_b,
            encoder_key=encoder_eval_name,
            counts=multicounts_eval,
            n_post_samples=n_samples_queries,
            importance_sampling=True,
            do_observed_library=do_observed_library,
            encoder=eval_encoder,
        )
        y_pred_is = y_pred_is.numpy()

        true_fdr_arr = true_fdr(y_true=is_significant_de_local, y_pred=y_pred_is)
        pe_fdr_arr, y_decision_rule = posterior_expected_fdr(y_pred=y_pred_is)
        # Fdr related
        fdr_gt[:, ipick] = true_fdr_arr
        pe_fdr[:, ipick] = pe_fdr_arr

        _, y_decision_rule10 = posterior_expected_fdr(y_pred=y_pred_is, fdr_target=0.1)
        decision_rule_fdr10[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )
        decision_rule_tpr10[ipick] = tpr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )

        _, decision_rule_fdr05 = posterior_expected_fdr(
            y_pred=y_pred_is, fdr_target=0.05
        )
        decision_rule_fdr05[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )
        _, decision_rule_fdr20 = posterior_expected_fdr(
            y_pred=y_pred_is, fdr_target=0.2
        )
        decision_rule_fdr20[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )

        logging.info("Plugin flavor ...")
        y_pred_plugin = get_predictions(
            train_post,
            samples_a,
            samples_b,
            encoder_key=encoder_eval_name,
            counts=multicounts_eval,
            n_post_samples=n_samples_queries,
            importance_sampling=False,
            do_observed_library=do_observed_library,
            encoder=eval_encoder,
        )
        y_pred_plugin = y_pred_plugin.numpy()
        true_fdr_plugin_arr = true_fdr(
            y_true=is_significant_de_local, y_pred=y_pred_plugin
        )
        fdr_gt_plugin[:, ipick] = true_fdr_plugin_arr
        pe_fdr_plugin_arr, y_decision_rule = posterior_expected_fdr(
            y_pred=y_pred_plugin
        )
        pe_fdr_plugin[:, ipick] = pe_fdr_plugin_arr
        _, y_decision_rule10 = posterior_expected_fdr(
            y_pred=y_pred_plugin, fdr_target=0.1
        )
        decision_rule_fdr10_plugin[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )
        decision_rule_tpr10_plugin[ipick] = tpr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )
        y_preds_is[:, ipick] = y_pred_is
        y_preds_plugin[:, ipick] = y_pred_plugin
        y_gt[:, ipick] = is_significant_de_local

    prauc_plugin = np.array(
        [prauc(y=y_it, pred=y_pred) for (y_it, y_pred) in zip(y_gt.T, y_preds_plugin.T)]
    )

    # prauc_is = None
    prauc_is = np.array(
        [prauc(y=y_it, pred=y_pred) for (y_it, y_pred) in zip(y_gt.T, y_preds_is.T)]
    )
    all_fdr_gt = np.array(fdr_gt)
    all_pe_fdr = np.array(pe_fdr)
    fdr_gt_plugin = np.array(fdr_gt_plugin)
    fdr_diff = all_fdr_gt - all_pe_fdr
    loop_res = dict(
        iwelbo5000=np.array(iwelbo5000_loss),
        iwelbo5000_train=np.array(iwelbo5000train_loss),
        pe_fdr_plugin=pe_fdr_plugin,
        fdr_gt_plugin=fdr_gt_plugin,
        all_fdr_gt=all_fdr_gt,
        all_pe_fdr=all_pe_fdr,
        l1_fdr=np.linalg.norm(fdr_diff, axis=0, ord=1),
        l2_fdr=np.linalg.norm(fdr_diff, axis=0, ord=2),
        y_gt=y_gt,
        y_pred_is=y_preds_is,
        y_pred_plugin=y_preds_plugin,
        prauc_plugin=prauc_plugin,
        prauc_is=prauc_is,
        fdr_controlled_fdr10=np.array(decision_rule_fdr10),
        fdr_controlled_fdr05=np.array(decision_rule_fdr05),
        fdr_controlled_fdr20=np.array(decision_rule_fdr20),
        fdr_controlled_tpr10=np.array(decision_rule_tpr10),
        fdr_controlled_fdr10_plugin=np.array(decision_rule_fdr10_plugin),
        fdr_controlled_tpr10_plugin=np.array(decision_rule_tpr10_plugin),
        khat_10000=np.array(khats),
        lfc_gt=lfc_loc,
        ess=ess_here.numpy(),
    )
    return loop_res


"""
    Decision theory: Cross Experiment for FDR control
"""


import logging
import pandas as pd
import torch
import torch.nn as nn

from sbvae.inference import UnsupervisedTrainer
from sbvae.models import VAE
from sbvae.models.modules import Encoder, EncoderIAF, EncoderStudent

FILENAME = "symsim-refactor_logpoisson_med_small_d1"

# FILENAME = "symsim-refactor_dbug"
COUNTS_EVAL = pd.Series(dict(IWELBO=50, CUBO=50, REVKL=50, prior=50,))

MODEL_DIR = "models/{}".format(FILENAME)
DO_OBSERVED_LIBRARY = True
SCHEDULE_N = 50
N_IWSAMPLES = 500

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Training parameters
N_EPOCHS = 100  # High number of epochs sounds vital to reach acceptable levels of khat
LR = 1e-3
N_PARTICULES = 5
TRAIN_SIZE = 0.8
BATCH_SIZE = 128
N_HIDDEN = 128
N_CELLS = 100
N_LATENT = 10

N_PICKS = 1
N_EXPERIMENTS = 15
DO_BATCH_NORM = False


EVAL_ENCODERS = [
    dict(encoder_type="TRAIN", reparam=None, eval_encoder_name="train"),
    dict(encoder_type="AIS", reparam=None, eval_encoder_name="AIS"),
    dict(
        encoder_type="CUBO",
        reparam=True,
        variant="student",
        student_df_eval="learn",
        eval_encoder_name="$\\chi$",
    ),
    dict(encoder_type="ELBO", reparam=True, eval_encoder_name="VAE"),
    dict(encoder_type="IWELBO", reparam=True, eval_encoder_name="IWAE"),
    dict(encoder_type="REVKL", reparam=False, eval_encoder_name="WW"),
    dict(encoder_type="defensive", reparam=True, eval_encoder_name="Mixt."),
]


# 2. Experiment
SCENARIOS = [  # WAKE updates
    # IW Decoder
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_wphi=True,
        n_epochs=N_EPOCHS,
        model_name="VAE",
    ),
    dict(
        loss_gen="IWELBO",
        loss_wvar="IWELBO",
        reparam_wphi=True,
        n_epochs=N_EPOCHS,
        model_name="IWAE",
    ),
    dict(
        loss_gen="IWELBO",
        loss_wvar="REVKL",
        reparam_wphi=False,
        n_epochs=N_EPOCHS,
        model_name="WW",
    ),
    dict(
        loss_gen="IWELBO",
        loss_wvar="CUBO",
        reparam_wphi=True,
        n_epochs=N_EPOCHS,
        do_student=True,
        student_df="learn",
        model_name="$\\chi$",
    ),
]


DF_LI = []
for scenario in SCENARIOS:
    loss_gen = scenario["loss_gen"]
    loss_wvar = scenario["loss_wvar"]
    reparam_wphi = scenario["reparam_wphi"]

    n_samples_theta = scenario.get("n_samples_theta", N_PARTICULES)
    n_samples_phi = scenario.get("n_samples_phi", N_PARTICULES)
    batch_size = scenario.get("batch_size", BATCH_SIZE)
    iaf_t = scenario.get("iaf_t", 0)
    model_name = scenario.get("model_name", "{}_{}".format(loss_gen, loss_wvar))

    counts = scenario.get("counts", None)
    n_epochs = scenario.get("n_epochs", N_EPOCHS)
    do_student = scenario.get("do_student", False)
    student_df = scenario.get("student_df", None)

    gen_model_setup = dict(
        loss_gen=loss_gen,
        loss_wvar=loss_wvar,
        reparam_wphi=reparam_wphi,
        n_samples_theta=n_samples_theta,
        n_samples_phi=n_samples_phi,
        batch_size=batch_size,
        iaf_t=iaf_t,
        model_name=model_name,
        counts=counts,
        n_epochs=n_epochs,
        do_student=do_student,
        student_df=student_df,
    )

    for num in range(N_EXPERIMENTS):
        scenario["num"] = num
        scenario["n_hidden"] = N_HIDDEN
        scenario["n_latent"] = N_LATENT
        scenario["do_batch_norm"] = DO_BATCH_NORM
        mdl_name_prefix = ""
        for st in scenario.values():
            mdl_name_prefix = mdl_name_prefix + str(st) + "_"
        mdl_name_prefix = str(mdl_name_prefix)
        mdl_name = os.path.join(MODEL_DIR, "{}.pt".format(mdl_name_prefix))
        print(mdl_name)
        multi_encoder_keys = (
            ["default"] if loss_wvar != "defensive" else ["CUBO", "EUBO"]
        )
        encoder_eval_name = "default" if loss_wvar != "defensive" else "defensive"
        logging.info("Init model")
        mdl = VAE(
            n_input=N_GENES,
            n_hidden=N_HIDDEN,
            n_latent=N_LATENT,
            n_layers=1,
            prevent_library_saturation=False,  # Maybe this parameter is better set to False
            iaf_t=iaf_t,
            multi_encoder_keys=multi_encoder_keys,
            use_batch_norm=DO_BATCH_NORM,
            do_student=do_student,
            student_df=student_df,
        )
        if os.path.exists(mdl_name):
            print("model exists; loading from .pt")
            mdl.load_state_dict(torch.load(mdl_name))
        mdl.cuda()
        trainer = UnsupervisedTrainer(
            model=mdl,
            gene_dataset=DATASET,
            batch_size=batch_size,
            data_loader_kwargs=dict(pin_memory=False),
        )
        # try:
        logging.info("training moodel if needed")
        if not os.path.exists(mdl_name):
            logging.info("Training model ...")
            if loss_wvar == "defensive":
                print("Training using defensive sampling with counts: ", counts)
                trainer.train_defensive(
                    n_epochs=n_epochs,
                    lr=1e-3,
                    wake_theta=loss_gen,
                    wake_psi=loss_wvar,
                    n_samples_theta=n_samples_theta,
                    n_samples_phi=n_samples_phi,
                    counts=counts,
                )
            else:
                trainer.train(
                    n_epochs=n_epochs,
                    lr=1e-3,
                    wake_theta=loss_gen,
                    wake_psi=loss_wvar,
                    n_samples_theta=n_samples_theta,
                    n_samples_phi=n_samples_phi,
                    reparam=reparam_wphi,
                    do_observed_library=DO_OBSERVED_LIBRARY,
                )
        torch.save(mdl.state_dict(), mdl_name)
        logging.info("training evaluation encoders")
        mdl.eval()

        for encoder_scenario in EVAL_ENCODERS:
            encoder_type = encoder_scenario.get("encoder_type")
            reparam = encoder_scenario.get("reparam")
            encoder_variant = encoder_scenario.get("variant", "MF")
            student_df_eval = encoder_scenario.get("student_df_eval", None)
            eval_encoder_name = encoder_scenario.get("eval_encoder_name", None)
            # encoder_type
            # reparam
            # variant
            # student_df_eval
            # eval_encoder_name
            setup_loop = {
                "eval_encoder_name": eval_encoder_name,
                "wake_theta": loss_gen,
                "wake_psi": loss_wvar,
                "batch_size": batch_size,
                "encoder_type": encoder_type,
                "n_samples_theta": n_samples_theta,
                "n_samples_phi": n_samples_phi,
                "reparam_wphi": reparam_wphi,
                "counts": counts,
                "do_student": do_student,
                "student_df": student_df,
                "n_epochs": n_epochs,
                "iaf_t": iaf_t,
                "n_hidden": N_HIDDEN,
                "n_latent": N_LATENT,
                "n_layers": 1,
                "n_cells": N_CELLS,
                "do_batch_norm": DO_BATCH_NORM,
                "encoder_variant": encoder_variant,
                "student_df_eval": student_df_eval,
            }
            print("Setup loop", setup_loop)
            res_loop = dict()

            if encoder_type == "AIS":
                res_loop = model_ais_evaluation_loop(
                    trainer=trainer, n_latent=10, schedule_n=20, n_picks=1, n_cells=500
                )
                res_loop_10 = model_ais_evaluation_loop(
                    trainer=trainer, n_latent=10, schedule_n=20, n_picks=5, n_cells=50
                )
                res_loop_10 = {key + "_50": val for key, val in res_loop_10.items()}
            # All cases except AIS LOOP
            else:
                if encoder_type == "TRAIN":
                    # TODO: Test OK behavior
                    print("Using training variational posterior for inference")
                    eval_encoder = None
                    encoder_eval_name = encoder_eval_name
                    # do_defensive_eval = False
                    counts_eval = counts
                else:
                    eval_enc_name = os.path.join(
                        MODEL_DIR,
                        "evalencoder_{}_{}_variant{}_df{}.pt".format(
                            encoder_type,
                            mdl_name_prefix,
                            encoder_variant,
                            student_df_eval,
                        ),
                    )
                    print(eval_enc_name)
                    if encoder_type == "defensive":
                        multi_encoder_keys = ["IWELBO", "REVKL", "CUBO"]
                        encoder_eval_name = "defensive"
                        do_defensive_eval = True

                        hparams = dict(
                            n_input=N_GENES,
                            n_output=N_LATENT,
                            n_layers=1,
                            n_hidden=N_HIDDEN,
                            use_batch_norm=DO_BATCH_NORM,
                            dropout_rate=0.1,
                        )
                        encs = dict(
                            IWELBO=Encoder(**hparams),
                            REVKL=Encoder(**hparams),
                            CUBO=EncoderStudent(df="learn", **hparams),
                        )
                        eval_encoder = nn.ModuleDict(encs).to("cuda")
                        counts_eval = COUNTS_EVAL
                        if os.path.exists(eval_enc_name):
                            print("eval enc mdl exists; loading from .pt")
                            eval_encoder.load_state_dict(torch.load(eval_enc_name))
                            eval_encoder.cuda()
                        else:
                            trainer.train_eval_defensive_encoder(
                                eval_encoder,
                                n_epochs=n_epochs,
                                lr=1e-3,
                                n_samples_phi=N_PARTICULES,
                                counts=counts_eval,
                            )

                    else:
                        multi_encoder_keys = ["default"]
                        encoder_eval_name = "default"
                        do_defensive_eval = False
                        encoder_type_psi_loss = encoder_type
                        if encoder_variant == "IAF":
                            encoder_type_psi_loss = "ELBO"
                            eval_encoder = nn.ModuleDict(
                                {
                                    "default": EncoderIAF(
                                        n_in=N_GENES,
                                        n_latent=N_LATENT,
                                        n_cat_list=None,
                                        n_hidden=N_HIDDEN,
                                        n_layers=1,
                                        t=3,
                                        dropout_rate=0.1,
                                        use_batch_norm=DO_BATCH_NORM,
                                        do_h=True,
                                    )
                                }
                            ).to("cuda")

                        elif encoder_variant == "student":
                            eval_encoder = nn.ModuleDict(
                                {
                                    "default": EncoderStudent(
                                        n_input=N_GENES,
                                        n_output=N_LATENT,
                                        df=student_df_eval,
                                        n_layers=1,
                                        n_hidden=N_HIDDEN,
                                        use_batch_norm=DO_BATCH_NORM,
                                        dropout_rate=0.1,
                                    )
                                }
                            ).to("cuda")

                        else:
                            eval_encoder = nn.ModuleDict(
                                {
                                    "default": Encoder(
                                        n_input=N_GENES,
                                        n_output=N_LATENT,
                                        n_layers=1,
                                        n_hidden=N_HIDDEN,
                                        use_batch_norm=DO_BATCH_NORM,
                                        dropout_rate=0.1,
                                    )
                                }
                            ).to("cuda")

                        counts_eval = None
                        if os.path.exists(eval_enc_name):
                            print("eval enc mdl exists; loading from .pt")
                            eval_encoder.load_state_dict(torch.load(eval_enc_name))
                            eval_encoder.cuda()
                        else:
                            trainer.train_eval_encoder(
                                eval_encoder,
                                n_epochs=n_epochs,
                                lr=1e-3,
                                wake_psi=encoder_type_psi_loss,
                                n_samples_phi=N_PARTICULES,
                                reparam=reparam,
                                # do_observed_library=DO_OBSERVED_LIBRARY,
                            )

                    torch.save(eval_encoder.state_dict(), eval_enc_name)
                    logging.info(
                        "Initialized and saved the evaluation encoder {}... defensive {}".format(
                            eval_enc_name, do_defensive_eval
                        )
                    )
                    eval_encoder.eval()

                try:
                    res_loop_100 = model_evaluation_loop(
                        trainer=trainer,
                        eval_encoder=eval_encoder,
                        counts_eval=counts_eval,
                        encoder_eval_name=encoder_eval_name,
                        n_picks=1,
                        n_cells=500,
                        do_observed_library=True,
                        n_iwsamples=N_IWSAMPLES,
                    )
                    res_loop_10 = model_evaluation_loop(
                        trainer=trainer,
                        eval_encoder=eval_encoder,
                        counts_eval=counts_eval,
                        encoder_eval_name=encoder_eval_name,
                        n_picks=5,
                        n_cells=50,
                        do_observed_library=True,
                        n_iwsamples=N_IWSAMPLES,
                    )
                    res_loop_10 = {key + "_50": val for key, val in res_loop_10.items()}
                except:
                    pass

            # Updating Dataframe in any case
            res = {
                **gen_model_setup,
                **setup_loop,
                **res_loop_100,
                **res_loop_10,
            }

            print(res)
            DF_LI.append(res)
            DF = pd.DataFrame(DF_LI)
            DF.to_pickle("{}.pkl".format(FILENAME))

DF = pd.DataFrame(DF_LI)
DF.to_pickle("{}.pkl".format(FILENAME))
