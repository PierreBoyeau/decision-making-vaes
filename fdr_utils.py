import logging
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score
from torch import nn
from arviz.stats import psislw

from sbvae.ais import ais_trajectory
from sbvae.dataset import GeneExpressionDataset


NUMS = 5
N_PICKS = 30
N_CELLS = 5
COUNTS_EVAL = torch.tensor([1, 1, 0])

# 1. Dataset
## Option 1
# DO_POISSON = False
# SYMSIM_DATA_PATH = "/data/yosef2/users/pierreboyeau/sbVAE/DE"

# ## Option 2
# DO_POISSON = False
# # SYMSIM_DATA_PATH = "/data/yosef2/users/pierreboyeau/symsim_result_complete/DE"
# # K_ON = pd.read_csv(os.path.join(SYMSIM_DATA_PATH, "DE_med.kon_mat.csv")).values.T
# # K_OFF = pd.read_csv(os.path.join(SYMSIM_DATA_PATH, "DE_med.koff_mat.csv")).values.T
# # S_MAT = pd.read_csv(os.path.join(SYMSIM_DATA_PATH, "DE_med.s_mat.csv")).values.T


# X_OBS_ALL = pd.read_csv(
#     os.path.join(SYMSIM_DATA_PATH, "DE_med.obsv.3.csv"), index_col=0
# ).T
# # SELECT_GENE = np.where(X_OBS_ALL.mean(0) <= 1000)[0]
# SELECT_GENE = np.arange(X_OBS_ALL.shape[1])
# X_OBS = X_OBS_ALL.iloc[:, SELECT_GENE]
# BATCH_INFO = (
#     pd.read_csv(os.path.join(SYMSIM_DATA_PATH, "DE_med.batchid.csv"), index_col=0) - 1
# )
# METADATA = pd.read_csv(
#     os.path.join(SYMSIM_DATA_PATH, "DE_med.cell_meta.csv"), index_col=0
# )
# TRUE_ = pd.read_csv(
#     os.path.join(SYMSIM_DATA_PATH, "DE_med.true.csv"), index_col=0
# ).T.iloc[:, SELECT_GENE]
# LFC_INFO = pd.read_csv(
#     os.path.join(SYMSIM_DATA_PATH, "med_theoreticalFC.csv"), index_col=0
# ).iloc[SELECT_GENE, :]
# DATASET = GeneExpressionDataset(
#     *GeneExpressionDataset.get_attributes_from_matrix(
#         X=X_OBS.values,
#         batch_indices=BATCH_INFO["x"].values,
#         labels=METADATA["pop"].values,
#     )
# )

# SAMPLE_IDX = np.loadtxt(
#     os.path.join(SYMSIM_DATA_PATH, "sample_idx_symsim.tsv"), dtype=np.int32
# )
# X_U = torch.tensor(DATASET.X[SAMPLE_IDX]).to("cuda")
# LOCAL_L_MEAN = torch.tensor(DATASET.local_vars[SAMPLE_IDX]).to("cuda")
# LOCAL_L_VAR = torch.tensor(DATASET.local_means[SAMPLE_IDX]).to("cuda")
# LABEL_A = 0
# LABEL_B = 1
# N_GENES = DATASET.nb_genes
# Y = METADATA["pop"].values
# IS_SIGNIFICANT_DE = (LFC_INFO["12"].abs() >= 0.5).values
# LFC_GT = LFC_INFO["12"].values


# option 3
n_genes = 1000
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


means = 10 + 100 * torch.rand(n_genes)
means = means.numpy()

means[means >= 1000] = 1000

lfc_sampler = SignedGamma(dim=2, proba_pos=0.5)
lfcs = lfc_sampler.sample(n_genes).numpy()
non_de_genes = np.random.choice(n_genes, size=300)
lfcs[non_de_genes, :] = 0.0

# Constructing sigma and mus
log2_mu0 = lfcs[:, 0] + np.log2(means)
log2_mu1 = lfcs[:, 1] + np.log2(means)

loge_mu0 = log2_mu0 / np.log2(np.e)
loge_mu1 = log2_mu1 / np.log2(np.e)

a = (2.0 * np.random.random(size=(n_genes, 1)) - 1).astype(float)
sigma = 2.0 * a.dot(a.T) + 0.5 * (
    1.0 + 0.5 * (2.0 * np.random.random(n_genes) - 1.0)
) * np.eye(n_genes)

sigma0 = 0.1 * sigma
sigma1 = sigma0

# Poisson rates
h0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor(loge_mu0).float(), covariance_matrix=torch.tensor(sigma0).float()
).sample((5000,))
h1 = torch.distributions.MultivariateNormal(
    loc=torch.tensor(loge_mu1).float(), covariance_matrix=torch.tensor(sigma1).float()
).sample((5000,))
h = torch.cat([h0, h1])

# Data sampling
x_obs = torch.distributions.Poisson(rate=h.exp()).sample()
# Zero inflation
is_zi = torch.rand_like(x_obs) <= 0.3
# print("Added zeros: ", is_zi.mean())
x_obs = x_obs * (1.0 - is_zi.double())
labels = torch.zeros((10000, 1))
labels[5000:] = 1

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

SAMPLE_IDX = np.random.choice(10000, 64)
Y = labels
X_U = torch.tensor(DATASET.X[SAMPLE_IDX]).to("cuda")
LOCAL_L_MEAN = torch.tensor(DATASET.local_vars[SAMPLE_IDX]).to("cuda")
LOCAL_L_VAR = torch.tensor(DATASET.local_means[SAMPLE_IDX]).to("cuda")
N_GENES = 1000


# def get_predictions(
#     post,
#     samp_a,
#     samp_b,
#     encoder_key,
#     counts,
#     n_post_samples=50,
#     importance_sampling=True,
#     encoder=None,
# ):
#     n_samples = len(samp_a)
#     softmax = nn.Softmax(dim=0)

#     my_multicounts = None
#     if encoder_key == "defensive":
#         factor_counts = n_post_samples // counts.sum()
#         n_post_samples = factor_counts * counts.sum()
#         my_multicounts = factor_counts * counts
#     post_vals = post.getter(
#         keys=["px_scale", "log_ratio"],
#         n_samples=n_post_samples,
#         do_observed_library=True,
#         encoder_key=encoder_key,
#         counts=my_multicounts,
#         z_encoder=encoder,
#     )

#     if importance_sampling:
#         w_a = post_vals["log_ratio"][:, samp_a]
#         w_a = softmax(w_a).view(n_post_samples, 1, n_samples, 1)
#         w_b = post_vals["log_ratio"][:, samp_b]
#         w_b = softmax(w_b).view(1, n_post_samples, n_samples, 1)

#         y_pred = post_vals["px_scale"][:, samp_a].log2().view(
#             n_post_samples, 1, n_samples, -1
#         ) - post_vals["px_scale"][:, samp_b].log2().view(
#             1, n_post_samples, n_samples, -1
#         )
#         y_pred = (y_pred.abs() >= 0.5).float()
#         y_pred = y_pred * w_a * w_b
#         y_pred = y_pred.sum([0, 1]).mean(0)
#     else:
#         y_pred = (
#             post_vals["px_scale"][:, samp_a].log2()
#             - post_vals["px_scale"][:, samp_b].log2()
#         )
#         y_pred = (y_pred.abs() >= 0.5).float()
#         y_pred = y_pred.mean((0, 1))
#     return y_pred


def get_predictions(
    post,
    samp_a,
    samp_b,
    encoder_key,
    counts,
    n_post_samples=50,
    importance_sampling=True,
    encoder=None,
):
    n_samples = len(samp_a)
    softmax = nn.Softmax(dim=0)

    my_multicounts = None
    if encoder_key == "defensive":
        factor_counts = n_post_samples // counts.sum()
        n_post_samples = factor_counts * counts.sum()
        my_multicounts = factor_counts * counts
    post_vals = post.getter(
        keys=["px_scale", "log_ratio"],
        n_samples=n_post_samples,
        do_observed_library=True,
        encoder_key=encoder_key,
        counts=my_multicounts,
        z_encoder=encoder,
    )

    if importance_sampling:
        # sampling pairs of cells
        cells_a = np.random.choice(samp_a, size=4 * n_samples)
        cells_b = np.random.choice(samp_b, size=4 * n_samples)

        all_preds = []
        for cell_a, cell_b in zip(tqdm(cells_a), cells_b):
            w_a = post_vals["log_ratio"][:, cell_a]
            w_a = softmax(w_a).view(n_post_samples, 1, 1)
            w_b = post_vals["log_ratio"][:, cell_b]
            w_b = softmax(w_b).view(1, n_post_samples, 1)

            y_pred = post_vals["px_scale"][:, cell_a].log2().view(
                n_post_samples, 1, -1
            ) - post_vals["px_scale"][:, cell_b].log2().view(1, n_post_samples, -1)
            y_pred = (y_pred.abs() >= 0.5).float()
            y_pred = y_pred * w_a * w_b
            y_pred = y_pred.sum([0, 1])
            all_preds.append(y_pred.unsqueeze(0))
        all_preds = torch.cat(all_preds, dim=0)
        return all_preds.mean(0)
    else:
        y_pred = (
            post_vals["px_scale"][:, samp_a].log2()
            - post_vals["px_scale"][:, samp_b].log2()
        )
        y_pred = (y_pred.abs() >= 0.5).float()
        y_pred = y_pred.mean((0, 1))
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
        log_h_a = (
            model.decoder(
                model.dispersion,
                z_a.cuda(),
                torch.ones(len(z_a), n_batch, 1, device="cuda"),
                None,
            )[0]
            .log2()
            .cpu()
            .view(n_post_samples, 1, n_batch, -1)
        )
        log_h_b = (
            model.decoder(
                model.dispersion,
                z_b.cuda(),
                torch.ones(len(z_b), n_batch, 1, device="cuda"),
                None,
            )[0]
            .log2()
            .cpu()
            .view(1, n_post_samples, n_batch, -1)
        )
        w_a = softmax(log_wa).view(n_post_samples, 1, n_batch, 1)
        w_b = softmax(log_wb).view(1, n_post_samples, n_batch, 1)

        y_pred_is = ((log_h_a - log_h_b).abs() >= 0.5).float()
        y_pred_is = y_pred_is * w_a * w_b
        y_pred_is = y_pred_is.sum([0, 1]).mean(0)

    y_pred_is = y_pred_is.numpy()
    return y_pred_is


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
    true_fdr_array = np.zeros(n_genes)
    for idx in range(1, len(probas_sorted) + 1):
        y_pred_tresh = np.zeros(n_genes, dtype=bool)
        where_pos = probas_sorted[:idx]
        y_pred_tresh[where_pos] = True
        # print(y_pred_tresh)
        true_fdr_array[idx - 1] = fdr_score(y_true, y_pred_tresh)
    return true_fdr_array


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


def model_ais_evaluation_loop(trainer, n_latent):
    schedule = np.linspace(0, 1, 20)
    mdl = trainer.model
    train_post = trainer.train_set.sequential(batch_size=128)
    _, logwa = ais_trajectory(
        model=mdl, schedule=schedule, loader=train_post, n_sample=50, n_latent=n_latent,
    )
    iwelbo5000_train = (torch.logsumexp(logwa, dim=0) - np.log(logwa.shape[0])).mean()

    test_post = trainer.test_set.sequential(batch_size=128)
    _, logwa = ais_trajectory(
        model=mdl, schedule=schedule, loader=test_post, n_sample=50, n_latent=n_latent,
    )
    iwelbo5000 = (torch.logsumexp(logwa, dim=0) - np.log(logwa.shape[0])).mean()

    logging.info("FDR/TPR ...")
    train_post = trainer.train_set.sequential()
    train_indices = train_post.indices
    y_train = Y[train_indices]

    decision_rule_fdr10 = np.zeros(N_PICKS)
    decision_rule_tpr10 = np.zeros(N_PICKS)
    decision_rule_fdr10_plugin = np.zeros(N_PICKS)
    decision_rule_tpr10_plugin = np.zeros(N_PICKS)
    fdr_gt = np.zeros((N_GENES, N_PICKS))
    pe_fdr = np.zeros((N_GENES, N_PICKS))
    for ipick in range(N_PICKS):
        print(np.unique(y_train))
        if DO_POISSON:
            where_a = np.where(y_train == 0)[0]
            where_b = np.where(y_train == 1)[0]
        else:
            where_a = np.where(y_train == 1)[0]
            where_b = np.where(y_train == 2)[0]

        samples_a = np.random.choice(where_a, size=N_CELLS)
        samples_b = np.random.choice(where_b, size=N_CELLS)

        samples_a_overall = train_indices[samples_a]
        samples_b_overall = train_indices[samples_b]
        # means_a = (
        #     S_MAT[samples_a_overall]
        #     * K_ON[samples_a_overall]
        #     / (K_ON[samples_a_overall] + K_OFF[samples_a_overall])
        # )
        # means_b = (
        #     S_MAT[samples_b_overall]
        #     * K_ON[samples_b_overall]
        #     / (K_ON[samples_b_overall] + K_OFF[samples_b_overall])
        # )
        # lfc_dist_gt = np.log2(means_a) - np.log2(means_b)
        # is_significant_de_local = (np.abs(lfc_dist_gt) >= 0.5).mean(0) >= 0.5
        is_significant_de_local = IS_SIGNIFICANT_DE

        logging.info("IS flavor ...")
        schedule = np.linspace(0.0, 1.0, 100)
        post_a = trainer.create_posterior(
            model=mdl, gene_dataset=DATASET, indices=samples_a_overall
        ).sequential(batch_size=5)

        post_b = trainer.create_posterior(
            model=mdl, gene_dataset=DATASET, indices=samples_b_overall
        ).sequential(batch_size=5)
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
        return dict(
            iwelbo5000=np.array(iwelbo5000),
            iwelbo5000_train=np.array(iwelbo5000_train),
            all_fdr_gt=np.array(fdr_gt),
            all_pe_fdr=np.array(pe_fdr),
            fdr_controlled_fdr10=np.array(decision_rule_fdr10),
            fdr_controlled_tpr10=np.array(decision_rule_tpr10),
            fdr_controlled_fdr10_plugin=None,
            fdr_controlled_tpr10_plugin=None,
            khat_10000=None,
            ess=None,
        )


def model_evaluation_loop(
    trainer, eval_encoder, counts_eval, encoder_eval_name,
):
    test_post = trainer.test_set.sequential()
    mdl = trainer.model
    train_post = trainer.train_set.sequential()

    # *** IWELBO 5000
    logging.info("IWELBO 5K estimation...")
    multicounts_eval = None
    if counts_eval is not None:
        multicounts_eval = (5000 / counts_eval.sum()) * counts_eval
    iwelbo5000_loss = (
        test_post.getter(
            keys=["IWELBO"],
            n_samples=5000,
            batch_size=64,
            do_observed_library=True,
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
            n_samples=5000,
            batch_size=64,
            do_observed_library=True,
            encoder_key=encoder_eval_name,
            counts=multicounts_eval,
            z_encoder=eval_encoder,
        )["IWELBO"]
        .cpu()
        .numpy()
    ).mean()

    # *** KHAT
    multicounts = None
    if counts_eval is not None:
        multicounts = (5000 / counts_eval.sum()) * counts_eval
    log_ratios = []
    n_samples_total = 1e4
    n_samples_per_pass = 25 if encoder_eval_name == "default" else multicounts.sum()
    n_iter = int(n_samples_total / n_samples_per_pass)
    logging.info("Multicounts: {}".format(multicounts))
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
                do_observed_library=True,
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

    decision_rule_fdr10 = np.zeros(N_PICKS)
    decision_rule_tpr10 = np.zeros(N_PICKS)
    decision_rule_fdr10_plugin = np.zeros(N_PICKS)
    decision_rule_tpr10_plugin = np.zeros(N_PICKS)
    fdr_gt = np.zeros((N_GENES, N_PICKS))
    pe_fdr = np.zeros((N_GENES, N_PICKS))
    for ipick in range(N_PICKS):
        print(np.unique(y_train))
        if DO_POISSON:
            where_a = np.where(y_train == 0)[0]
            where_b = np.where(y_train == 1)[0]
        else:
            where_a = np.where(y_train == 1)[0]
            where_b = np.where(y_train == 2)[0]

        samples_a = np.random.choice(where_a, size=N_CELLS)
        samples_b = np.random.choice(where_b, size=N_CELLS)

        samples_a_overall = train_indices[samples_a]
        samples_b_overall = train_indices[samples_b]

        # means_a = (
        #     S_MAT[samples_a_overall]
        #     * K_ON[samples_a_overall]
        #     / (K_ON[samples_a_overall] + K_OFF[samples_a_overall])
        # )
        # means_b = (
        #     S_MAT[samples_b_overall]
        #     * K_ON[samples_b_overall]
        #     / (K_ON[samples_b_overall] + K_OFF[samples_b_overall])
        # )
        # lfc_dist_gt = np.log2(means_a) - np.log2(means_b)
        # is_significant_de_local = (np.abs(lfc_dist_gt) >= 0.5).mean(0) >= 0.5
        is_significant_de_local = IS_SIGNIFICANT_DE

        logging.info("IS flavor ...")
        multicounts_eval = None
        if counts_eval is not None:
            multicounts_eval = (200 / counts_eval.sum()) * counts_eval
        y_pred_is = get_predictions(
            train_post,
            samples_a,
            samples_b,
            encoder_key=encoder_eval_name,
            counts=multicounts_eval,
            n_post_samples=200,
            importance_sampling=True,
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

        logging.info("Plugin flavor ...")
        y_pred_plugin = get_predictions(
            train_post,
            samples_a,
            samples_b,
            encoder_key=encoder_eval_name,
            counts=multicounts_eval,
            n_post_samples=200,
            importance_sampling=False,
            encoder=eval_encoder,
        )
        y_pred_plugin = y_pred_plugin.numpy()
        true_fdr_arr = true_fdr(y_true=is_significant_de_local, y_pred=y_pred_plugin)

        _, y_decision_rule10 = posterior_expected_fdr(
            y_pred=y_pred_plugin, fdr_target=0.1
        )
        decision_rule_fdr10_plugin[ipick] = fdr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )
        decision_rule_tpr10_plugin[ipick] = tpr_score(
            y_true=is_significant_de_local, y_pred=y_decision_rule10
        )

    loop_res = dict(
        iwelbo5000=np.array(iwelbo5000_loss),
        iwelbo5000_train=np.array(iwelbo5000train_loss),
        all_fdr_gt=np.array(fdr_gt),
        all_pe_fdr=np.array(pe_fdr),
        fdr_controlled_fdr10=np.array(decision_rule_fdr10),
        fdr_controlled_tpr10=np.array(decision_rule_tpr10),
        fdr_controlled_fdr10_plugin=np.array(decision_rule_fdr10_plugin),
        fdr_controlled_tpr10_plugin=np.array(decision_rule_tpr10_plugin),
        khat_10000=np.array(khats),
        ess=ess_here.numpy(),
    )
    return loop_res
