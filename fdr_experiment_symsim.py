"""
    Decision theory: Experiment for FDR control
"""


import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arviz.stats import psislw
from sklearn.metrics import auc, average_precision_score, precision_score, recall_score
from tqdm.auto import tqdm

from sbvae.dataset import GeneExpressionDataset
from sbvae.inference import UnsupervisedTrainer
from sbvae.models import VAE

NUMS = 1
FILENAME = "simu_fdr_symsim"
MODEL_DIR = "models/fdr_symsim"
DO_OBSERVED_LIBRARY = True

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Training parameters
N_EPOCHS = 200  # High number of epochs sounds vital to reach acceptable levels of khat
LR = 1e-3
TRAIN_SIZE = 0.8
BATCH_SIZE = 128
N_HIDDEN_ARR = [128]
# Params for FDR measurements
N_PICKS = 5


# 1. Dataset
SYMSIM_DATA_PATH = "DE"
X_OBS_ALL = pd.read_csv(
    os.path.join(SYMSIM_DATA_PATH, "DE_med.obsv.3.csv"), index_col=0
).T
SELECT_GENE = np.where(X_OBS_ALL.mean(0) <= 1000)[0]
X_OBS = X_OBS_ALL.iloc[:, SELECT_GENE]
BATCH_INFO = (
    pd.read_csv(os.path.join(SYMSIM_DATA_PATH, "DE_med.batchid.csv"), index_col=0) - 1
)
METADATA = pd.read_csv(
    os.path.join(SYMSIM_DATA_PATH, "DE_med.cell_meta.csv"), index_col=0
)
TRUE_ = pd.read_csv(
    os.path.join(SYMSIM_DATA_PATH, "DE_med.true.csv"), index_col=0
).T.iloc[:, SELECT_GENE]
LFC_INFO = pd.read_csv(
    os.path.join(SYMSIM_DATA_PATH, "med_theoreticalFC.csv"), index_col=0
).iloc[SELECT_GENE, :]
DATASET = GeneExpressionDataset(
    *GeneExpressionDataset.get_attributes_from_matrix(
        X=X_OBS.values,
        batch_indices=BATCH_INFO["x"].values,
        labels=METADATA["pop"].values,
    )
)

SAMPLE_IDX = np.loadtxt(
    os.path.join(SYMSIM_DATA_PATH, "sample_idx_symsim.tsv"), dtype=np.int32
)
X_U = torch.tensor(DATASET.X[SAMPLE_IDX]).to("cuda")
LOCAL_L_MEAN = torch.tensor(DATASET.local_vars[SAMPLE_IDX]).to("cuda")
LOCAL_L_VAR = torch.tensor(DATASET.local_means[SAMPLE_IDX]).to("cuda")
LABEL_A = 0
LABEL_B = 1
N_GENES = DATASET.nb_genes
Y = METADATA["pop"].values
IS_SIGNIFICANT_DE = (LFC_INFO["12"].abs() >= 0.5).values
LFC_GT = LFC_INFO["12"].values


def get_predictions_is(post, samp_a, samp_b, encoder_key, counts, n_post_samples=50):
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
        do_observed_library=DO_OBSERVED_LIBRARY,
        encoder_key=encoder_key,
        counts=my_multicounts,
    )

    w_a = post_vals["log_ratio"][:, samp_a]
    w_a = softmax(w_a).view(n_post_samples, 1, n_samples, 1)
    w_b = post_vals["log_ratio"][:, samp_b]
    w_b = softmax(w_b).view(1, n_post_samples, n_samples, 1)

    y_pred = post_vals["px_scale"][:, samp_a].log2().view(
        n_post_samples, 1, n_samples, -1
    ) - post_vals["px_scale"][:, samp_b].log2().view(1, n_post_samples, n_samples, -1)
    y_pred = (y_pred.abs() >= 0.5).float()
    y_pred = y_pred * w_a * w_b
    y_pred = y_pred.sum([0, 1]).mean(0)
    return y_pred


def fdr_score(y_true, y_pred):
    return 1.0 - precision_score(y_true, y_pred)


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


# 2. Experiment
SCENARIOS = [  # WAKE updates
    dict(
        loss_gen="ELBO",
        loss_wvar="defensive",
        reparam_wphi=True,
        n_samples_theta=25,
        counts=torch.tensor([1, 1, 0]),
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="CUBO",
        reparam_wphi=True,
        n_samples_theta=1,
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="REVKL",
        reparam_wphi=False,
        n_samples_theta=1,
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        reparam_wphi=True,
        n_samples_theta=1,
        n_samples_phi=1,
        iaf_t=0,
        n_epochs=200,
    ),
]


if __name__ == "__main__":
    DF_LI = []
    for scenario in SCENARIOS:
        loss_gen = scenario["loss_gen"]
        loss_wvar = scenario["loss_wvar"]
        n_samples_theta = scenario["n_samples_theta"]
        n_samples_phi = scenario["n_samples_phi"]
        reparam_wphi = scenario["reparam_wphi"]
        iaf_t = scenario["iaf_t"]
        if "batch_size" in scenario:
            batch_size = scenario["batch_size"]
        else:
            batch_size = BATCH_SIZE
        if "counts" in scenario:
            counts = scenario["counts"]
        else:
            counts = None

        if "n_epochs" in scenario:
            n_epochs = scenario["n_epochs"]
        else:
            n_epochs = 200

        for n_hidden in N_HIDDEN_ARR:
            cubo_arr = []
            iwelbo_arr = []
            khat_arr_100 = []
            khat_arr_25 = []
            khat_arr_10000 = []
            ess = []
            fdr_l2_err = []
            fdr_l1_err = []
            precision_arr = []
            auc_arr = []
            all_fdr_gt = []
            all_pe_fdr = []
            fdr_controlled_fdr = []
            fdr_controlled_tpr = []
            fdr_controlled_fdr10 = []
            fdr_controlled_tpr10 = []

            for num in range(NUMS):
                scenario["num"] = num
                scenario["n_hidden"] = n_hidden
                mdl_name = ""
                for st in scenario.values():
                    mdl_name = mdl_name + str(st) + "_"
                mdl_name = str(mdl_name)
                mdl_name = os.path.join(MODEL_DIR, "{}.pt".format(mdl_name))
                print(mdl_name)
                multi_encoder_keys = (
                    ["default"] if loss_wvar != "defensive" else ["CUBO", "EUBO"]
                )
                encoder_eval_name = "default" if loss_wvar != "defensive" else "defensive"

                mdl = VAE(
                    n_input=N_GENES,
                    n_hidden=n_hidden,
                    n_latent=10,
                    n_layers=1,
                    prevent_library_saturation=False,  # Maybe this parameter is better set to False
                    iaf_t=iaf_t,
                    multi_encoder_keys=multi_encoder_keys,
                )
                if os.path.exists(mdl_name):
                    print("model exists; loading from .pt")
                    mdl.load_state_dict(torch.load(mdl_name))
                mdl.cuda()
                trainer = UnsupervisedTrainer(
                    model=mdl, gene_dataset=DATASET, batch_size=batch_size,
                )
                # try:
                if not os.path.exists(mdl_name):
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

                test_post = trainer.test_set.sequential()
                train_post = trainer.train_set.sequential()

                # *** CUBO
                multicounts = None
                if counts is not None:
                    multicounts = (100 / counts.sum()) * counts
                cubo_loss = (
                    test_post.getter(
                        keys=["CUBO"],
                        n_samples=100,
                        do_observed_library=DO_OBSERVED_LIBRARY,
                        encoder_key=encoder_eval_name,
                        counts=multicounts,
                    )["CUBO"]
                    .cpu()
                    .numpy()
                ).mean()
                cubo_arr.append(cubo_loss)

                # *** IWELBO
                iwelbo_loss = (
                    test_post.getter(
                        keys=["IWELBO"],
                        n_samples=100,
                        do_observed_library=DO_OBSERVED_LIBRARY,
                        encoder_key=encoder_eval_name,
                        counts=multicounts,
                    )["IWELBO"]
                    .cpu()
                    .numpy()
                ).mean()
                iwelbo_arr.append(iwelbo_loss)

                # *** KHAT
                # Using 1000 samples
                log_ratios = []
                n_samples_total = 1e4
                n_samples_per_pass = 25 if encoder_eval_name == "default" else counts.sum()
                n_iter = int(n_samples_total / n_samples_per_pass)
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
                            counts=multicounts,
                            do_observed_library=DO_OBSERVED_LIBRARY,
                        )
                    out = out["log_ratio"].cpu()
                    log_ratios.append(out)

                log_ratios = torch.cat(log_ratios)
                wi = torch.softmax(log_ratios, 0)
                ess_here = 1.0 / (wi ** 2).sum(0)

                _, khats = psislw(log_ratios.T.clone())

                khat_arr_10000.append(khats)
                ess.append(ess_here.numpy())

                # B. SymSim
                test_indices = train_post.indices
                y_test = Y[test_indices]

                IS_SIGNIFICANT_DE = (LFC_INFO["12"].abs() >= 0.5).values
                l2_errs = np.zeros(N_PICKS)
                l1_errs = np.zeros(N_PICKS)
                m_ap_vals = np.zeros(N_PICKS)
                auc_vals = np.zeros(N_PICKS)
                decision_rule_fdr = np.zeros(N_PICKS)
                decision_rule_tpr = np.zeros(N_PICKS)
                decision_rule_fdr10 = np.zeros(N_PICKS)
                decision_rule_tpr10 = np.zeros(N_PICKS)
                fdr_gt = np.zeros((N_GENES, N_PICKS))
                pe_fdr = np.zeros((N_GENES, N_PICKS))
                for ipick in range(N_PICKS):
                    samples_a = np.random.choice(np.where(y_test == 1)[0], size=10)
                    samples_b = np.random.choice(np.where(y_test == 2)[0], size=10)

                    y_pred_is = get_predictions_is(
                        train_post,
                        samples_a,
                        samples_b,
                        encoder_key=encoder_eval_name,
                        counts=multicounts,
                        n_post_samples=200,
                    )
                    y_pred_is = y_pred_is.numpy()

                    true_fdr_arr = true_fdr(y_true=IS_SIGNIFICANT_DE, y_pred=y_pred_is)
                    pe_fdr_arr, y_decision_rule = posterior_expected_fdr(y_pred=y_pred_is)
                    # Fdr related
                    fdr_gt[:, ipick] = true_fdr_arr
                    pe_fdr[:, ipick] = pe_fdr_arr

                    l2_errs[ipick] = np.linalg.norm(true_fdr_arr - pe_fdr_arr, ord=2)
                    l1_errs[ipick] = np.linalg.norm(true_fdr_arr - pe_fdr_arr, ord=1)

                    # Overall classification
                    m_ap_vals[ipick] = average_precision_score(IS_SIGNIFICANT_DE, y_pred_is)
                    auc_vals[ipick] = auc(IS_SIGNIFICANT_DE, y_pred_is)

                    # Decision rule related
                    decision_rule_fdr[ipick] = 1 - precision_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule
                    )
                    decision_rule_tpr[ipick] = recall_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule
                    )

                    _, y_decision_rule10 = posterior_expected_fdr(
                        y_pred=y_pred_is, fdr_target=0.1
                    )
                    decision_rule_fdr10[ipick] = 1 - precision_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule10
                    )
                    decision_rule_tpr10[ipick] = recall_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule10
                    )

                all_fdr_gt.append(fdr_gt)
                all_pe_fdr.append(pe_fdr)
                fdr_l1_err.append(l1_errs)
                fdr_l2_err.append(l2_errs)
                precision_arr.append(m_ap_vals)
                auc_arr.append(auc_vals)
                fdr_controlled_fdr.append(decision_rule_fdr)
                fdr_controlled_tpr.append(decision_rule_tpr)
                fdr_controlled_fdr10.append(decision_rule_fdr10)
                fdr_controlled_tpr10.append(decision_rule_tpr10)

            res = {
                "wake_theta": loss_gen,
                "wake_psi": loss_wvar,
                "batch_size": batch_size,
                "n_samples_theta": n_samples_theta,
                "n_samples_phi": n_samples_phi,
                "reparam_wphi": reparam_wphi,
                "counts": counts,
                "n_epochs": n_epochs,
                "iaf_t": iaf_t,
                "n_hidden": n_hidden,
                "n_latent": 10,
                "n_layers": 1,
                "cubo": cubo_arr,
                "iwelbo": iwelbo_arr,
                "khat_100": khat_arr_100,
                "khat_25": khat_arr_25,
                "khat_10000": np.array(khat_arr_10000),
                "ess": np.array(ess),
                "fdr_gt": np.array(all_fdr_gt),
                "pe_fdr": np.array(all_pe_fdr),
                "fdr_l1": np.array(fdr_l1_err),
                "fdr_l2": np.array(fdr_l2_err),
                "m_ap": np.array(precision_arr),
                "auc": np.array(auc_arr),
                "fdr_controlled_fdr": np.array(fdr_controlled_fdr),
                "fdr_controlled_tpr": np.array(fdr_controlled_tpr),
                "fdr_controlled_fdr10": np.array(fdr_controlled_fdr10),
                "fdr_controlled_tpr10": np.array(fdr_controlled_tpr10),        
            }

            print(res)
            DF_LI.append(res)
            DF = pd.DataFrame(DF_LI)
            DF.to_csv("{}.csv".format(FILENAME), sep="\t")
            DF.to_pickle("{}.pkl".format(FILENAME))
    DF = pd.DataFrame(DF_LI)
    DF.to_csv("{}.csv".format(FILENAME), sep="\t")
    DF.to_pickle("{}.pkl".format(FILENAME))
