"""
    Decision theory: Experiment for FDR control
"""


import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arviz.stats import psislw
from sklearn.metrics import auc, average_precision_score, precision_score, recall_score
from tqdm.auto import tqdm
import logging

from sbvae.dataset import GeneExpressionDataset
from sbvae.inference import UnsupervisedTrainer
from sbvae.models import VAE
from sbvae.models.modules import Encoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

NUMS = 5
FILENAME = "simu_cross_fdr_symsim_with_fdr  "
MODEL_DIR = "models/fdr_symsim"
DO_OBSERVED_LIBRARY = True

N_SAMPLES_PHI_EVAL = 25

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Training parameters
N_EPOCHS = 200  # High number of epochs sounds vital to reach acceptable levels of khat
LR = 1e-3
TRAIN_SIZE = 0.8
BATCH_SIZE = 128
N_HIDDEN = 128
# Params for FDR measurements
N_PICKS = 5

EVAL_ENCODERS = [
    ("defensive", True),
    ("ELBO", True),
    ("REVKL",False), 
    ("CUBO", True),
    ("IWELBO", True),
]
COUNTS_EVAL = torch.tensor([1, 1, 0])

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


def get_predictions(post, samp_a, samp_b, encoder_key, counts, n_post_samples=50, importance_sampling=True, encoder=None):
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
        z_encoder=encoder,
    )

    if importance_sampling:
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
    else:
        y_pred = post_vals["px_scale"][:, samp_a].log2() - post_vals["px_scale"][:, samp_b].log2()
        y_pred = (y_pred.abs() >= 0.5).float()
        y_pred = y_pred.mean((0, 1))
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
        loss_wvar="defensive",
        reparam_wphi=True,
        n_samples_theta=1,
        counts=torch.tensor([1, 1, 0]),
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="IWELBO",
        reparam_wphi=True,
        n_samples_theta=1,
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

    # IW Decoder
    dict(
        loss_gen="IWELBO",
        loss_wvar="defensive",
        reparam_wphi=True,
        n_samples_theta=25,
        counts=torch.tensor([1, 1, 0]),
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="IWELBO",
        loss_wvar="IWELBO",
        reparam_wphi=True,
        n_samples_theta=25,
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="IWELBO",
        loss_wvar="CUBO",
        reparam_wphi=True,
        n_samples_theta=25,
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="IWELBO",
        loss_wvar="REVKL",
        reparam_wphi=False,
        n_samples_theta=25,
        n_samples_phi=25,
        iaf_t=0,
        n_epochs=200,
    ),
    dict(
        loss_gen="IWELBO",
        loss_wvar="ELBO",
        reparam_wphi=True,
        n_samples_theta=25,
        n_samples_phi=25,
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
        
        for num in range(NUMS):
            scenario["num"] = num
            scenario["n_hidden"] = N_HIDDEN
            mdl_name_prefix = ""
            for st in scenario.values():
                mdl_name_prefix = mdl_name_prefix + str(st) + "_"
            mdl_name_prefix = str(mdl_name_prefix)
            mdl_name = os.path.join(MODEL_DIR, "{}.pt".format(mdl_name_prefix))
            print(mdl_name)
            multi_encoder_keys = (
                ["default"] if loss_wvar != "defensive" else ["CUBO", "EUBO"]
            )
            encoder_eval_name = (
                "default" if loss_wvar != "defensive" else "defensive"
            )
            logging.info("Init model")
            mdl = VAE(
                n_input=N_GENES,
                n_hidden=N_HIDDEN,
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
                model=mdl, gene_dataset=DATASET, batch_size=batch_size, data_loader_kwargs=dict(pin_memory=False)
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
            for encoder_type, reparam in EVAL_ENCODERS:

                cubo_arr = []
                cubo5000_arr = []
                iwelbo_arr = []
                iwelbo5000_arr = []
                khat_arr_100 = []
                khat_arr_25 = []
                khat_arr_10000 = []
                ess = []
                all_fdr_gt = []
                all_pe_fdr = []
                fdr_controlled_fdr10 = []
                fdr_controlled_tpr10 = []
                fdr_controlled_fdr10_plugin = []
                fdr_controlled_tpr10_plugin = []
                
                eval_enc_name = os.path.join(MODEL_DIR, "evalencoder_{}_{}.pt".format(encoder_type, mdl_name_prefix))
                print(eval_enc_name)
                if encoder_type == "defensive":
                    multi_encoder_keys = ["CUBO", "EUBO"]
                    encoder_eval_name = "defensive"
                    do_defensive_eval = True
                    eval_encoder = nn.ModuleDict(
                        {
                            key: Encoder(
                                n_input=N_GENES,
                                n_output=10,
                                n_layers=1,
                                n_hidden=N_HIDDEN,
                                dropout_rate=0.1,
                            )
                            for key in multi_encoder_keys
                        }
                    ).to("cuda")
                    counts_eval = COUNTS_EVAL
                    if os.path.exists(eval_enc_name):
                        print("eval enc mdl exists; loading from .pt")
                        eval_encoder.load_state_dict(torch.load(eval_enc_name))
                        eval_encoder.cuda()                    
                    else:
                        trainer.train_eval_defensive_encoder(
                            eval_encoder,
                            n_epochs=200,
                            lr=1e-3,
                            n_samples_phi=N_SAMPLES_PHI_EVAL,
                            counts=counts_eval,
                        )

                else:
                    multi_encoder_keys = ["default"]
                    encoder_eval_name = "default"
                    do_defensive_eval = False
                    eval_encoder = nn.ModuleDict(
                        {
                            "default": Encoder(
                                n_input=N_GENES,
                                n_output=10,
                                n_layers=1,
                                n_hidden=N_HIDDEN,
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
                            n_epochs=200,
                            lr=1e-3,
                            wake_psi=encoder_type,
                            n_samples_phi=n_samples_phi,
                            reparam=reparam,
                            # do_observed_library=DO_OBSERVED_LIBRARY,
                        )

                torch.save(eval_encoder.state_dict(), eval_enc_name)
                logging.info("Initialized and saved the evaluation encoder {}... defensive {}".format(eval_enc_name, do_defensive_eval))

                

                test_post = trainer.test_set.sequential()
                train_post = trainer.train_set.sequential()

                # *** CUBO 5000
                logging.info("CUBO 5K estimation...")
                multicounts_eval = None
                if counts_eval is not None:
                    multicounts_eval = (5000 / counts_eval.sum()) * counts_eval
                cubo5000_loss = (
                    test_post.getter(
                        keys=["CUBO"],
                        n_samples=5000,
                        batch_size=64,
                        do_observed_library=DO_OBSERVED_LIBRARY,
                        encoder_key=encoder_eval_name,
                        counts=multicounts_eval,
                        z_encoder=eval_encoder,
                    )["CUBO"]
                    .cpu()
                    .numpy()
                ).mean()
                cubo5000_arr.append(cubo5000_loss)

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
                        do_observed_library=DO_OBSERVED_LIBRARY,
                        encoder_key=encoder_eval_name,
                        counts=multicounts_eval,
                        z_encoder=eval_encoder,
                    )["IWELBO"]
                    .cpu()
                    .numpy()
                ).mean()
                iwelbo5000_arr.append(iwelbo5000_loss)

                # *** KHAT
                multicounts = None
                if counts_eval is not None:
                    multicounts = (5000 / counts_eval.sum()) * counts_eval
                log_ratios = []
                n_samples_total = 1e4
                n_samples_per_pass = (
                    25 if encoder_eval_name == "default" else multicounts.sum()
                )
                n_iter = int(n_samples_total / n_samples_per_pass)
                logging.info("Multicounts: {}".format(multicounts))
                logging.info("Khat computation using {} samples".format(n_samples_per_pass * n_iter))
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
                            do_observed_library=DO_OBSERVED_LIBRARY,
                            z_encoder=eval_encoder,
                        )
                    out = out["log_ratio"].cpu()
                    log_ratios.append(out)

                log_ratios = torch.cat(log_ratios)
                wi = torch.softmax(log_ratios, 0)
                ess_here = 1.0 / (wi ** 2).sum(0)

                _, khats = psislw(log_ratios.T.clone())

                khat_arr_10000.append(khats)
                ess.append(ess_here.numpy())

                logging.info("FDR/TPR ...")
                test_indices = train_post.indices
                y_test = Y[test_indices]

                IS_SIGNIFICANT_DE = (LFC_INFO["12"].abs() >= 0.5).values
                decision_rule_fdr10 = np.zeros(N_PICKS)
                decision_rule_tpr10 = np.zeros(N_PICKS)
                decision_rule_fdr10_plugin = np.zeros(N_PICKS)
                decision_rule_tpr10_plugin = np.zeros(N_PICKS)
                fdr_gt = np.zeros((N_GENES, N_PICKS))
                pe_fdr = np.zeros((N_GENES, N_PICKS))
                for ipick in range(N_PICKS):
                    samples_a = np.random.choice(np.where(y_test == 1)[0], size=10)
                    samples_b = np.random.choice(np.where(y_test == 2)[0], size=10)

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

                    true_fdr_arr = true_fdr(y_true=IS_SIGNIFICANT_DE, y_pred=y_pred_is)
                    pe_fdr_arr, y_decision_rule = posterior_expected_fdr(
                        y_pred=y_pred_is
                    )
                    # Fdr related
                    fdr_gt[:, ipick] = true_fdr_arr
                    pe_fdr[:, ipick] = pe_fdr_arr

                    _, y_decision_rule10 = posterior_expected_fdr(
                        y_pred=y_pred_is, fdr_target=0.1
                    )
                    decision_rule_fdr10[ipick] = 1 - precision_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule10
                    )
                    decision_rule_tpr10[ipick] = recall_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule10
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
                    true_fdr_arr = true_fdr(y_true=IS_SIGNIFICANT_DE, y_pred=y_pred_plugin)

                    _, y_decision_rule10 = posterior_expected_fdr(
                        y_pred=y_pred_plugin, fdr_target=0.1
                    )
                    decision_rule_fdr10_plugin[ipick] = 1 - precision_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule10
                    )
                    decision_rule_tpr10_plugin[ipick] = recall_score(
                        y_true=IS_SIGNIFICANT_DE, y_pred=y_decision_rule10
                    )
                    

                all_fdr_gt.append(fdr_gt)
                all_pe_fdr.append(pe_fdr)
                fdr_controlled_fdr10.append(decision_rule_fdr10)
                fdr_controlled_tpr10.append(decision_rule_tpr10)
                fdr_controlled_fdr10_plugin.append(decision_rule_fdr10_plugin)
                fdr_controlled_tpr10_plugin.append(decision_rule_tpr10_plugin)

                res = {
                    "wake_theta": loss_gen,
                    "wake_psi": loss_wvar,
                    "batch_size": batch_size,
                    "encoder_type": encoder_type,
                    "n_samples_theta": n_samples_theta,
                    "n_samples_phi": n_samples_phi,
                    "reparam_wphi": reparam_wphi,
                    "counts": counts,
                    "n_epochs": n_epochs,
                    "iaf_t": iaf_t,
                    "n_hidden": N_HIDDEN,
                    "n_latent": 10,
                    "n_layers": 1,
                    "cubo": cubo_arr,
                    "cubo5000": cubo5000_arr,
                    "iwelbo": iwelbo_arr,
                    "iwelbo5000": iwelbo5000_arr,
                    "khat_100": khat_arr_100,
                    "khat_25": khat_arr_25,
                    "khat_10000": np.array(khat_arr_10000),
                    "ess": np.array(ess),
                    "fdr_controlled_fdr10": np.array(fdr_controlled_fdr10),
                    "fdr_controlled_tpr10": np.array(fdr_controlled_tpr10),
                    "fdr_controlled_fdr10_plugin": np.array(fdr_controlled_fdr10_plugin),
                    "fdr_controlled_tpr10_plugin": np.array(fdr_controlled_tpr10_plugin),
                }

                print(res)
                DF_LI.append(res)
                DF = pd.DataFrame(DF_LI)
                DF.to_csv("{}.csv".format(FILENAME), sep="\t")
                DF.to_pickle("{}.pkl".format(FILENAME))
    DF = pd.DataFrame(DF_LI)
    DF.to_csv("{}.csv".format(FILENAME), sep="\t")
    DF.to_pickle("{}.pkl".format(FILENAME))
