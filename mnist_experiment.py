"""
    Decision theory: Experiment for M1+M1 model on MNIST
"""

import os

import numpy as np
import pandas as pd
import torch
from arviz.stats import psislw
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm

from sbvae.dataset import MnistDataset
from sbvae.inference import MnistTrainer
from sbvae.models import SemiSupervisedVAE

NUM = 300
N_EXPERIMENTS = 1
LABELLED_PROPORTIONS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
LABELLED_PROPORTIONS = LABELLED_PROPORTIONS / LABELLED_PROPORTIONS.sum()
LABELLED_FRACTION = 0.05
np.random.seed(42)
N_INPUT = 28 * 28
N_LABELS = 9

CLASSIFICATION_RATIO = 50.0
N_EVAL_SAMPLES = 25
N_EPOCHS = 100
LR = 3e-4
BATCH_SIZE = 512

FILENAME = "mnist_final.pkl"
MDL_DIR = "models/mnist_final"

if not os.path.exists(MDL_DIR):
    os.makedirs(MDL_DIR)

DATASET = MnistDataset(
    labelled_fraction=LABELLED_FRACTION,
    labelled_proportions=LABELLED_PROPORTIONS,
    root="sbvae_data/mnist",
    download=True,
    do_1d=True,
    test_size=0.5,
)

X_TRAIN, Y_TRAIN = DATASET.train_dataset.tensors
RDM_INDICES = np.random.choice(len(X_TRAIN), 200)
X_SAMPLE = X_TRAIN[RDM_INDICES].to("cuda")
Y_SAMPLE = Y_TRAIN[RDM_INDICES].to("cuda")

DO_OVERALL = True

print("train all examples", len(DATASET.train_dataset.tensors[0]))
print("train labelled examples", len(DATASET.train_dataset_labelled.tensors[0]))


SCENARIOS = [  # WAKE updates
    # High number of epochs
    # 100 epochs with more thetas
    dict(
        loss_gen="ELBO",
        loss_wvar="defensive",
        n_samples_train=None,
        n_samples_wtheta=1,
        n_samples_wphi=15,
        batch_norm=True,
        reparam_latent=None,
        z2_with_elbo=None,
        n_epochs=100,
        n_hidden=128,
        n_latent=10,
        lr=1e-3,
        counts=torch.tensor([1, 1, 0]),
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="CUBO",
        batch_norm=True,
        n_samples_train=None,
        n_samples_wtheta=1,
        n_samples_wphi=15,
        reparam_latent=True,
        z2_with_elbo=False,
        n_epochs=100,
        n_hidden=128,
        n_latent=10,
        lr=1e-3,
        counts=None,
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="REVKL",
        n_samples_train=None,
        n_samples_wtheta=1,
        n_samples_wphi=25,
        reparam_latent=False,
        z2_with_elbo=False,
        n_epochs=100,
        n_hidden=128,
        n_latent=10,
        lr=1e-3,
        counts=None,
    ),
    dict(
        loss_gen="ELBO",
        loss_wvar="ELBO",
        n_samples_train=1,
        n_samples_wtheta=1,
        n_samples_wphi=1,
        reparam_latent=True,
        z2_with_elbo=True,
        n_epochs=100,
        n_hidden=128,
        n_latent=10,
        lr=1e-4,
        counts=None,
    ),
]

DF_LI = []


# Utils functions
def compute_reject_score(y_true: np.ndarray, y_pred: np.ndarray, num=20):
    """
        Computes precision recall properties for the discovery label using
        Bayesian decision theory
    """
    _, n_pos_classes = y_pred.shape

    assert np.unique(y_true).max() == (n_pos_classes - 1) + 1
    thetas = np.linspace(0.1, 1.0, num=num)
    properties = dict(
        precision_discovery=np.zeros(num),
        recall_discovery=np.zeros(num),
        accuracy=np.zeros(num),
        thresholds=thetas,
    )

    for idx, theta in enumerate(thetas):
        y_pred_theta = y_pred.argmax(1)
        reject = y_pred.max(1) <= theta
        y_pred_theta[reject] = (n_pos_classes - 1) + 1

        properties["accuracy"][idx] = accuracy_score(y_true, y_pred_theta)

        y_true_discovery = y_true == (n_pos_classes - 1) + 1
        y_pred_discovery = y_pred_theta == (n_pos_classes - 1) + 1
        properties["precision_discovery"][idx] = precision_score(
            y_true_discovery, y_pred_discovery
        )
        properties["recall_discovery"][idx] = recall_score(
            y_true_discovery, y_pred_discovery
        )
    return properties


# Main script
for scenario in SCENARIOS:
    loss_gen = scenario["loss_gen"]
    loss_wvar = scenario["loss_wvar"]
    n_samples_train = scenario["n_samples_train"]
    n_samples_wtheta = scenario["n_samples_wtheta"]
    n_samples_wphi = scenario["n_samples_wphi"]
    reparam_latent = scenario["reparam_latent"]
    n_epochs = scenario["n_epochs"]
    n_latent = scenario["n_latent"]
    n_hidden = scenario["n_hidden"]
    lr = scenario["lr"]
    z2_with_elbo = scenario["z2_with_elbo"]
    counts = scenario["counts"]
    if "batch_norm" in scenario:
        batch_norm = scenario["batch_norm"]
    else:
        batch_norm = False
    # if counts is not None:
    #     counts = torch.tensor([0, 1, 0])
    if "cubo_z2_with_elbo" in scenario:
        cubo_z2_with_elbo = scenario["cubo_z2_with_elbo"]
    else:
        cubo_z2_with_elbo = False

    if "batch_size" in scenario:
        batch_size = scenario["batch_size"]
    else:
        batch_size = BATCH_SIZE

    # batch_size = scenario["batch_size"]

    iwelbo = []
    cubo = []
    khat = []
    khat1e4 = []
    m_accuracy_arr = []
    m_ap_arr = []
    m_recall_arr = []
    m_ap_arr_is = []
    m_recall_arr_is = []
    auc_pr_arr_is = []
    m_accuracy_arr_is = []
    auc_pr_arr = []
    entropy_arr = []

    do_defensive = loss_wvar == "defensive"
    multi_encoder_keys = ["CUBO", "EUBO"] if do_defensive else ["default"]
    encoder_inference_key = "defensive" if do_defensive else "default"
    for t in range(N_EXPERIMENTS):
        scenario["num"] = t
        mdl_name = ""
        for st in scenario.values():
            mdl_name = mdl_name + str(st) + "_"
        mdl_name = str(mdl_name)
        mdl_name = os.path.join(MDL_DIR, "{}.pt".format(mdl_name))
        print(mdl_name)
        mdl = SemiSupervisedVAE(
            n_input=N_INPUT,
            n_labels=N_LABELS,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=1,
            do_batch_norm=batch_norm,
            multi_encoder_keys=multi_encoder_keys,
        )
        if os.path.exists(mdl_name):
            print("model exists; loading from .pt")
            mdl.load_state_dict(torch.load(mdl_name))
        mdl.cuda()
        trainer = MnistTrainer(
            dataset=DATASET, model=mdl, use_cuda=True, batch_size=batch_size
        )

        overall_loss = None
        if (loss_gen == loss_wvar) and loss_gen == "ELBO":
            overall_loss = "ELBO"

        try:
            if not os.path.exists(mdl_name):
                if do_defensive:
                    trainer.train_defensive(
                        n_epochs=n_epochs,
                        lr=lr,
                        wake_theta=loss_gen,
                        cubo_wake_psi="CUBO",
                        n_samples_phi=n_samples_wphi,
                        n_samples_theta=n_samples_wtheta,
                        classification_ratio=CLASSIFICATION_RATIO,
                        update_mode="all",
                        counts=counts,
                        cubo_z2_with_elbo=cubo_z2_with_elbo,
                    )
                else:
                    trainer.train(
                        n_epochs=n_epochs,
                        lr=lr,
                        overall_loss=overall_loss,
                        wake_theta=loss_gen,
                        wake_psi=loss_wvar,
                        n_samples=n_samples_train,
                        n_samples_theta=n_samples_wtheta,
                        n_samples_phi=n_samples_wphi,
                        reparam_wphi=reparam_latent,
                        classification_ratio=CLASSIFICATION_RATIO,
                        z2_with_elbo=z2_with_elbo,
                        update_mode="all",
                    )
            torch.save(mdl.state_dict(), mdl_name)

            # TODO: find something cleaner
            if do_defensive:
                factor = N_EVAL_SAMPLES / counts.sum()
                multi_counts = factor * counts
            else:
                multi_counts = None

            # Eval
            with torch.no_grad():
                train_res = trainer.inference(
                    trainer.test_loader,
                    # trainer.train_loader,
                    keys=["qc_z1_all_probas", "y", "log_ratios", "qc_z1"],
                    n_samples=N_EVAL_SAMPLES,
                    encoder_key=encoder_inference_key,
                    counts=multi_counts,
                )
            y_pred = train_res["qc_z1_all_probas"].mean(0).numpy()
            y_pred = y_pred / y_pred.sum(1, keepdims=True)

            log_ratios = train_res["log_ratios"]  # n_cat, n_samples, n_batch
            weights = torch.softmax(log_ratios, dim=1)
            y_pred_is = (weights * train_res["qc_z1"]).sum(1).T.numpy()
            y_pred_is = y_pred_is / y_pred_is.sum(1, keepdims=True)
            assert y_pred.shape == y_pred_is.shape

            y_true = train_res["y"].numpy()

            # Precision / Recall for discovery class
            # And accuracy
            res_baseline = compute_reject_score(y_true=y_true, y_pred=y_pred, num=NUM)
            m_ap = res_baseline["precision_discovery"]
            m_recall = res_baseline["recall_discovery"]
            auc_pr = np.trapz(
                x=res_baseline["recall_discovery"],
                y=res_baseline["precision_discovery"],
            )
            m_ap_arr.append(m_ap)
            m_recall_arr.append(m_recall)
            auc_pr_arr.append(auc_pr)

            res_baseline_is = compute_reject_score(
                y_true=y_true, y_pred=y_pred_is, num=NUM
            )
            m_ap_is = res_baseline_is["precision_discovery"]
            m_recall_is = res_baseline_is["recall_discovery"]
            auc_pr_is = np.trapz(
                x=res_baseline_is["recall_discovery"],
                y=res_baseline_is["precision_discovery"],
            )
            m_ap_arr_is.append(m_ap_is)
            m_recall_arr_is.append(m_recall_is)
            auc_pr_arr_is.append(auc_pr_is)

            # Cubo / Iwelbo with 1e4 samples
            print("Heldout CUBO/IWELBO computation ...")

            n_samples_total = 1e4
            n_samples_per_pass = 25 if not do_defensive else multi_counts.sum()
            n_iter = int(n_samples_total / n_samples_per_pass)

            cubo_vals = []
            iwelbo_vals = []
            with torch.no_grad():
                i = 0
                for tensors in tqdm(trainer.test_loader):
                    x, _ = tensors
                    x = x.cuda()
                    log_ratios_batch = []
                    log_qc_batch = []
                    for _ in tqdm(range(n_iter)):
                        out = mdl.inference(
                            x,
                            n_samples=n_samples_per_pass,
                            encoder_key=encoder_inference_key,
                            counts=multi_counts,
                        )
                        if do_defensive:
                            log_ratio = out["log_ratio"].cpu()
                        else:
                            log_ratio = (
                                out["log_px_z"]
                                + out["log_pz2"]
                                + out["log_pc"]
                                + out["log_pz1_z2"]
                                - out["log_qz1_x"]
                                - out["log_qc_z1"]
                                - out["log_qz2_z1"]
                            ).cpu()
                        log_ratios_batch.append(log_ratio)
                        log_qc_batch.append(out["log_qc_z1"].cpu())

                    i += 1
                    if i == 2:
                        break
                    # Concatenation
                    log_ratios_batch = torch.cat(log_ratios_batch, dim=1)
                    log_qc_batch = torch.cat(log_qc_batch, dim=1)

                    # Lower bounds
                    # 1. Cubo
                    n_cat, n_samples, n_batch = log_ratios_batch.shape
                    cubo_val = torch.logsumexp(
                        (2 * log_ratios_batch + log_qc_batch).view(
                            n_cat * n_samples, n_batch
                        ),
                        dim=0,
                        keepdim=False,
                    ) - np.log(n_samples)

                    iwelbo_val = torch.logsumexp(
                        (log_ratios_batch + log_qc_batch).view(
                            n_cat * n_samples, n_batch
                        ),
                        dim=0,
                        keepdim=False,
                    ) - np.log(n_samples)

                    cubo_vals.append(cubo_val.cpu())
                    iwelbo_vals.append(iwelbo_val.cpu())
                cubo_vals = torch.cat(cubo_vals)
                iwelbo_vals = torch.cat(iwelbo_vals)
            cubo.append(cubo_vals.mean())
            iwelbo.append(iwelbo_vals.mean())

            # Entropy
            where9 = train_res["y"] == 9
            probas9 = train_res["qc_z1_all_probas"].mean(0)[where9]
            entropy_arr.append((-probas9 * probas9.log()).sum(-1).mean(0))

            where_non9 = train_res["y"] != 9
            y_non9 = train_res["y"][where_non9]
            y_pred_non9 = y_pred[where_non9].argmax(1)
            m_accuracy = accuracy_score(y_non9, y_pred_non9)
            m_accuracy_arr.append(m_accuracy)

            y_pred_non9_is = y_pred_is[where_non9].argmax(1)
            m_accuracy_is = accuracy_score(y_non9, y_pred_non9_is)
            m_accuracy_arr_is.append(m_accuracy_is)

            # k_hat
            n_samples_total = 1e4
            n_samples_per_pass = 25 if not do_defensive else multi_counts.sum()
            n_iter = int(n_samples_total / n_samples_per_pass)

            # a. Unsupervised case
            log_ratios = []
            qc_z = []
            for _ in tqdm(range(n_iter)):
                with torch.no_grad():
                    out = mdl.inference(
                        X_SAMPLE,
                        n_samples=n_samples_per_pass,
                        encoder_key=encoder_inference_key,
                        counts=multi_counts,
                    )
                if do_defensive:
                    log_ratio = out["log_ratio"].cpu()
                else:
                    log_ratio = (
                        out["log_px_z"]
                        + out["log_pz2"]
                        + out["log_pc"]
                        + out["log_pz1_z2"]
                        - out["log_qz1_x"]
                        - out["log_qc_z1"]
                        - out["log_qz2_z1"]
                    ).cpu()
                qc_z_here = out["log_qc_z1"].cpu().exp()
                qc_z.append(qc_z_here)
                log_ratios.append(log_ratio)
            # Concatenation over samples
            log_ratios = torch.cat(log_ratios, 1)
            qc_z = torch.cat(qc_z, 1)
            log_ratios_sum = (log_ratios * qc_z).sum(0)  # Sum over labels
            wi = torch.softmax(log_ratios_sum, 0)
            _, khats = psislw(log_ratios_sum.T.clone())
            # _, khats = psislw(log_ratios.view(-1, len(x_u)).numpy())
            khat1e4.append(khats)

        except Exception as e:
            raise e
            print(e)
            pass

    res = {
        "CONFIGURATION": scenario,
        "LOSS_GEN": loss_gen,
        "LOSS_WVAR": loss_wvar,
        "BATCH_SIZE": BATCH_SIZE,
        "N_SAMPLES_TRAIN": n_samples_train,
        "N_SAMPLES_WTHETA": n_samples_wtheta,
        "N_SAMPLES_WPHI": n_samples_wphi,
        "REPARAM_LATENT": reparam_latent,
        "N_LATENT": n_latent,
        "N_HIDDEN": n_hidden,
        "N_EPOCHS": n_epochs,
        "COUNTS": counts,
        "LR": lr,
        "Z2_WITH_ELBO": z2_with_elbo,
        "IWELBO": (np.mean(iwelbo), np.std(iwelbo)),
        "IWELBO_SAMPLES": np.array(iwelbo),
        "CUBO": (np.mean(cubo), np.std(cubo)),
        "CUBO_SAMPLES": np.array(cubo),
        "KHAT": np.array(khat),
        "KHAT1e4": np.array(khat1e4),
        "M_ACCURACY": np.array(m_accuracy_arr),
        "MEAN_AP": np.array(m_ap_arr),
        "MEAN_RECALL": np.array(m_recall_arr),
        "M_ACCURACY_IS": np.array(m_accuracy_arr_is),
        "MEAN_AP_IS": np.array(m_ap_arr_is),
        "MEAN_RECALL_IS": np.array(m_recall_arr_is),
        "AUC_IS": np.array(auc_pr_arr_is),
        "AUC": np.array(auc_pr_arr),
        "ENTROPY": np.array(entropy_arr),
    }
    print(res)
    DF_LI.append(res)
    DF = pd.DataFrame(DF_LI)
    DF.to_pickle(FILENAME)

DF = pd.DataFrame(DF_LI)
DF.to_pickle(FILENAME)
