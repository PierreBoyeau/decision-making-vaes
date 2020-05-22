"""
    Decision theory: Experiment for pPCA experiment
"""

import os
import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ax import optimize
from tqdm.auto import tqdm

from sbvae.dataset import SyntheticGaussianDataset
from sbvae.inference import GaussianDefensiveTrainer
from sbvae.models import LinearGaussianDefensive
from sbvae.models.modules import Encoder, EncoderStudent
from simu_gaussian_utils import model_evaluation_loop, DATASET, DIM_Z, DIM_X


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)
FILENAME = "simu_cross_gaussian_def3"
# FILENAME = "deleteme"
n_simu = 5
n_epochs = 1000
LINEAR_ENCODER = False
MULTIMODAL_VAR_LANDSCAPE = True
N_SAMPLES_PHI = 50
N_SAMPLES_THETA = 50
LR = 1e-3


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


# TODO Adaptative M

EVAL_ENCODERS = [
    dict(
        encoder_type="train", eval_encoder_name="train"
    ),  # Variational distribution used to train the gen model
    dict(encoder_type=["ELBO"], reparam=True, eval_encoder_name="ELBO"),
    dict(encoder_type=["IWELBO"], reparam=True, eval_encoder_name="IWELBO"),
    dict(encoder_type=["REVKL"], reparam=True, eval_encoder_name="Forward KL"),
    # dict(encoder_type=["CUBO"], reparam=True, eval_encoder_name="$\\chi$"),
    dict(
        encoder_type=["CUBO"],
        reparam=True,
        do_student=True,
        student_df="learn",
        student_encs=["CUBO"],
        eval_encoder_name="$\\chi$ (St)",
    ),
    # dict(
    #     encoder_type=["IWELBO"],
    #     reparam=True,
    #     do_student=True,
    #     student_df="learn",
    #     student_encs=["IWELBO"],
    #     eval_encoder_name="IWELBO (St)",
    # ),
    dict(
        encoder_type=["IWELBO", "CUBO", "REVKL"],
        counts_eval=pd.Series(
            dict(
                IWELBO=N_SAMPLES_THETA // 4,
                CUBO=N_SAMPLES_THETA // 4,
                REVKL=N_SAMPLES_THETA // 4,
                prior=N_SAMPLES_THETA // 4,
            )
        ),
        reparam=True,
        student_encs=["CUBO"],
        eval_encoder_name="MsbVAE",
    ),
]
scenarios = [  # WAKE updates
    dict(learn_var=True, loss_gen="ELBO", losses_wvar=["ELBO"], model_name="VAE"),
    dict(learn_var=True, loss_gen="IWELBO", losses_wvar=["ELBO"], model_name="PIWAE"),
    dict(learn_var=True, loss_gen="IWELBO", losses_wvar=["IWELBO"], model_name="IWAE"),
    # dict(
    #     learn_var=True,
    #     loss_gen="IWELBO",
    #     losses_wvar=["IWELBO"],
    #     model_name="IWAE(ST)",
    #     do_student=True,
    #     student_df="learn",
    # ),
    dict(learn_var=True, loss_gen="IWELBO", losses_wvar=["REVKL"], model_name="WW"),
    dict(learn_var=True, loss_gen="IWELBO", losses_wvar=["CUBO"], model_name="$\\chi$"),
    dict(
        learn_var=True,
        loss_gen="IWELBO",
        losses_wvar=["IWELBO", "CUBO", "REVKL"],
        counts=pd.Series(
            dict(
                IWELBO=N_SAMPLES_THETA // 4,
                CUBO=N_SAMPLES_THETA // 4,
                REVKL=N_SAMPLES_THETA // 4,
                prior=N_SAMPLES_THETA // 4,
            )
        ),
        model_name="M-sbVAE",
        student_encs=["CUBO"],
    ),
]

# nus = np.geomspace(1e-4, 1e2, num=20)
nus = np.geomspace(1e-2, 1e1, num=40)

# n_hidden_ranges = [16, 32, 64, 128, 256, 512]
n_hidden_ranges = [128]

df = []
for dic in scenarios:
    print(dic)
    learn_var = dic.get("learn_var", None)
    loss_gen = dic.get("loss_gen", None)
    losses_wvar = dic.get("losses_wvar", None)
    do_student = dic.get("do_student", False)
    student_df = dic.get("student_df", None)
    lr = dic.get("lr", LR)
    n_samples_phi = dic.get("n_samples_phi", N_SAMPLES_PHI)
    n_samples_theta = dic.get("n_samples_theta", N_SAMPLES_THETA)
    counts = dic.get("counts", None)
    model_name = dic.get("model_name", None)
    do_linear_encoder = dic.get("do_linear_encoder", LINEAR_ENCODER)
    for n_hidden in tqdm(n_hidden_ranges):
        logging.info("{} {} {}".format(learn_var, loss_gen, losses_wvar))
        for t in tqdm(range(n_simu)):
            print(t)
            params_gen = None
            params_wvar = None
            learn_var = False

            if loss_gen is not None:
                learn_var = True

            model = LinearGaussianDefensive(
                DATASET.A,
                DATASET.pxz_log_det,
                DATASET.pxz_inv_sqrt,
                gamma=DATASET.gamma,
                n_latent=DIM_Z,
                n_input=DIM_X,
                learn_gen=False,
                do_student=do_student,
                student_df=student_df,
                multimodal_var_landscape=MULTIMODAL_VAR_LANDSCAPE,
                learn_var=learn_var,
                linear_encoder=do_linear_encoder,
                n_hidden=n_hidden,
                multi_encoder_keys=losses_wvar,
            )

            trainer = GaussianDefensiveTrainer(
                model, DATASET, train_size=0.8, use_cuda=True, frequency=5
            )

            if loss_gen is not None:
                params_gen = [model._px_log_diag_var]

            params_wvar = {
                key: filter(lambda p: p.requires_grad, model.encoder[key].parameters())
                for key in model.encoder
            }

            losses = loss_gen, losses_wvar, None
            params = params_gen, params_wvar, None

            trainer.train_all_cases(
                params,
                losses,
                n_epochs=n_epochs,
                counts=counts,
                n_samples_phi=n_samples_phi,
            )

            for eval_dic in EVAL_ENCODERS:
                print(eval_dic)
                encoder_type = eval_dic.get("encoder_type", None)
                reparam = eval_dic.get("reparam", None)
                counts_eval = eval_dic.get("counts_eval", None)
                eval_encoder_name = eval_dic.get("eval_encoder_name", None)
                optim_mixture = eval_dic.get("optim_mixture", False)
                # do_student_eval = eval_dic.get("do_student", False)
                # student_df_eval = eval_dic.get("student_df", None)
                student_encs = eval_dic.get("student_encs", [])
                setup_loop = {
                    "CONFIGURATION": (learn_var, loss_gen, losses_wvar),
                    "eval_encoder_name": eval_encoder_name,
                    "optim_mixture": optim_mixture,
                    "do_student": do_student,
                    "student_df": student_df,
                    "multi_counts_eval": None,
                    "gamma": DATASET.gamma,
                    "model_name": model_name,
                    "sigma": model.px_log_diag_var.detach().cpu().numpy(),
                    "learn_var": learn_var,
                    "lr": lr,
                    "experiment": t,
                    "counts": counts,
                    "counts_eval": counts_eval,
                    "n_epochs": n_epochs,
                    "loss_gen": loss_gen,
                    "loss_wvar": losses_wvar,
                    "n_samples_phi": n_samples_phi,
                    "n_samples_theta": n_samples_theta,
                    "multimodal_var_landscape": MULTIMODAL_VAR_LANDSCAPE,
                    "n_hidden": n_hidden,
                    "encoder_type": encoder_type,
                    "student_encs": student_encs,
                    # "do_student_eval": do_student_eval,
                    # "student_df_eval": student_df_eval,
                }

                if encoder_type == "train":
                    logging.info(
                        "Using train variational distribution for evaluation ..."
                    )
                    eval_encoder = None
                    multi_counts_eval = None
                    if counts is not None:
                        multi_counts_eval = ((5000 / counts.sum()) * counts).astype(int)
                    encoder_eval_name = losses_wvar
                else:
                    logging.info(
                        "Training eval variational distribution for evaluation with {} ...".format(
                            encoder_type
                        )
                    )

                    modules = dict()
                    for enc_key in encoder_type:
                        if enc_key in student_encs:
                            modules[enc_key] = EncoderStudent(
                                n_input=DIM_X,
                                n_output=DIM_Z,
                                df="learn",
                                n_layers=1,
                                n_hidden=n_hidden,
                                dropout_rate=0.1,
                            ).cuda()
                        else:
                            modules[enc_key] = Encoder(
                                n_input=DIM_X,
                                n_output=DIM_Z,
                                n_layers=1,
                                n_hidden=n_hidden,
                                dropout_rate=0.1,
                            ).cuda()
                    eval_encoder = nn.ModuleDict(modules)
                    params_wvar = {
                        key: filter(
                            lambda p: p.requires_grad, eval_encoder[key].parameters()
                        )
                        for key in eval_encoder
                    }

                    losses_eval = None, encoder_type, None
                    params_eval = None, params_wvar, None
                    logging.info("training {}".format(encoder_type))
                    encoder_eval_name = encoder_type
                    trainer.train_all_cases(
                        params_eval,
                        losses_eval,
                        n_epochs=n_epochs,
                        counts=counts_eval,
                        n_samples_phi=n_samples_phi,
                        z_encoder=eval_encoder,
                    )

                    # Evalulation procedure
                    multi_counts_eval = None
                    if counts_eval is not None:
                        multi_counts_eval = (
                            (5000 / counts_eval.sum()) * counts_eval
                        ).astype(int)

                logging.info("Evaluation performance ...")
                if optim_mixture:
                    # In this setup the mixture weights are learned using
                    # Bayesian optimization
                    assert multi_counts_eval is not None

                    def evaluation_fn(parameters):
                        cts = pd.Series(parameters)
                        cts_ser = (5000 * softmax(cts)).astype(int)
                        print(cts_ser)
                        cts_ser += 1
                        with torch.no_grad():
                            res_eval_lp = model_evaluation_loop(
                                trainer=trainer,
                                eval_encoder=eval_encoder,
                                counts_eval=cts_ser,
                                encoder_eval_name=encoder_type,
                            )
                        print(np.median(res_eval_lp["KHAT"]))
                        res = np.median(res_eval_lp["KHAT"])
                        if np.isinf(res):
                            res = 5.0
                        return res

                    parameters = [
                        dict(name=key, type="range", bounds=[-10.0, 10.0])
                        for key in multi_counts_eval.keys()
                    ]
                    print(parameters)
                    print(eval_encoder.keys())
                    best_parameters, best_values, experiment, model = optimize(
                        parameters=parameters,
                        evaluation_function=evaluation_fn,
                        minimize=True,
                        total_trials=25,
                    )
                    multi_counts_eval = (
                        5000 * softmax(pd.Series(best_parameters))
                    ).astype(int)
                    multi_counts_eval += 1

                # Computing model results
                res_eval_loop = model_evaluation_loop(
                    trainer=trainer,
                    eval_encoder=eval_encoder,
                    counts_eval=multi_counts_eval,
                    encoder_eval_name=encoder_eval_name,
                )
                print(res_eval_loop)
                res = {
                    "custom_metrics": trainer.custom_metrics,
                    **setup_loop,
                    **res_eval_loop,
                }
                df.append(res)

                df_res = pd.DataFrame(df)
                df_res.to_csv("{}.csv".format(FILENAME), sep="\t")
                df_res.to_pickle("{}.pkl".format(FILENAME))
df_res = pd.DataFrame(df)
df_res.to_csv("{}.csv".format(FILENAME), sep="\t")
df_res.to_pickle("{}.pkl".format(FILENAME))
