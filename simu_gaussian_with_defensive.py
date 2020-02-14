"""
    Decision theory: Experiment for pPCA experiment
"""


import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from arviz.stats import psislw
from scipy.linalg import sqrtm
from scipy.stats import norm
from tqdm.auto import tqdm

from sbvae.dataset import SyntheticGaussianDataset
from sbvae.inference import GaussianDefensiveTrainer
from sbvae.models import LinearGaussianDefensive

FILENAME = "simu_gaussian_res_paper_def"
print("STARTED TRAINING", flush=True)
n_simu = 10

dim_z = 6
dim_x = 10
n_epochs = 100
dataset = SyntheticGaussianDataset(dim_z=dim_z, dim_x=dim_x, n_samples=1000, nu=1)
# plt.imshow(dataset.pz_condx_var)
# plt.colorbar()
# plt.savefig("figures/post_covariance.png")
# plt.clf()

LINEAR_ENCODER = False

TIME_100_SAMPLES = []
TIME_1000_SAMPLES = []

# learn_var, wake only, sleep only, wake-sleep, linear_encoder for each loss
scenarios = [  # WAKE updates
    (False, None, "defensive", torch.tensor([10, 10, 0]), LINEAR_ENCODER),
    (False, None, "ELBO", None, LINEAR_ENCODER),
    (False, None, "REVKL", None, LINEAR_ENCODER),
    (False, None, "CUBO", None, LINEAR_ENCODER),
]

# nus = np.geomspace(1e-4, 1e2, num=20)
nus = np.geomspace(1e-2, 1e1, num=40)

# n_hidden_ranges = [16, 32, 64, 128, 256, 512]
n_hidden_ranges = [128]

df = []
for learn_var, loss_gen, loss_wvar, counts, do_linear_encoder in scenarios:
    do_defensive = loss_wvar == "defensive"
    multi_encoder_keys = ["CUBO", "EUBO"] if do_defensive else ["default"]
    encoder_key = "defensive" if do_defensive else "default"
    for n_hidden in tqdm(n_hidden_ranges):
        print(learn_var, loss_gen, loss_wvar)
        iwelbo = []
        cubo = []
        l1_gen_dis = []
        l1_gen_sign = []
        l1_post_dis = []
        l1_post_sign = []
        l1_err_ex_plugin = []
        l1_err_ex_is = []
        l2_ess = []
        l1_errs_is = []
        khat = []
        a_2 = []
        for t in tqdm(range(n_simu)):
            print(t)
            params_gen = None
            params_wvar = None
            params_wvar_bis = None
            learn_var = False

            if loss_gen is not None:
                learn_var = True

            model = LinearGaussianDefensive(
                dataset.A,
                dataset.pxz_log_det,
                dataset.pxz_inv_sqrt,
                gamma=dataset.gamma,
                n_latent=dim_z,
                n_input=dim_x,
                learn_var=learn_var,
                linear_encoder=do_linear_encoder,
                n_hidden=n_hidden,
                multi_encoder_keys=multi_encoder_keys,
            )

            trainer = GaussianDefensiveTrainer(
                model, dataset, train_size=0.8, use_cuda=True, frequency=5
            )

            if loss_gen is not None:
                params_gen = [model.px_log_diag_var]
            if do_defensive:
                params_wvar = filter(
                    lambda p: p.requires_grad, model.encoder["CUBO"].parameters()
                )
                params_wvar_bis = filter(
                    lambda p: p.requires_grad, model.encoder["EUBO"].parameters()
                )
            else:
                params_wvar = filter(
                    lambda p: p.requires_grad, model.encoder["default"].parameters()
                )

            losses = loss_gen, loss_wvar, None
            params = params_gen, params_wvar, params_wvar_bis
            if do_defensive:
                trainer.train_defensive(
                    params, losses, n_epochs=n_epochs, counts=counts
                )
            else:
                trainer.train(params, losses, n_epochs=n_epochs)

            ll_train_set = trainer.history["elbo_train_set"][1:]
            ll_test_set = trainer.history["elbo_test_set"][1:]
            x = np.linspace(0, n_epochs, len(ll_train_set))
            # plt.plot(x, ll_train_set)
            # plt.plot(x, ll_test_set)
            # plt.savefig("figures/training_stats.png")
            # plt.clf()

            # trainer.test_set.elbo()
            multi_counts = None
            if do_defensive:
                multi_counts = (100 / counts.sum()) * counts
            iwelbo += [
                trainer.test_set.iwelbo(
                    1000, encoder_key=encoder_key, counts=multi_counts
                )
            ]
            # trainer.test_set.exact_log_likelihood()

            start = time.time()
            cubo += [
                trainer.test_set.cubo(
                    1000, encoder_key=encoder_key, counts=multi_counts
                )
            ]
            TIME_1000_SAMPLES.append(time.time() - start)

            start = time.time()
            trainer.test_set.cubo(100, encoder_key=encoder_key, counts=multi_counts)
            TIME_100_SAMPLES.append(time.time() - start)

            # posterior query evaluation: groundtruth
            seq = trainer.test_set.sequential(batch_size=10)
            mean = np.dot(dataset.mz_cond_x_mean, dataset.X[seq.indices, :].T)[0, :]
            std = np.sqrt(dataset.pz_condx_var[0, 0])
            exact_cdf = norm.cdf(0, loc=mean, scale=std)

            # IS L1_err comparison
            if do_defensive:
                multi_counts = (1000 / counts.sum()) * counts
            is_cdf_nus = seq.prob_eval(
                1000, nu=nus, encoder_key=encoder_key, counts=multi_counts
            )[2]
            exact_cdfs_nus = np.array(
                [norm.cdf(nu, loc=mean, scale=std) for nu in nus]
            ).T
            l1_errs_is += [np.abs(is_cdf_nus - exact_cdfs_nus).mean(0)]

            # k_hat
            if do_defensive:
                multi_counts = (100 / counts.sum()) * counts
            log_ratios = (
                trainer.test_set.log_ratios(
                    n_samples_mc=100, encoder_key=encoder_key, counts=multi_counts
                )
                .detach()
                .numpy()
            )
            # Input should be n_obs, n_samples
            log_ratios = log_ratios.T
            _, khat_vals = psislw(log_ratios)
            khat.append(khat_vals)

            # posterior query evaluation: aproposal distribution
            seq_mean, seq_var, is_cdf, ess = seq.prob_eval(
                1000, encoder_key=encoder_key, counts=multi_counts
            )
            plugin_cdf = norm.cdf(0, loc=seq_mean[:, 0], scale=np.sqrt(seq_var[:, 0]))

            l1_err_ex_plugin += [np.mean(np.abs(exact_cdf - plugin_cdf))]
            l1_err_ex_is += [np.mean(np.abs(exact_cdf - is_cdf))]
            l2_ess += [ess]

            # a norm
            # model.eval()
            gt_post_var = dataset.pz_condx_var
            sigma_sqrt = sqrtm(gt_post_var)

            a_2_it = np.zeros(len(seq_var))
            for it in range(len(seq_var)):
                seq_var_item = seq_var[it]  # Posterior variance
                d_inv = np.diag(1.0 / seq_var_item)  # Variationnal posterior precision
                a = sigma_sqrt @ (d_inv @ sigma_sqrt) - np.eye(dim_z)
                a_2_it[it] = np.linalg.norm(a, ord=2)
            a_2_it = a_2_it.mean()
            a_2.append(a_2_it)

        res = {
            "CONFIGURATION": (learn_var, loss_gen, loss_wvar),
            "learn_var": learn_var,
            "loss_gen": loss_gen,
            "loss_wvar": loss_wvar,
            "n_hidden": n_hidden,
            "IWELBO": (np.mean(iwelbo), np.std(iwelbo)),
            "CUBO": (np.mean(cubo), np.std(cubo)),
            "L1 loss gen_variance_dis": (np.mean(l1_gen_dis), np.std(l1_gen_dis)),
            "L1 loss gen_variance_sign": (np.mean(l1_gen_sign), np.std(l1_gen_sign)),
            "L1 loss post_variance_dis": (np.mean(l1_post_dis), np.std(l1_post_dis)),
            "L1 loss post_variance_sign": (np.mean(l1_post_sign), np.std(l1_post_sign)),
            "AVE L1 ERROR EXACT <-> PLUGIN": (
                np.mean(l1_err_ex_plugin),
                np.std(l1_err_ex_plugin),
            ),
            "ESS": (np.mean(l2_ess), np.std(l2_ess)),
            "L1_IS_ERRS": np.array(l1_errs_is),
            "KHAT": np.array(khat),
            "A_2": np.array(a_2),
        }
        df.append(res)

df = pd.DataFrame(df)
df.to_csv("{}.csv".format(FILENAME), sep="\t")
df.to_pickle("{}.pkl".format(FILENAME))
