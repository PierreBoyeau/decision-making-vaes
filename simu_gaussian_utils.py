import numpy as np
from scipy.linalg import sqrtm
from arviz.stats import psislw
from scipy.stats import norm

from dmvaes.dataset import SyntheticGaussianDataset


nus = np.geomspace(1e-2, 1e1, num=40)
DIM_Z = 6
DIM_X = 10
DATASET = SyntheticGaussianDataset(dim_z=DIM_Z, dim_x=DIM_X, n_samples=1000, nu=1)


def model_evaluation_loop(
    my_trainer, my_eval_encoder, my_counts_eval, my_encoder_eval_name,
):
    # posterior query evaluation: groundtruth
    seq = my_trainer.test_set.sequential(batch_size=10)
    mean = np.dot(DATASET.mz_cond_x_mean, DATASET.X[seq.indices, :].T)[0, :]
    std = np.sqrt(DATASET.pz_condx_var[0, 0])
    exact_cdf = norm.cdf(0, loc=mean, scale=std)

    is_cdf_nus = seq.prob_eval(
        1000,
        nu=nus,
        encoder_key=my_encoder_eval_name,
        counts=my_counts_eval,
        z_encoder=my_eval_encoder,
    )[2]
    plugin_cdf_nus = seq.prob_eval(
        1000,
        nu=nus,
        encoder_key=my_encoder_eval_name,
        counts=my_counts_eval,
        z_encoder=my_eval_encoder,
        plugin_estimator=True,
    )[2]
    exact_cdfs_nus = np.array([norm.cdf(nu, loc=mean, scale=std) for nu in nus]).T

    log_ratios = (
        my_trainer.test_set.log_ratios(
            n_samples_mc=5000,
            encoder_key=my_encoder_eval_name,
            counts=my_counts_eval,
            z_encoder=my_eval_encoder,
        )
        .detach()
        .numpy()
    )
    # Input should be n_obs, n_samples
    log_ratios = log_ratios.T
    _, khat_vals = psislw(log_ratios)

    # posterior query evaluation: aproposal distribution
    seq_mean, seq_var, is_cdf, ess = seq.prob_eval(
        1000,
        encoder_key=my_encoder_eval_name,
        counts=my_counts_eval,
        z_encoder=my_eval_encoder,
    )

    gt_post_var = DATASET.pz_condx_var
    sigma_sqrt = sqrtm(gt_post_var)
    a_2_it = np.zeros(len(seq_var))
    #  Check that generative model is not defensive to compute A
    if seq_var[0] is not None:
        for it in range(len(seq_var)):
            seq_var_item = seq_var[it]  # Posterior variance
            d_inv = np.diag(1.0 / seq_var_item)  # Variational posterior precision
            a = sigma_sqrt @ (d_inv @ sigma_sqrt) - np.eye(DIM_Z)
            a_2_it[it] = np.linalg.norm(a, ord=2)
    a_2_it = a_2_it.mean()

    return {
        "IWELBO": my_trainer.test_set.iwelbo(
            5000,
            encoder_key=my_encoder_eval_name,
            counts=my_counts_eval,
            z_encoder=my_eval_encoder,
        ),
        "L1_IS_ERRS": np.abs(is_cdf_nus - exact_cdfs_nus).mean(0),
        "L1_PLUGIN_ERRS": np.abs(plugin_cdf_nus - exact_cdfs_nus).mean(0),
        "KHAT": khat_vals,
        "exact_lls_test": my_trainer.test_set.exact_log_likelihood(),
        "exact_lls_train": my_trainer.train_set.exact_log_likelihood(),
        "model_lls_test": my_trainer.test_set.model_log_likelihood(),
        "model_lls_train": my_trainer.train_set.model_log_likelihood(),
        # "plugin_cdf": norm.cdf(0, loc=seq_mean[:, 0], scale=np.sqrt(seq_var[:, 0])),
        "l1_err_ex_is": np.mean(np.abs(exact_cdf - is_cdf)),
        "l2_ess": ess,
        "gt_post_var": DATASET.pz_condx_var,
        "a2_norm": a_2_it,
        # "sigma_sqrt": sqrtm(gt_post_var),
    }
