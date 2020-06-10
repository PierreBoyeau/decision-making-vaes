import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from tqdm import trange

from . import Posterior, Trainer

plt.switch_backend("agg")


class GaussianDefensiveTrainer(Trainer):
    r"""UnsupervisedTrainer but for Gaussian datasets. Also implements the wake-sleep methods

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :dataset: A gaussian_dataset instance``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.
    """
    default_metrics_to_monitor = ["ll"]

    def __init__(self, model, dataset, train_size=0.8, test_size=None, **kwargs):
        super().__init__(model, dataset, **kwargs)
        if type(self) is GaussianDefensiveTrainer:
            self.train_set, self.test_set = self.train_test(
                model,
                dataset,
                train_size,
                test_size,
                type_class=GaussianDefensivePosterior,
            )
            self.train_set.to_monitor = ["elbo"]
            self.test_set.to_monitor = ["elbo"]

    def train(
        self,
        params,
        losses,
        n_epochs=20,
        lr=1e-3,
        eps=0.01,
        z_encoder: nn.Module = None,
        n_samples_phi: int = None,
        n_samples_theta: int = 1,
    ):
        params_gen, params_wvar, params_svar = params
        loss_gen, loss_wvar, loss_svar = losses
        begin = time.time()
        self.model.train()
        self.custom_metrics = dict(
            sgm_norm=[], a_err_norm=[], mdl_ll_train=[], mdl_ll_test=[]
        )
        print("using n_samples_phi", n_samples_phi)
        print("train with custom encoder {}".format(z_encoder is not None))

        optimizers = [0, 0, 0]

        if params_gen is not None:
            print("WAKE UPDATE GENERATIVE MODEL")
            optimizers[0] = torch.optim.Adam(params_gen, lr=lr, eps=eps)
        if params_wvar is not None:
            print("WAKE UPDATE VAR MODEL")
            optimizers[1] = torch.optim.Adam(params_wvar, lr=lr, eps=eps)
        if params_svar is not None:
            print("SLEEP UPDATE VAR MODEL")
            optimizers[2] = torch.optim.Adam(params_svar, lr=lr, eps=eps)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        # self.compute_metrics()

        with trange(n_epochs, desc="training", file=sys.stderr, disable=True) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                if params_gen is not None:
                    # WAKE PHASE for generative model
                    for tensors_list in self.data_loaders_loop():
                        data_tensor = torch.stack(*tensors_list, 0)
                        loss = torch.mean(
                            self.model(
                                data_tensor,
                                loss_gen,
                                z_encoder=z_encoder,
                                n_samples_mc=n_samples_theta,
                            )
                        )
                        optimizers[0].zero_grad()
                        loss.backward()
                        optimizers[0].step()

                if params_wvar is not None:
                    # WAKE PHASE for variational model
                    if loss_wvar == "REVKL+CUBO":
                        if self.epoch <= int(n_epochs / 3):
                            loss_wvar_epoch = "REVKL"
                        else:
                            loss_wvar_epoch = "CUBO"
                    elif loss_wvar == "ELBO+CUBO":
                        if self.epoch <= int(n_epochs / 3):
                            loss_wvar_epoch = "ELBO"
                        else:
                            loss_wvar_epoch = "CUBO"
                    else:
                        loss_wvar_epoch = loss_wvar

                    for tensors_list in self.data_loaders_loop():
                        data_tensor = torch.stack(*tensors_list, 0)
                        loss = torch.mean(
                            self.model(
                                data_tensor,
                                loss_wvar_epoch,
                                z_encoder=z_encoder,
                                n_samples_mc=n_samples_phi,
                            )
                        )
                        optimizers[1].zero_grad()
                        loss.backward()
                        optimizers[1].step()

                if params_svar is not None:
                    # SLEEP PHASE for variational model
                    # use loss loss_svar
                    # do not loop through real data but through simulated data
                    for tensors_list in self.data_loaders_loop():
                        # ignore the data
                        x, z = self.model.generate_prior_data()
                        loss = torch.mean(
                            self.model((x, z), loss_svar, z_encoder=z_encoder)
                        )
                        optimizers[2].zero_grad()
                        loss.backward()
                        optimizers[2].step()

                if params_gen is not None:
                    # sgm_norm
                    # a_err_norm
                    sgm_err_mat = (
                        np.diag(
                            self.model.px_log_diag_var.exp().detach().cpu().squeeze()
                        )
                        - self.gene_dataset.gamma
                    )
                    self.custom_metrics["sgm_norm"].append(
                        np.linalg.norm(sgm_err_mat, ord=2)
                    )
                    a_err_mat = self.model.A.detach().cpu() - self.gene_dataset.A
                    self.custom_metrics["a_err_norm"].append(
                        np.linalg.norm(a_err_mat, ord=2)
                    )
                    self.custom_metrics["mdl_ll_train"].append(
                        self.train_set.model_log_likelihood()
                    )
                    self.custom_metrics["mdl_ll_test"].append(
                        self.test_set.model_log_likelihood()
                    )

                # if not self.on_epoch_end():
                #     break

        if self.early_stopping.save_best_state_metric is not None:
            self.model.load_state_dict(self.best_state_dict)
            # self.compute_metrics()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.verbose and self.frequency:
            print(
                "\nTraining time:  %i s. / %i epochs"
                % (int(self.training_time), self.n_epochs)
            )

    def train_defensive(
        self,
        params,
        losses,
        n_epochs=20,
        lr=1e-3,
        eps=0.01,
        counts=None,
        z_encoder: nn.Module = None,
        n_samples_phi: int = None,
    ):
        print("train with custom encoder {}".format(z_encoder is not None))
        params_gen, params_w_cubo_var, params_w_eubo_var = params
        loss_gen, _, _ = losses
        begin = time.time()
        self.model.train()
        self.custom_metrics = dict(
            sgm_norm=[], a_err_norm=[], mdl_ll_train=[], mdl_ll_test=[]
        )
        optimizers = [0, 0, 0]

        if params_gen is not None:
            print("WAKE UPDATE GENERATIVE MODEL")
            optimizers[0] = torch.optim.Adam(params_gen, lr=lr, eps=eps)
        print("WAKE CUBO UPDATE VAR MODEL")
        optimizers[1] = torch.optim.Adam(params_w_cubo_var, lr=lr, eps=eps)
        print("WAKE EUBO UPDATE VAR MODEL")
        optimizers[2] = torch.optim.Adam(params_w_eubo_var, lr=lr, eps=eps)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        # self.compute_metrics()

        with trange(n_epochs, desc="training", file=sys.stderr, disable=True) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                if params_gen is not None:
                    # WAKE PHASE for generative model
                    for tensors_list in self.data_loaders_loop():
                        data_tensor = torch.stack(*tensors_list, 0)
                        loss = torch.mean(
                            self.model(
                                data_tensor,
                                loss_gen,
                                encoder_key="defensive",
                                counts=counts,
                                z_encoder=z_encoder,
                            )
                        )
                        optimizers[0].zero_grad()
                        loss.backward()
                        optimizers[0].step()

                for tensors_list in self.data_loaders_loop():
                    data_tensor = torch.stack(*tensors_list, 0)
                    # CUBO wake update
                    loss = torch.mean(
                        self.model(
                            data_tensor,
                            "CUBO",
                            encoder_key="CUBO",
                            z_encoder=z_encoder,
                            n_samples_mc=n_samples_phi,
                        )
                    )
                    optimizers[1].zero_grad()
                    loss.backward()
                    optimizers[1].step()

                    # EUBO wake update
                    loss = torch.mean(
                        self.model(
                            data_tensor,
                            "REVKL",
                            encoder_key="EUBO",
                            z_encoder=z_encoder,
                            n_samples_mc=n_samples_phi,
                        )
                    )
                    optimizers[2].zero_grad()
                    loss.backward()
                    optimizers[2].step()

                # if not self.on_epoch_end():
                #     break

                if params_gen is not None:
                    # sgm_norm
                    # a_err_norm
                    sgm_err_mat = (
                        np.diag(
                            self.model.px_log_diag_var.exp().detach().cpu().squeeze()
                        )
                        - self.gene_dataset.gamma
                    )
                    self.custom_metrics["sgm_norm"].append(
                        np.linalg.norm(sgm_err_mat, ord=2)
                    )
                    a_err_mat = self.model.A.detach().cpu() - self.gene_dataset.A
                    self.custom_metrics["a_err_norm"].append(
                        np.linalg.norm(a_err_mat, ord=2)
                    )
                    self.custom_metrics["mdl_ll_train"].append(
                        self.train_set.model_log_likelihood()
                    )
                    self.custom_metrics["mdl_ll_test"].append(
                        self.test_set.model_log_likelihood()
                    )

        if self.early_stopping.save_best_state_metric is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.compute_metrics()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.verbose and self.frequency:
            print(
                "\nTraining time:  %i s. / %i epochs"
                % (int(self.training_time), self.n_epochs)
            )

    def train_all_cases(
        self,
        params,
        losses,
        n_epochs=20,
        lr=1e-3,
        eps=0.01,
        z_encoder: nn.Module = None,
        counts=None,
        n_samples_theta: int = None,
        n_samples_phi: int = None,
    ):
        print("train with custom encoder {}".format(z_encoder is not None))
        my_params_gen, my_params_wvar, _ = params
        my_loss_gen, my_losses_wvar, _ = losses
        begin = time.time()
        self.model.train()
        self.custom_metrics = dict(
            sgm_norm=[], a_err_norm=[], mdl_ll_train=[], mdl_ll_test=[]
        )
        optimizers = dict()

        if my_params_gen is not None:
            print("WAKE UPDATE GENERATIVE MODEL")
            optimizers["theta"] = torch.optim.Adam(my_params_gen, lr=lr, eps=eps)
        print("TRAIN WITH {}".format(my_losses_wvar))
        for enc_loss_key in my_losses_wvar:
            optimizers[enc_loss_key] = torch.optim.Adam(
                my_params_wvar[enc_loss_key], lr=lr, eps=eps
            )

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        # self.compute_metrics()

        with trange(n_epochs, desc="training", file=sys.stderr, disable=True) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                # if my_params_gen is not None:
                #     # WAKE PHASE for generative model
                #     for tensors_list in self.data_loaders_loop():
                #         data_tensor = torch.stack(*tensors_list, 0)
                #         loss = torch.mean(
                #             self.model(
                #                 data_tensor,
                #                 my_loss_gen,
                #                 encoder_key=my_losses_wvar,
                #                 n_samples_mc=n_samples_theta,
                #                 counts=counts,
                #                 z_encoder=z_encoder,
                #             )
                #         )
                #         optimizers["theta"].zero_grad()
                #         loss.backward()
                #         optimizers["theta"].step()

                for tensors_list in self.data_loaders_loop():
                    data_tensor = torch.stack(*tensors_list, 0)
                    if my_params_gen is not None:
                        loss = torch.mean(
                            self.model(
                                data_tensor,
                                my_loss_gen,
                                encoder_key=my_losses_wvar,
                                n_samples_mc=n_samples_theta,
                                counts=counts,
                                z_encoder=z_encoder,
                            )
                        )
                        optimizers["theta"].zero_grad()
                        loss.backward()
                        optimizers["theta"].step()

                    for enc_loss_key in my_losses_wvar:
                        loss_enc = torch.mean(
                            self.model(
                                data_tensor,
                                enc_loss_key,
                                encoder_key=enc_loss_key,
                                z_encoder=z_encoder,
                                n_samples_mc=n_samples_phi,
                            )
                        )
                        optimizers[enc_loss_key].zero_grad()
                        loss_enc.backward()
                        optimizers[enc_loss_key].step()

                # if not self.on_epoch_end():
                #     break

                if my_params_gen is not None:
                    sgm_err_mat = (
                        np.diag(self.model.px_log_diag_var.exp().detach().cpu().squeeze())
                        - self.gene_dataset.gamma
                    )
                    self.custom_metrics["sgm_norm"].append(
                        np.linalg.norm(sgm_err_mat, ord=2)
                    )
                    a_err_mat = self.model.A.detach().cpu() - self.gene_dataset.A
                    self.custom_metrics["a_err_norm"].append(
                        np.linalg.norm(a_err_mat, ord=2)
                    )
                    self.custom_metrics["mdl_ll_train"].append(
                        self.train_set.model_log_likelihood()
                    )
                    self.custom_metrics["mdl_ll_test"].append(
                        self.test_set.model_log_likelihood()
                    )

        if self.early_stopping.save_best_state_metric is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.compute_metrics()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.verbose and self.frequency:
            print(
                "\nTraining time:  %i s. / %i epochs"
                % (int(self.training_time), self.n_epochs)
            )

    @property
    def posteriors_loop(self):
        return ["train_set"]


class GaussianDefensivePosterior(Posterior):
    @torch.no_grad()
    def elbo(self, verbose=False, encoder_key="default", counts=None):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.neg_elbo(data_tensor)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = -log_lkl / n_samples
        if verbose:
            print("ELBO : %.4f" % ll)
        return ll

    @torch.no_grad()
    def exact_log_likelihood(self, verbose=False):
        mean = self.gene_dataset.px_mean
        cov = self.gene_dataset.px_var
        data = self.gene_dataset.X
        ll = multivariate_normal.logpdf(data, mean=mean, cov=cov).mean()
        if verbose:
            print("log p*(x) : %.4f" % ll)
        return ll

    @torch.no_grad()
    def model_log_likelihood(self, verbose=False):
        sigma_z_x = np.diag(self.model.px_log_diag_var.exp().squeeze().detach().cpu())
        w = self.model.A.detach().cpu().numpy()
        X = self.gene_dataset.X[self.indices]
        cov = sigma_z_x + w @ w.T
        mean = np.zeros(X.shape[1])
        try:
            ll = multivariate_normal.logpdf(X, mean=mean, cov=cov).mean()
        except np.linalg.LinAlgError:
            raise ValueError(sigma_z_x)
        except ValueError:
            raise ValueError(sigma_z_x, w)
        return ll

    @torch.no_grad()
    def iwelbo(
        self,
        n_samples_mc,
        verbose=False,
        encoder_key="default",
        counts=None,
        z_encoder: nn.Module = None,
    ):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.neg_iwelbo(
                data_tensor,
                n_samples_mc,
                encoder_key=encoder_key,
                counts=counts,
                z_encoder=z_encoder,
            )
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = -log_lkl / n_samples
        if verbose:
            print("IWELBO", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def cubo(self, n_samples_mc, verbose=False, encoder_key="default", counts=None):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.cubo(
                data_tensor, n_samples_mc, encoder_key=encoder_key, counts=counts
            )
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = log_lkl / n_samples
        if verbose:
            print("CUBO", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def vr_max(self, n_samples_mc, verbose=False, encoder_key="default", counts=None):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.vr_max(data_tensor, n_samples_mc)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = log_lkl / n_samples
        if verbose:
            print("VR_max", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def posterior_var(self, encoder_key="default", counts=None):
        # Iterate once over the posterior and get the marginal variance
        ave_var = np.zeros(self.model.n_latent)
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            _, _, _, qz_v, _ = self.model.inference(
                data_tensor, encoder_key=encoder_key, counts=counts
            )
            ave_var += torch.sum(qz_v, dim=0).cpu().detach().numpy()
        n_samples = len(self.indices)
        return ave_var / n_samples

    @torch.no_grad()
    def prob_eval(
        self,
        n_samples_mc,
        nu=0.0,
        encoder_key="default",
        counts=None,
        z_encoder: nn.Module = None,
        plugin_estimator=False,
    ):
        # Iterate once over the posterior and get the marginal variance
        prob = []
        qz_m = []
        qz_v = []
        ess = []
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            if plugin_estimator:
                x, y, z, t = self.model.prob_event_plugin(
                    data_tensor,
                    n_samples_mc,
                    nu=nu,
                    encoder_key=encoder_key,
                    counts=counts,
                    z_encoder=z_encoder,
                )
            else:
                x, y, z, t = self.model.prob_event(
                    data_tensor,
                    n_samples_mc,
                    nu=nu,
                    encoder_key=encoder_key,
                    counts=counts,
                    z_encoder=z_encoder,
                )
            if x is not None:
                x = x.cpu()
            if y is not None:
                y = y.cpu()
            qz_m += [x]
            qz_v += [y]
            prob += [z.cpu()]
            ess += [t.cpu()]

        if x is not None:
            qz_m = np.array(torch.cat(qz_m))
        if y is not None:
            qz_v = np.array(torch.cat(qz_v))
        return (
            qz_m,
            qz_v,
            np.array(torch.cat(prob)),
            np.array(torch.cat(ess)),
        )

    def log_ratios(
        self,
        n_samples_mc,
        encoder_key="default",
        counts=None,
        z_encoder: nn.Module = None,
    ):
        all_log_ratios = []
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            px_mean, px_var, qz_m, qz_v, z, log_qz_given_x = self.model.inference(
                data_tensor,
                n_samples=n_samples_mc,
                reparam=True,
                encoder_key=encoder_key,
                counts=counts,
                z_encoder=z_encoder,
            )
            log_ratios = self.model.log_ratio(
                data_tensor, px_mean, px_var, log_qz_given_x, z
            )
            all_log_ratios.append(log_ratios.cpu())
            # Shapes (n_samples, n_batch)
        return torch.cat(all_log_ratios, dim=-1)
