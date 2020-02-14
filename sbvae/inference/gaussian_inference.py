import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal
from tqdm import trange

from . import Posterior, Trainer

plt.switch_backend("agg")


class GaussianTrainer(Trainer):
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
        if type(self) is GaussianTrainer:
            self.train_set, self.test_set = self.train_test(
                model, dataset, train_size, test_size, type_class=GaussianPosterior
            )
            self.train_set.to_monitor = ["elbo"]
            self.test_set.to_monitor = ["elbo"]

    def train(self, params, losses, n_epochs=20, lr=1e-3, eps=0.01):
        params_gen, params_wvar, params_svar = params
        loss_gen, loss_wvar, loss_svar = losses
        begin = time.time()
        self.model.train()

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
        self.compute_metrics()

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
                        loss = torch.mean(self.model(data_tensor, loss_gen))
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
                        loss = torch.mean(self.model(data_tensor, loss_wvar_epoch))
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
                        loss = torch.mean(self.model((x, z), loss_svar))
                        optimizers[2].zero_grad()
                        loss.backward()
                        optimizers[2].step()

                if not self.on_epoch_end():
                    break

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


class GaussianPosterior(Posterior):
    @torch.no_grad()
    def elbo(self, verbose=False):
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
    def iwelbo(self, n_samples_mc, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.neg_iwelbo(data_tensor, n_samples_mc)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = -log_lkl / n_samples
        if verbose:
            print("IWELBO", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def cubo(self, n_samples_mc, verbose=False):
        # Iterate once over the posterior and computes the total log_likelihood
        log_lkl = 0
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            loss = self.model.cubo(data_tensor, n_samples_mc)
            log_lkl += torch.sum(loss).item()
        n_samples = len(self.indices)
        ll = log_lkl / n_samples
        if verbose:
            print("CUBO", n_samples_mc, " : %.4f" % ll)
        return ll

    @torch.no_grad()
    def vr_max(self, n_samples_mc, verbose=False):
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
    def posterior_var(self):
        # Iterate once over the posterior and get the marginal variance
        ave_var = np.zeros(self.model.n_latent)
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            _, _, _, qz_v, _ = self.model.inference(data_tensor)
            ave_var += torch.sum(qz_v, dim=0).cpu().detach().numpy()
        n_samples = len(self.indices)
        return ave_var / n_samples

    @torch.no_grad()
    def prob_eval(self, n_samples_mc, nu=0.0):
        # Iterate once over the posterior and get the marginal variance
        prob = []
        qz_m = []
        qz_v = []
        ess = []
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            x, y, z, t = self.model.prob_event(data_tensor, n_samples_mc, nu=nu)
            qz_m += [x.cpu()]
            qz_v += [y.cpu()]
            prob += [z.cpu()]
            ess += [t.cpu()]

        if not self.model.linear_encoder:
            qz_v_return = np.array(torch.cat(qz_v))
        else:
            qz_v_return = qz_v[0]  # Constant cov structure

        return (
            np.array(torch.cat(qz_m)),
            qz_v_return,
            np.array(torch.cat(prob)),
            np.array(torch.cat(ess)),
        )

    def log_ratios(self, n_samples_mc):
        all_log_ratios = []
        for i_batch, tensors in enumerate(self):
            data_tensor = torch.stack(tensors, 0)
            px_mean, px_var, qz_m, qz_v, z = self.model.inference(
                data_tensor, n_samples=n_samples_mc, reparam=True
            )
            log_ratios = self.model.log_ratio(
                data_tensor, px_mean, px_var, qz_m, qz_v, z
            )
            all_log_ratios.append(log_ratios.cpu())
            # Shapes (n_samples, n_batch)
        return torch.cat(all_log_ratios, dim=-1)
