import copy
import sys
import time

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from . import Trainer

plt.switch_backend("agg")


class UnsupervisedTrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or and integer for the number of training samples
         to use Default: ``0.8``.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)
    """
    default_metrics_to_monitor = ["ll"]

    def __init__(
        self, model, gene_dataset, train_size=0.8, test_size=None, kl=None, **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)
        self.kl = kl
        self.iter = 0
        if type(self) is UnsupervisedTrainer:
            self.train_set, self.test_set = self.train_test(
                model, gene_dataset, train_size, test_size
            )
            self.train_set.to_monitor = ["ll"]
            self.test_set.to_monitor = ["ll"]

        self.metrics = dict(
            train_theta_wake=[],
            train_phi_wake=[],
            train_phi_sleep=[],
            train_loss=[],
            classification_loss=[],
            train_cubo=[],
        )

    def train_aevb(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        begin = time.time()
        self.model.train()

        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        self.compute_metrics()

        with trange(
            n_epochs, desc="training", file=sys.stdout, disable=self.verbose
        ) as pbar:
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                for tensors_list in self.data_loaders_loop():
                    (
                        sample_batch,
                        local_l_mean,
                        local_l_var,
                        batch_index,
                        _,
                    ) = tensors_list[0]
                    elbo = self.model(
                        sample_batch,
                        local_l_mean,
                        local_l_var,
                        batch_index,
                        loss_type="ELBO",
                    )
                    optimizer.zero_grad()
                    elbo.backward()
                    optimizer.step()

                if not self.on_epoch_end():
                    break

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time

    def train(
        self,
        n_epochs=20,
        lr=1e-3,
        lr_theta=None,
        lr_phi=None,
        eps=0.01,
        wake_theta="ELBO",
        wake_psi="ELBO",
        n_samples_theta=1,
        n_samples_phi=1,
        n_warmup=5,
        n_every_theta=1,
        n_every_phi=1,
        reparam=True,
        include_library_in_elbo=False,
        do_observed_library=False,
    ):
        if (lr_theta is None) and (lr_phi is None):
            lr_theta = lr
            lr_phi = lr
        begin = time.time()
        self.model.train()

        if include_library_in_elbo:
            params_gen = list(
                filter(
                    lambda p: p.requires_grad,
                    list(self.model.decoder.parameters())
                    + list(self.model.l_encoder.parameters()),
                )
            ) + [self.model.px_r]

            params_var = filter(
                lambda p: p.requires_grad,
                list(self.model.z_encoder["default"].parameters()),
            )
        else:
            params_gen = list(
                filter(lambda p: p.requires_grad, self.model.decoder.parameters())
            ) + [self.model.px_r]

            params_var = filter(
                lambda p: p.requires_grad,
                list(self.model.l_encoder.parameters())
                + list(self.model.z_encoder["default"].parameters()),
            )

        optimizer_gen = torch.optim.Adam(params_gen, lr=lr_theta, eps=eps)

        optimizer_var_wake = torch.optim.Adam(params_var, lr=lr_phi, eps=eps)
        # optimizer_var_sleep = torch.optim.Adam(params_var, lr=lr, eps=eps)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        self.compute_metrics()

        with trange(
            n_epochs, desc="training", file=sys.stdout, disable=self.verbose
        ) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)

                # for all minibatches, update phi and psi alternately
                for tensors_list in self.data_loaders_loop():
                    (
                        sample_batch,
                        local_l_mean,
                        local_l_var,
                        batch_index,
                        _,
                    ) = tensors_list[0]

                    # wake theta update
                    if self.iter % n_every_theta == 0:
                        elbo = self.model(
                            sample_batch,
                            local_l_mean,
                            local_l_var,
                            batch_index,
                            loss_type=wake_theta,
                            n_samples=n_samples_theta,
                            do_observed_library=do_observed_library,
                        )
                        loss = torch.mean(elbo)
                        optimizer_gen.zero_grad()
                        loss.backward()
                        optimizer_gen.step()

                    if self.iter % 100 == 0:
                        self.metrics["train_theta_wake"].append(loss.item())

                    # wake phi update
                    # Wake phi
                    if wake_psi == "REVKL+CUBO":
                        if self.epoch <= n_warmup:
                            wake_psi_epoch = "REVKL"
                            reparam_epoch = False
                        else:
                            wake_psi_epoch = "CUBO"
                            reparam_epoch = True
                    elif wake_psi == "ELBO+CUBO":
                        reparam_epoch = True
                        if self.epoch <= n_warmup:
                            wake_psi_epoch = "ELBO"
                        else:
                            wake_psi_epoch = "CUBO"
                    elif wake_psi == "ELBO+REVKL":
                        if self.epoch <= n_warmup:
                            wake_psi_epoch = "ELBO"
                            reparam_epoch = True
                        else:
                            wake_psi_epoch = "REVKL"
                            reparam_epoch = False

                    else:
                        wake_psi_epoch = wake_psi
                        reparam_epoch = reparam

                    if self.iter % n_every_phi == 0:
                        loss = self.model(
                            sample_batch,
                            local_l_mean,
                            local_l_var,
                            batch_index,
                            loss_type=wake_psi_epoch,
                            n_samples=n_samples_phi,
                            reparam=reparam_epoch,
                            do_observed_library=do_observed_library,
                        )
                        loss = torch.mean(loss)
                        optimizer_var_wake.zero_grad()
                        loss.backward()
                        optimizer_var_wake.step()

                    if self.iter % 100 == 0:
                        self.metrics["train_phi_wake"].append(loss.item())
                    # # Sleep phi update
                    # synthetic_obs = self.model.generate_new_obs(
                    #     sample_batch,
                    #     batch_index=batch_index,
                    # )
                    #
                    # loss = self.mod

                    if self.iter % 100 == 0:
                        other_metrics = self.test_set.sequential().getter(
                            keys=["CUBO"], n_samples=10
                        )
                        self.metrics["train_cubo"].append(other_metrics["CUBO"].mean())

                    self.iter += 1

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

    def train_defensive(
        self,
        n_epochs=20,
        lr=1e-3,
        lr_theta=None,
        lr_phi=None,
        eps=0.01,
        wake_theta="ELBO",
        wake_psi="ELBO",
        n_samples_theta=1,
        n_samples_phi=1,
        counts=torch.tensor([8, 8, 2]),
    ):
        if (lr_theta is None) and (lr_phi is None):
            lr_theta = lr
            lr_phi = lr
        begin = time.time()
        self.model.train()

        params_gen = list(
            filter(lambda p: p.requires_grad, self.model.decoder.parameters())
        ) + [self.model.px_r]

        params_var_cubo = filter(
            lambda p: p.requires_grad, list(self.model.z_encoder["CUBO"].parameters())
        )

        params_var_eubo = filter(
            lambda p: p.requires_grad, list(self.model.z_encoder["EUBO"].parameters())
        )

        optimizer_gen = torch.optim.Adam(params_gen, lr=lr_theta, eps=eps)
        optimizer_var_cubo = torch.optim.Adam(params_var_cubo, lr=lr_phi, eps=eps)
        optimizer_var_eubo = torch.optim.Adam(params_var_eubo, lr=lr_phi, eps=eps)
        # optimizer_var_sleep = torch.optim.Adam(params_var, lr=lr, eps=eps)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        self.compute_metrics()

        with trange(
            n_epochs, desc="training", file=sys.stdout, disable=self.verbose
        ) as pbar:
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)
                for tensors_list in self.data_loaders_loop():
                    (
                        sample_batch,
                        local_l_mean,
                        local_l_var,
                        batch_index,
                        _,
                    ) = tensors_list[0]

                    # wake theta update
                    elbo = self.model(
                        sample_batch,
                        local_l_mean,
                        local_l_var,
                        batch_index,
                        loss_type=wake_theta,
                        n_samples=n_samples_theta,
                        do_observed_library=True,
                        encoder_key="defensive",
                        counts=counts,
                    )
                    loss = torch.mean(elbo)
                    optimizer_gen.zero_grad()
                    loss.backward()
                    optimizer_gen.step()

                    if self.iter % 100 == 0:
                        self.metrics["train_theta_wake"].append(loss.item())

                    # wake phi update
                    # Wake phi CUBO
                    loss = self.model(
                        sample_batch,
                        local_l_mean,
                        local_l_var,
                        batch_index,
                        loss_type="CUBO",
                        n_samples=n_samples_phi,
                        reparam=True,
                        encoder_key="CUBO",
                        do_observed_library=True,
                    )
                    loss = torch.mean(loss)
                    optimizer_var_cubo.zero_grad()
                    loss.backward()
                    optimizer_var_cubo.step()

                    # wake phi update
                    # Wake phi EUBO
                    loss = self.model(
                        sample_batch,
                        local_l_mean,
                        local_l_var,
                        batch_index,
                        loss_type="REVKL",
                        n_samples=n_samples_phi,
                        reparam=False,
                        encoder_key="EUBO",
                        do_observed_library=True,
                    )
                    loss = torch.mean(loss)
                    optimizer_var_eubo.zero_grad()
                    loss.backward()
                    optimizer_var_eubo.step()

                    if self.iter % 100 == 0:
                        other_metrics = self.test_set.sequential().getter(
                            keys=["CUBO"],
                            n_samples=10,
                            encoder_key="defensive",
                            counts=counts,
                            do_observed_library=True,
                        )
                        self.metrics["train_cubo"].append(other_metrics["CUBO"].mean())

                    self.iter += 1

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


class AdapterTrainer(UnsupervisedTrainer):
    def __init__(self, model, gene_dataset, posterior_test, frequency=5):
        super().__init__(model, gene_dataset, frequency=frequency)
        self.test_set = posterior_test
        self.test_set.to_monitor = ["ll"]
        self.params = list(self.model.z_encoder.parameters()) + list(
            self.model.l_encoder.parameters()
        )
        self.z_encoder_state = copy.deepcopy(model.z_encoder.state_dict())
        self.l_encoder_state = copy.deepcopy(model.l_encoder.state_dict())

    @property
    def posteriors_loop(self):
        return ["test_set"]

    def train(self, n_path=10, n_epochs=50, **kwargs):
        for i in range(n_path):
            # Re-initialize to create new path
            self.model.z_encoder.load_state_dict(self.z_encoder_state)
            self.model.l_encoder.load_state_dict(self.l_encoder_state)
            super().train(n_epochs, params=self.params, **kwargs)

        return min(self.history["ll_test_set"])
