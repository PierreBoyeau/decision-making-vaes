import logging
from itertools import cycle

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sbvae.dataset import MnistDataset

logger = logging.getLogger(__name__)


class MnistRTrainer:
    def __init__(
        self,
        dataset: MnistDataset,
        model,
        batch_size: int = 128,
        classify_mode: str = "vanilla",
        r: float = 5e-4,
        use_cuda=True,
        save_metrics=False,
        debug_gradients=False,
    ):
        self.classify_mode = classify_mode
        self.r = r
        self.debug_gradients = debug_gradients
        self.dataset = dataset
        self.model = model
        self.train_loader = DataLoader(
            self.dataset.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        self.train_annotated_loader = DataLoader(
            self.dataset.train_dataset_labelled,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        self.test_loader = DataLoader(
            self.dataset.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_cuda,
        )
        self.cross_entropy_fn = CrossEntropyLoss()
        self.it = 0

        self.save_metrics = save_metrics
        self.iterate = 0
        self.metrics = dict(
            train_theta_wake=[],
            train_phi_wake=[],
            train_phi_sleep=[],
            train_loss=[],
            classification_loss=[],
            train_cubo=[],
            classification_gradients=[],
        )

    @property
    def temperature(self):
        t_ref = self.it - (self.it % 500)
        t_ref = t_ref * self.r
        t_ref = np.exp(-t_ref)
        return np.maximum(0.5, t_ref)

    def train(
        self,
        n_epochs,
        lr=1e-3,
        overall_loss: str = None,
        wake_theta: str = "ELBO",
        wake_psi: str = "ELBO",
        n_samples: int = 1,
        n_samples_phi: int = None,
        n_samples_theta: int = None,
        classification_ratio: float = 50.0,
        update_mode: str = "all",
        reparam_wphi: bool = True,
        z2_with_elbo: bool = False,
    ):
        assert update_mode in ["all", "alternate"]
        assert (n_samples_phi is None) == (n_samples_theta is None)

        if n_samples is not None:
            n_samples_theta = n_samples
            n_samples_phi = n_samples
        logger.info(
            "Using {n_samples_theta} and {n_samples_phi} samples for theta wake / phi wake".format(
                n_samples_theta=n_samples_theta, n_samples_phi=n_samples_phi
            )
        )

        optim = None
        optim_gen = None
        optim_var_wake = None
        if overall_loss is not None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            optim = Adam(params, lr=lr)
            logger.info("Monobjective using {} loss".format(overall_loss))
        else:
            params_gen = filter(
                lambda p: p.requires_grad,
                list(self.model.decoder_z1_z2.parameters())
                + list(self.model.x_decoder.parameters()),
            )
            optim_gen = Adam(params_gen, lr=lr)

            params_var = filter(
                lambda p: p.requires_grad,
                list(self.model.classifier.parameters())
                + list(self.model.encoder_z1.parameters())
                + list(self.model.encoder_z2_z1.parameters()),
            )

            optim_var_wake = Adam(params_var, lr=lr)
            logger.info(
                "Multiobjective training using {} / {}".format(wake_theta, wake_psi)
            )

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            for (tensor_all, tensor_superv) in zip(
                self.train_loader, cycle(self.train_annotated_loader)
            ):
                self.it += 1

                x_u, _ = tensor_all
                x_s, y_s = tensor_superv

                x_u = x_u.to("cuda")
                x_s = x_s.to("cuda")
                y_s = y_s.to("cuda")

                if overall_loss is not None:
                    loss = self.loss(
                        x_u=x_u,
                        x_s=x_s,
                        y_s=y_s,
                        loss_type=overall_loss,
                        n_samples=n_samples,
                        reparam=True,
                        classification_ratio=classification_ratio,
                        mode=update_mode,
                    )
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    # torch.cuda.synchronize()

                    if self.iterate % 100 == 0:
                        self.metrics["train_loss"].append(loss.item())
                else:
                    # Wake theta
                    theta_loss = self.loss(
                        x_u=x_u,
                        x_s=x_s,
                        y_s=y_s,
                        loss_type=wake_theta,
                        n_samples=n_samples_theta,
                        reparam=True,
                        classification_ratio=classification_ratio,
                        mode=update_mode,
                    )
                    optim_gen.zero_grad()
                    theta_loss.backward()
                    optim_gen.step()
                    # torch.cuda.synchronize()

                    if self.iterate % 100 == 0:
                        self.metrics["train_theta_wake"].append(theta_loss.item())

                    reparam_epoch = reparam_wphi
                    wake_psi_epoch = wake_psi

                    psi_loss = self.loss(
                        x_u=x_u,
                        x_s=x_s,
                        y_s=y_s,
                        loss_type=wake_psi_epoch,
                        n_samples=n_samples_phi,
                        reparam=reparam_epoch,
                        classification_ratio=classification_ratio,
                        mode=update_mode,
                    )
                    optim_var_wake.zero_grad()
                    psi_loss.backward()
                    optim_var_wake.step()
                    # torch.cuda.synchronize()
                    if self.iterate % 100 == 0:
                        self.metrics["train_phi_wake"].append(psi_loss.item())
                        if self.debug_gradients:
                            self.metrics["classification_gradients"].append(
                                self.model.classifier["default"]
                                .classifier[0]
                                .to_hidden.weight.grad.cpu()
                            )

                self.iterate += 1
            pbar.set_description("{0:.2f}".format(theta_loss.item()))

    def train_eval_encoder(
        self,
        encoders: dict,
        n_epochs: int,
        lr: float = 1e-3,
        wake_psi: str = "ELBO",
        n_samples_phi: int = None,
        classification_ratio: float = 50.0,
        reparam_wphi: bool = True,
    ):
        reparam_mapper = dict(
            default=reparam_wphi,
            ELBO=True,
            CUBO=True,
            REVKL=False,
            IWELBO=True,
            IWELBOC=True,
        )

        logger.info(
            "Using {n_samples_phi} samples for eval encoder training".format(
                n_samples_phi=n_samples_phi
            )
        )
        classifier = encoders["classifier"]
        encoder_z1 = encoders["encoder_z1"]
        encoder_z2_z1 = encoders["encoder_z2_z1"]
        self.model.update_q(
            classifier=classifier, encoder_z1=encoder_z1, encoder_z2_z1=encoder_z2_z1,
        )

        # params_var = filter(
        #     lambda p: p.requires_grad,
        #     list(classifier.parameters())
        #     + list(encoder_z1.parameters())
        #     + list(encoder_z2_z1.parameters()),
        # )
        # optim_var_wake = Adam(params_var, lr=lr)

        if type(wake_psi) == list:
            encoder_keys = wake_psi
        else:
            encoder_keys = ["default"]

        def get_params(key):
            return filter(
                lambda p: p.requires_grad,
                list(classifier[key].parameters())
                + list(encoder_z1[key].parameters())
                + list(encoder_z2_z1[key].parameters()),
            )

        params_var = {key: get_params(key) for key in encoder_keys}
        optim_vars = {key: Adam(params_var[key], lr=lr) for key in encoder_keys}

        logger.info("Training using {}".format(wake_psi))

        for epoch in tqdm(range(n_epochs)):
            for (tensor_all, tensor_superv) in zip(
                self.train_loader, cycle(self.train_annotated_loader)
            ):

                x_u, _ = tensor_all
                x_s, y_s = tensor_superv

                x_u = x_u.to("cuda")
                x_s = x_s.to("cuda")
                y_s = y_s.to("cuda")

                # Wake phi
                for key in encoder_keys:
                    if key == "default":
                        reparam_epoch = reparam_wphi
                        wake_psi_epoch = wake_psi
                    else:
                        reparam_epoch = reparam_mapper[key]
                        wake_psi_epoch = key

                    psi_loss = self.loss(
                        x_u=x_u,
                        x_s=x_s,
                        y_s=y_s,
                        loss_type=wake_psi_epoch,
                        n_samples=n_samples_phi,
                        reparam=reparam_epoch,
                        encoder_key=key,
                        classification_ratio=classification_ratio,
                    )
                    optim_vars[key].zero_grad()
                    psi_loss.backward()
                    optim_vars[key].step()
                    # torch.cuda.synchronize()
                    self.iterate += 1

    def train_defensive(
        self,
        n_epochs,
        counts: pd.Series,
        lr=1e-3,
        wake_theta: str = "ELBO",
        n_samples_phi: int = None,
        n_samples_theta: int = None,
        classification_ratio: float = 50.0,
        update_mode: str = "all",
    ):
        reparams_info = dict(
            CUBO=True, IWELBO=True, IWELBOC=True, ELBO=True, CUBOB=True, REVKL=False
        )
        assert update_mode in ["all", "alternate"]

        params_gen = filter(
            lambda p: p.requires_grad,
            list(self.model.decoder_z1_z2.parameters())
            + list(self.model.x_decoder.parameters()),
        )
        optim_gen = Adam(params_gen, lr=lr)

        def get_params(key):
            return filter(
                lambda p: p.requires_grad,
                list(self.model.classifier[key].parameters())
                + list(self.model.encoder_z1[key].parameters())
                + list(self.model.encoder_z2_z1[key].parameters()),
            )

        encoder_keys = counts.loc[lambda x: x.index != "prior"].keys()
        params_var = {key: get_params(key) for key in encoder_keys}
        optim_vars = {key: Adam(params_var[key], lr=lr) for key in encoder_keys}
        for epoch in tqdm(range(n_epochs)):
            for (tensor_all, tensor_superv) in zip(
                self.train_loader, cycle(self.train_annotated_loader)
            ):

                x_u, _ = tensor_all
                x_s, y_s = tensor_superv

                x_u = x_u.to("cuda")
                x_s = x_s.to("cuda")
                y_s = y_s.to("cuda")

                # Wake theta
                theta_loss = self.loss(
                    x_u=x_u,
                    x_s=x_s,
                    y_s=y_s,
                    loss_type=wake_theta,
                    n_samples=n_samples_theta,
                    reparam=True,
                    classification_ratio=classification_ratio,
                    encoder_key="defensive",
                    counts=counts,
                )
                optim_gen.zero_grad()
                theta_loss.backward()
                optim_gen.step()

                # if self.iterate % 100 == 0:
                #     self.metrics["train_theta_wake"].append(theta_loss.item())

                for key in encoder_keys:
                    do_reparam = reparams_info[key]
                    var_loss = self.loss(
                        x_u=x_u,
                        x_s=x_s,
                        y_s=y_s,
                        loss_type=key,
                        n_samples=n_samples_phi,
                        reparam=do_reparam,
                        classification_ratio=classification_ratio,
                        encoder_key=key,
                    )
                    optim_vars[key].zero_grad()
                    var_loss.backward()
                    optim_vars[key].step()
            self.iterate += 1

    def loss(
        self,
        x_u,
        x_s,
        y_s,
        loss_type,
        n_samples=5,
        reparam=True,
        classification_ratio=50.0,
        mode="all",
        encoder_key="default",
        counts=None,
    ):
        temp = self.temperature
        labelled_fraction = self.dataset.labelled_fraction
        s_every = int(1 / labelled_fraction)

        if mode == "all":
            outs_s = None
            l_u = self.model.forward(
                x_u,
                temperature=temp,
                loss_type=loss_type,
                n_samples=n_samples,
                reparam=reparam,
                encoder_key=encoder_key,
                counts=counts,
            )
            l_s = self.model.forward(
                x_s,
                temperature=temp,
                loss_type=loss_type,
                y=y_s,
                n_samples=n_samples,
                reparam=reparam,
                encoder_key=encoder_key,
                counts=counts,
            )
            # torch.cuda.synchronize()
            l_s = labelled_fraction * l_s
            j = l_u.mean() + l_s.mean()
        elif mode == "alternate":
            outs_s = None
            if self.iterate % s_every == 0:
                l_s = self.model.forward(
                    x_s,
                    temperature=temp,
                    loss_type=loss_type,
                    y=y_s,
                    n_samples=n_samples,
                    reparam=reparam,
                    encoder_key=encoder_key,
                    counts=counts,
                )
                j = l_s.mean()
            else:
                l_u = self.model.forward(
                    x_u,
                    temperature=temp,
                    loss_type=loss_type,
                    n_samples=n_samples,
                    reparam=reparam,
                    encoder_key=encoder_key,
                    counts=counts,
                )
                j = l_u.mean()
        else:
            raise ValueError("Mode {} not recognized".format(mode))

        if encoder_key == "defensive":
            # Classifiers' gradients are null wrt theta
            l_class = 0.0
        else:
            # y_pred = self.model.classify(
            #     x_s,
            #     encoder_key=encoder_key,
            #     mode=self.classify_mode,
            #     n_samples=n_samples,
            # )
            if self.classify_mode != "vanilla":
                y_pred = self.model.classify(
                    x_s,
                    encoder_key=encoder_key,
                    mode=self.classify_mode,
                    n_samples=n_samples,
                )
            else:
                y_pred = self.model.classify(x_s, encoder_key=encoder_key)
            l_class = self.cross_entropy_fn(y_pred, target=y_s)
        loss = j + classification_ratio * l_class

        if self.save_metrics:
            if self.iterate % 100 == 0:
                self.metrics["classification_loss"].append(l_class.item())
        return loss

    @torch.no_grad()
    def inference(
        self,
        data_loader,
        do_supervised=False,
        keys=None,
        n_samples: int = 10,
        eval_mode=True,
        encoder_key="default",
        counts=None,
    ) -> dict:
        all_res = dict()
        if eval_mode:
            self.model = self.model.eval()
        else:
            self.model = self.model.train()
        for tensor_all in data_loader:
            x, y = tensor_all
            x = x.to("cuda")
            y = y.to("cuda")
            if not do_supervised:
                res = self.model.inference(
                    x,
                    n_samples=n_samples,
                    encoder_key=encoder_key,
                    counts=counts,
                    temperature=0.5,
                    reparam=False,
                )
            else:
                raise ValueError("Not sure")
                res = self.model.inference(
                    x,
                    y=y,
                    n_samples=n_samples,
                    encoder_key=encoder_key,
                    counts=counts,
                    temperature=0.5,
                    reparam=False,
                )
            res["y"] = y
            if keys is not None:
                filtered_res = {key: val for (key, val) in res.items() if key in keys}
            else:
                filtered_res = res
            if "preds_is" in keys:
                filtered_res["preds_is"] = self.model.classify(
                    x,
                    n_samples=n_samples,
                    mode="is",
                    counts=counts,
                    encoder_key=encoder_key,
                )
            if "preds_plugin" in keys:
                filtered_res["preds_plugin"] = self.model.classify(
                    x,
                    n_samples=n_samples,
                    mode="plugin",
                    counts=counts,
                    encoder_key=encoder_key,
                )

            is_labelled = False
            if counts is None:
                log_ratios = (
                    res["log_pz2"]
                    + res["log_pc"]
                    + res["log_pz1_z2"]
                    + res["log_px_z"]
                    - res["log_qz1_x"]
                    - res["log_qz2_z1"]
                    - res["log_qc_z1"]
                )
            else:
                log_ratios = res["log_ratio"]

            if "CUBO" in keys:
                filtered_res["CUBO"] = self.model.cubo(
                    log_ratios=log_ratios, is_labelled=is_labelled, evaluate=True, **res
                )
            if "IWELBO" in keys:
                filtered_res["IWELBO"] = self.model.iwelbo(
                    log_ratios=log_ratios, is_labelled=is_labelled, evaluate=True, **res
                )
            if "log_ratios" in keys:
                # n_labels, n_samples, n_batch = log_ratios.shape
                # log_ratios = log_ratios.view(-1, n_batch)
                # samp = np.random.choice(n_labels * n_samples, size=n_samples)
                # log_ratios = log_ratios[samp, :]
                filtered_res["log_ratios"] = log_ratios

            all_res = dic_update(all_res, filtered_res)
        batch_size = data_loader.batch_size
        all_res = dic_concat(all_res, batch_size=batch_size)
        return all_res


def dic_update(dic: dict, new_dic: dict):
    """
    Updates dic by appending `new_dict` values
    """
    for key, li in new_dic.items():
        if key in dic:
            dic[key].append(li.cpu())
        else:
            dic[key] = [li.cpu()]
    return dic


def dic_concat(dic: dict, batch_size: int = 128):
    for key, li in dic.items():
        tensor_shape = np.array(li[0].shape)
        dim = np.where(tensor_shape == batch_size)[0][0]
        dim = int(dim)
        dic[key] = torch.cat(li, dim=dim)
    return dic
