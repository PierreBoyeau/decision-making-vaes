# -*- coding: utf-8 -*-

"""Handling datasets.
For the moment, is initialized with a torch Tensor of size (n_cells, nb_genes)"""
import copy
import os
import urllib.request
from collections import defaultdict

import numpy as np
import scipy.sparse as sp_sparse
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from scipy.linalg import sqrtm


class GaussianDataset(Dataset):
    """
    Gaussian dataset
    """

    def __init__(self, X):
        # Args:
        # Xs: a list of numpy tensors with .shape[1] identical (total_size*nb_genes)
        # or a list of scipy CSR sparse matrix,
        # or transposed CSC sparse matrix (the argument sparse must then be set to true)
        self.dense = type(X) is np.ndarray
        self._X = np.ascontiguousarray(X, dtype=np.float32) if self.dense else X
        self.nb_features = self.X.shape[1]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, batch):
        indexes = np.array(batch)
        X = self.X[indexes]
        return torch.from_numpy(X)


class SyntheticGaussianDataset(GaussianDataset):
    def __init__(self, dim_z=10, dim_x=100, rank_c=100, nu=0.5, n_samples=1000, seed=0):
        np.random.seed(seed)
        # Generating samples according to a Linear Gaussian system
        # mean of x
        self.px_mean = np.zeros((dim_x,))
        # conditional link
        self.A = 1 / np.sqrt(dim_z) * np.random.normal(size=(dim_x, dim_z))
        # conditional covar
        # sqrt = 1 / np.sqrt(rank_c) * np.random.normal(size=(dim_x, rank_c))
        # self.px_condvz_var = nu * np.eye(dim_x) + np.dot(sqrt.T, sqrt)
        self.gamma = nu * np.diag(np.random.normal(loc=1, scale=2, size=dim_x) ** 2)
        inv_gamma = np.linalg.inv(self.gamma)

        # marginal
        self.px_var = self.gamma + np.dot(self.A, self.A.T)

        # posterior
        inv_pz_condx_var = np.eye(dim_z) + np.dot(np.dot(self.A.T, inv_gamma), self.A)
        self.pz_condx_var = np.linalg.inv(inv_pz_condx_var)
        self.mz_cond_x_mean = np.dot(self.pz_condx_var, np.dot(self.A.T, inv_gamma))

        # joint distribution
        # store complete covariance matrix of z, x
        covar_joint = np.block([[np.eye(dim_z), self.A.T], [self.A, self.px_var]])
        self.pxz_log_det = np.log(np.linalg.det(covar_joint))
        self.pxz_inv_sqrt = sqrtm(np.linalg.inv(covar_joint))

        data = np.random.multivariate_normal(
            self.px_mean, self.px_var, size=(n_samples,)
        )

        super().__init__(data)


class SyntheticMixtureGaussianDataset(GaussianDataset):
    def __init__(self, dim_z=10, dim_x=10, nu=0.5, n_centers=3, n_samples=1000, seed=0):
        np.random.seed(seed)
        # Generating samples according to a Mixture Linear Gaussian system
        self.pi = np.random.dirichlet([1] * n_centers)
        self.pz_mean = np.random.normal(0, 3, size=(n_centers, dim_z))
        # we take all the covariances for pz_var to be equal to identity

        # link from z to x
        self.A = 1 / np.sqrt(dim_z) * np.random.normal(size=(dim_x, dim_z))
        self.gamma = (
            nu * np.random.normal(loc=1, scale=0.5, size=(n_centers, dim_x)) ** 2
        )

        # marginal of x conditioned on w
        self.px_mean = np.dot(self.pz_mean, self.A.T)  # shape (n_centers, dim_x)
        self.px_var = np.zeros(shape=(n_centers, dim_x, dim_x))
        for n in n_centers:
            self.px_var = np.diag(self.gamma[n]) + np.dot(self.A, self.A.T)

        # sampling process
        mixture_assignment = np.random.multinomial(
            1, self.pi, size=(n_samples,)
        )  # shape (n_samples, n_centers)
        gaussian_mean = np.dot(mixture_assignment, self.px_mean)
        gaussian_var = np.tensordot(mixture_assignment, self.px_var, axes=(1, 0))
        epsilon = np.random.multivariate_normal(
            np.zeros(shape=(dim_x,)), gaussian_var, size=(n_samples,)
        )
        data = gaussian_mean + epsilon

        # posterior variance of z conditioned on x, w
        inv_pz_condxw_var = np.eye(dim_z) + np.dot(np.dot(self.A.T, inv_gamma), self.A)
        self.pz_condxw_var = np.linalg.inv(inv_pz_condxw_var)  # shape (dim_z, dim_z)
        self.px_condxz_mean_link = np.dot(
            self.pz_condxw_var, np.dot(self.A.T, inv_gamma)
        )
        self.px_condxz_mean_bias = np.dot(
            self.pz_mean, self.pz_condxw_var
        )  # shape (n_centers, dim_z)

        # store complete covariance matrix of x, z conditioned on w
        off_diag_block = -np.dot(self.A.T, inv_gamma)
        precision_joint = np.block(
            [[inv_pz_condxw_var, off_diag_block], [off_diag_block.T, inv_gamma]]
        )
        self.pxz_condw_log_det = -np.log(np.linalg.det(precision_joint))
        self.pxz_condw_inv_sqrt = sqrtm(precision_joint)

        super().__init__(data)
