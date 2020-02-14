import numpy as np
import torch
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

mnist_data = MNIST(root="sbvae_data/mnist", download=True)
X = mnist_data.data
labels = mnist_data.targets


class MnistDataset(MNIST):
    def __init__(
        self,
        labelled_fraction,
        labelled_proportions,
        test_size=0.7,
        do_1d=True,
        do_preprocess=True,
        **kwargs
    ):
        """

        """
        super().__init__(**kwargs)
        assert len(labelled_proportions) == 10
        labelled_proportions = np.array(labelled_proportions)
        assert abs(labelled_proportions.sum() - 1.0) <= 1e-16
        self.labelled_fraction = labelled_fraction
        label_proportions = labelled_fraction * labelled_proportions
        x = self.data.float()
        y = self.targets
        non_labelled = labelled_proportions == 0.0
        assert (
            non_labelled[1:].astype(int) - non_labelled[:-1].astype(int) >= 0
        ).all(), (
            "For convenience please ensure that non labelled numbers are the last ones"
        )
        non_labelled = np.where(labelled_proportions == 0.0)[0]
        if len(non_labelled) >= 1:
            y[np.isin(y, non_labelled)] = int(non_labelled[0])

        if do_preprocess:
            x = x / (1.0 * x.max())
        if do_1d:
            n_examples = len(x)
            x = x.view(n_examples, -1)

        if test_size > 0.0:
            ind_train, ind_test = train_test_split(
                np.arange(len(x)), test_size=test_size, random_state=42
            )
        else:
            ind_train = np.arange(len(x))
            ind_test = []
        x_train = x[ind_train]
        y_train = y[ind_train]
        x_test = x[ind_test]
        y_test = y[ind_test]
        n_all = len(x_train)

        n_labelled_per_class = (n_all * label_proportions).astype(int)
        labelled_inds = []
        for label in y.unique():
            label_ind = np.where(y_train == label)[0]
            labelled_exs = np.random.choice(label_ind, size=n_labelled_per_class[label])
            labelled_inds.append(labelled_exs)
        labelled_inds = np.concatenate(labelled_inds)

        self.labelled_inds = labelled_inds
        x_train_labelled = x_train[labelled_inds]
        y_train_labelled = y_train[labelled_inds]

        assert not (np.isin(np.unique(y_train_labelled), non_labelled)).any()
        self.train_dataset = TensorDataset(x_train, y_train)
        self.train_dataset_labelled = TensorDataset(x_train_labelled, y_train_labelled)
        self.test_dataset = TensorDataset(x_test, y_test)
