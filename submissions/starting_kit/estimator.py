# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Each solution to be tested should be stored in its own directory within
submissions/. The name of this new directory will serve as the ID for
the submission. If you wish to launch a RAMP challenge you will need to
provide an example solution within submissions/starting_kit/. Even if
you are not launching a RAMP challenge on RAMP Studio, it is useful to
have an example submission as it shows which files are required, how they
need to be named and how each file should be structured.
"""

import os
from abc import ABCMeta
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn


class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Select only the 284 ROIs features.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, :284]


class VBMFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Select only the 331 695 VBMs features.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, 284:]


class Dataset(torch.utils.data.Dataset):
    """ A torch dataset for regression.
    """
    def __init__(self, X, y=None):
        """ Init class.

        Parameters
        ----------
        X: array-like (n_samples, n_features)
            training data.
        y: array-like (n_samples, ), default None
            target values.
        """
        self.X = torch.from_numpy(X)
        if y is not None:
            self.y = torch.from_numpy(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is not None:
            return self.X[i], self.y[i]
        else:
            return self.X[i]


class MLP(nn.Module):
    """ Define a simple one hidden layer MLP.
    """
    def __init__(self, in_features):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
          nn.Linear(in_features, 120),
          nn.ReLU(),
          nn.Linear(120, 84),
          nn.ReLU(),
          nn.Linear(84, 1))

    def forward(self, x):
        return self.layers(x)


class RegressionModel(metaclass=ABCMeta):
    """ Base class for Regression models.

    When the model has been trained locally, the trained weights are stored
    in the `__model_local_weights__` file.

    Notes
    -----
    The fold_idx parameter will be set at runtime.
    """
    __model_local_weights__ = os.path.join(
        os.path.dirname(__file__), "weights.pth")

    def __init__(self, model, batch_size=10):
        """ Init class.

        Parameters
        ----------
        model: nn.Module
            the input model.
        batch_size:int, default 10
            the mini_batch size.
        """
        self.model = model
        self.batch_size = batch_size

    def fit(self, X, y):
        """ Fit model only locally otherwise restore weights.

        Parameters
        ----------
        X: array-like (n_samples, n_features)
            training data.
        y: array-like (n_samples, )
            target values.
        """
        self.model.train()
        if not os.path.isfile(self.__model_local_weights__):
            raise ValueError("You must provide the model weigths in your "
                             "submission folder.")
        state = torch.load(self.__model_local_weights__,
                           map_location="cpu")
        if "model" not in state:
            raise ValueError("Model weigths are searched in the state "
                             "dictionnary at the 'model' key location.")
        self.model.load_state_dict(state["model"])

    def predict(self, X):
        """ Predict using the input model.

        Parameters
        ----------
        X: array-like (n_samples, n_features)
            samples.

        Returns
        -------
        C: array (n_samples, )
            returns predicted values.
        """
        self.model.eval()
        dataset = Dataset(X)
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            C = []
            for i, inputs in enumerate(testloader):
                inputs = inputs.float()
                C.append(self.model(inputs))
            C = torch.cat(C, dim=0)
        return C.numpy().squeeze()


def get_estimator():
    """ Build your estimator here.

    Notes
    -----
    You must create an instance of sklearn.pipeline.Pipeline.
    """
    mlp = MLP(284)
    estimator = make_pipeline(ROIsFeatureExtractor(), StandardScaler(),
                              RegressionModel(mlp))
    return estimator
