# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
This parametrizes the setup and uses building blocks from RAMP workflow. 
"""

import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.utils import _safe_indexing
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


class DeepEstimator(rw.workflows.SKLearnPipeline):
    """ Wrapper to convert a scikit-learn estimator into a Deep Learning
    RAMP workflow.

    Notes
    -----
    The training is not performed on the server side. Weights are required at
    the submission time and fold indices are tracked at training time.
    """
    def __init__(self, filename="estimator.py", additional_filenames=None):
        super().__init__(filename, additional_filenames)

    def train_submission(self, module_path, X, y, train_idx=None):
        """ Train the estimator of a given submission.

        The first train indices is the fold index.
        """
        if train_idx is None:
            raise ValueError("Please provide a cross validation strategy.")
        fold_idx = train_idx[0]
        train_idx = train_idx[1:]
        submission_module = rw.utils.import_module_from_source(
            os.path.join(module_path, self.filename),
            os.path.splitext(self.filename)[0],
            sanitize=True)
        estimator = submission_module.get_estimator()
        if not isinstance(estimator, Pipeline):
            raise ValueError("The estimator must be an instance of "
                             "sklearn.pipeline.Pipeline.")
        for item in estimator:
            if hasattr(item, "fold_idx"):
                item.fold_idx = fold_idx
        X_train = _safe_indexing(X, train_idx)
        y_train = _safe_indexing(y, train_idx)
        return estimator.fit(X_train, y_train)


N_FOLDS = 5
problem_title = "Predict age from brain grey matter (regression)"
_target_column_name = "age"
Predictions = rw.prediction_types.make_regression()
workflow = DeepEstimator(
    filename="estimator.py", additional_filenames=["weights.pth"])
score_types = [
    rw.score_types.RMSE()
]


def get_cv(X, y):
    """ Get N folds cross validation indices.

    Notes
    -----
    Fold indices are tracked at training time and append to train indices at
    position 0.
    """
    cv_train = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    folds = []
    for cnt, (train_idx, test_idx) in enumerate(cv_train.split(X, y)):
        train_idx = np.insert(train_idx, 0, cnt)
        folds.append((train_idx, test_idx))
    return folds


def _read_data(path, dataset, datatype=["rois", "vbm"]):
    """ Read data.

    Parameters
    ----------
    path: str
        the data location.
    dataset: str
        'train' or 'test'.
    datatype: list of str, default ['rois', 'vbm']
        dataset type within 'rois', 'vbm'. By default returns a
        concatenation of rois and vbm data.

    Returns
    -------
    x_arr: array (n_samples, n_features)
        input data.
    y_arr: array (n_samples, )
        target data.
    """
    participants = pd.read_csv(os.path.join(
        path, "data", "{0}_participants.csv".format(dataset)))
    y_arr = participants[_target_column_name].values

    x_arr_l = []
    if "rois" in datatype:
        rois = pd.read_csv(os.path.join(
            path, "data", "{0}_rois.csv".format(dataset)))
        x_rois_arr = rois.loc[:, "l3thVen_GM_Vol":]
        assert x_rois_arr.shape[1] == 284
        x_arr_l.append(x_rois_arr)
    if "vbm" in datatype:
        imgs_arr_zip = np.load(os.path.join(
            path, "data", "{0}_vbm.npz".format(dataset)))
        x_img_arr = imgs_arr_zip["imgs_arr"].squeeze()
        mask_arr = imgs_arr_zip["mask_arr"]
        x_img_arr = x_img_arr[:, mask_arr]
        x_arr_l.append(x_img_arr)
    x_arr = np.concatenate(x_arr_l, axis=1)

    return x_arr, y_arr


def get_train_data(path=".", datatype=["rois", "vbm"]):
    return _read_data(path, "train", datatype)


def get_test_data(path=".", datatype=["rois", "vbm"]):
    return _read_data(path, "test", datatype)
