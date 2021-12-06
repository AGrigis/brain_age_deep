![Pep8](https://github.com/AGrigis/brain_age_deep/actions/workflows/pep8.yml/badge.svg)
![Testing Conda](https://github.com/AGrigis/brain_age_deep/actions/workflows/testing_conda.yml/badge.svg)
![Testing Pip](https://github.com/AGrigis/brain_age_deep/actions/workflows/testing_pip.yml/badge.svg)
![Testing Notebook](https://github.com/AGrigis/brain_age_deep/actions/workflows/testing_notebook.yml/badge.svg)


# Brain age regression

Predict age from brain grey matter (regression) using Deep Learning.
Aging is associated with grey matter (GM) atrophy. Each year, an adult lose
0.1% of GM. We will try to learn a predictor of the chronological age (true age)
using GM measurements on a population of healthy control participants.

Such a predictor provides the expected **brain age** of a subject. Deviation from
this expected brain age indicates acceleration or slowdown of the aging process
which may be associated with a pathological neurobiological process or protective factor of aging.

## Dataset

There are 357 samples in the training set and 90 samples in the test set.

### Input data

Voxel-based_morphometry [VBM](https://en.wikipedia.org/wiki/Voxel-based_morphometry)
using [cat12](http://www.neuro.uni-jena.de/cat/) software which provides:

- Regions Of Interest (`rois`) of Grey Matter (GM) scaled for the Total
  Intracranial Volume (TIV): `[train|test]_rois.csv` 284 features.

- VBM GM 3D maps or images (`vbm3d`) of [voxels](https://en.wikipedia.org/wiki/Voxel) in the
  [MNI](https://en.wikipedia.org/wiki/Talairach_coordinates) space:
  `[train|test]_vbm.npz` contains 3D images of shapes (121, 145, 121).
  This npz contains also the 3D mask and the affine transformation to MNI
  referential. Masking the brain provides *flat* 331 695 input features (masked voxels)
  for each participant.

By default `problem.get_[train|test]_data()` return the concatenation of 284 ROIs of
Grey Matter (GM) features with 331 695 features (voxels) within a brain mask.
Those two blocks are higly redundant.
To select only `rois` features do:

```
X[:, :284]
```

To select only `vbm` features do:

```
X[:, 284:]
```

### Target

The target can be found in `[test|train]_participants.csv` files, selecting the
`age` column for regression problem.

## Evaluation metrics

[sklearn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

The main Evaluation metrics is the Root-mean-square deviation
[RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation). We will also
look at the R-squared
[R2](https://en.wikipedia.org/wiki/Coefficient_of_determination).

## Links

- [RAMP-workflow’s documentation](https://paris-saclay-cds.github.io/ramp-workflow)
- [RAMP-workflow’s github](https://github.com/paris-saclay-cds/ramp-workflow)
- [RAMP Kits](https://github.com/ramp-kits)

## Installation

This starting kit requires Python and the following dependencies:

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplolib`
* `seaborn`
* `jupyter`
* `torch`
* `ramp-workflow`

You can install the dependencies with the following command-line:

```
pip install -U -r requirements.txt
```

If you are using conda, we provide an environment.yml file for similar usage.

```
conda env create -n brain_age -f environment.yml
```

Then, you can activate/deasactivate the conda environment using:

```
conda activate brain_age
conda deactivate
```

1. Download the data

```
python download_data.py
```

The train/test data will be available in the `data` directory.

2. Execute the jupyter notebook

```
jupyter notebook brain_age_starting_kit.ipynb
```

Play with this notebook to create your new model.

3. Test Submission

The submissions need to be located in the `submissions` folder.
For instance to create a `linear_regression_rois` submission, start by
copying the starting kit

```
cp -r submissions/submissions/strating_kit submissions/submissions/linear_regression_rois`.
```
 
Tune the estimator in the`submissions/submissions/linear_regression_rois/estimator.py` file.
This file must contain a function `get_estimator()` that returns a scikit learn Pipeline.

Then, test your submission locally:

```
ramp-test --submission linear_regression_rois
```

4. Submission on N4H RAMP

Connect to your N4H RAMP account, select the `brain_age_deep` event, and submit your estimator in the
sandbox section. x
Note that no training will be performed on the server side.
You thus need to join the weights of the model in a file called `weights.pth`.
