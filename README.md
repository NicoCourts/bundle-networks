# Bundle Networks

This repository consists of code that accompanies the paper

> Nico Courts and Henry Kvinge. Bundle Networks: Fiber Bundles, Local Trivializations, and a
> Generative Approach to Exploring Many-to-one Maps. In International Conference on Learning Representations, 2022 (to appear).

A preprint of this work can be found [on arXiv](https://arxiv.org/abs/2110.06983).

## Organization

In this repository, one can find several folders

- `bundlenet` contains the code for our BundleNet model as well as the CGAN/WGAN implementations used in our paper. This is also where one can find our evaluation and training scripts.
- `datasets` contains all data splits used for training and evaluation. These files can be loaded directly using the `torch.load` method.
- `notebooks` consists of Jupyter notebooks that demonstrate the use of the library.

## Working with BundleNets

To install on your computer, first clone this repository then navigate to this folder and execute `pip install -e ./`.
