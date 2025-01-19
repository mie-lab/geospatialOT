[![Paper](https://img.shields.io/badge/paper-arXiv-brightgreen)](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/iclr2024/41/paper.pdf)
![Tests](https://github.com/mie-lab/geospatialOT/actions/workflows/python-tests.yml/badge.svg)

# GEOT: A spatially explicit framework for evaluating spatio-temporal predictions

This repo contains code to evaluate spatiotemporal predictions with Optimal Transport (OT). In contrast to standard evaluation metrics (MSE, MAE etc), the OT error is a *spatial* metric that evaluates the spatial distribution of the errors.

![Alt text](assets/overview.png)

## Install

Install the package via pypi:

```
pip install geot
```

You might have to install a suitable [torch](https://pytorch.org/get-started/locally/) version first. If you want to use Optimal Transport as a loss function for training a DL model, make sure to install torch for cuda.
 
Alternatively, install from source:
```
git clone https://github.com/mie-lab/geospatialOT.git
cd geospatialOT
conda create -n geot_env
conda activate geot_env
pip install .
```

## Tutorial

Check out our [tutorial](tutorial.ipynb) to get started with simple examples. 

#### Further explanations:

What is the goal?
* Evaluate and compare ML models for geospatial prediction tasks
* Assess their *spatial* goodness - measuring how well their predictions match the spatial distribution of the ground truth (standard metrics, e.g. MSE, ignore the spatial location; usually just the average error across locations is reported)

How does it work?
* The difference between the spatial distribution of the ground truth observations and the predicted spatial distribution is measured using Optimal Transport, i.e., the **Wasserstein Distance** or Earth Mover's Distance (EMD)
* For a simple explanation, check out the [EMD between signatures](https://en.wikipedia.org/wiki/Earth_mover%27s_distance#EMD_between_signatures). Our method is based on *discrete* OT, where the *sample points* are spatial locations and the *mass* at these points are the observations or predictions. OT finds the *minimal cost transportation matrix*, i.e. the mass that must be transported between each pair of locations to align the predicted with the observed distribution.
* In practice, we provide an interface to other OT python packages for the specific use case on geospatial data. Here, the input is usually just the `observations` at a set of location, the `predictions` and the `cost matrix`. The output is the OT error (a single number) or the optimal transport matrix T.
* Apart from evaluation, OT can be used as a loss function via the Sinkhorn loss. 

What are typical use cases?
* This method is relevant any application where the spatial distribution of the prediction errors matters.
* Typical examples are applications where prediction errors are associated with relocation or reallocation costs. For example, errors in bike sharing demand prediction lead to users relocating to another station that is further away, etc
* Further examples: predicting wildfire spread or glacier retreat, forecasting weather or traffic, estimating poverty or crime rates, etc.

How to use your own data?
* The tutorial notebook provides examples using random values. You can adapt the code to use your own data by replacing the variables such as `locations`, `observations`, `predictions` and `cost_matrix`.
* If you have trouble, open an issue or get in touch!

