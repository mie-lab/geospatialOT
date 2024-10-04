# GEOT: Optimal transport for evaluating geospatial predictions

This repo contains easy-to-use code for evaluating spatiotemporal predictions with Optimal Transport. Assumptions:
* Given is a fixed set of locations, and a cost matrix (pairwise costs between locations)
* At these locations, there are observations and predictions, e.g. generated with an ML model
* OT (or the earth mover's distance) should be computed between predicted spatial distribution and true spatial distribution of the observations.

![Alt text](assets/overview.png)

## Install

```
cd geospatialOT
conda create -n geot_env
conda activate geot_env
pip install .
```

If you have Cuda available, make sure to install [torch](https://pytorch.org/get-started/locally/) with GPU support instead. 

## Tutorial

Check out our [tutorial](tutorial.ipynb) to get started with a simple example.