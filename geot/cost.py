import numpy as np
import argparse
import collections
from scipy.spatial.distance import cdist


def space_cost_matrix(coords1, coords2=None, speed_factor=None, scale_function=None):
    """
    coords1, coords2: spatial coordinates (projected, distances in m) of shape (N, 2)
        if coords_2 is none, we take the pairwise distances of all locations in coords1
    speed_factor: speed (in km/h) for converting distances to time
    scale_function: function to scale the resulting matrix, e.g.
        lambda x: x**2
    """
    if coords2 is None:
        coords2 = coords1
    dist_matrix = cdist(coords1, coords2)

    # convert space to time (in h)
    if speed_factor is not None:
        time_matrix = (dist_matrix / 1000) / speed_factor
    else:
        time_matrix = dist_matrix

    # apply scaling function (e.g. **p or applying cutoff)
    if scale_function is not None:
        time_matrix = scale_function(time_matrix)
    return time_matrix


def spacetime_cost_matrix(
    time_matrix,
    time_steps=3,
    forward_cost=0,
    backward_cost=1,
):
    """
    Design a space-time cost matrix that quantifies the cost across space and time

    Args:
        dist_matrix: pairwise distances in m
        forward_cost: cost for using demand that was originally allocated for the
            preceding timestep (usually low) - in hours
        backward_cost: cost for using demand that was allocated for the next timestep - in hours
    Returns:
        2D Matrix with space-time costs, of shape
            (time_matrix.shape[0] * time_step x time_matrix.shape[0] * time_step)
        Cell i,j is the cost from timeslot=i//nr_stations and station=i%nr_stations
            to timeslot=j//nr_stations and station=j%nr_stations.
    """
    assert (
        time_matrix.shape[0] == time_matrix.shape[1]
    ), "only quadratic matrix supported atm for space_time_cost"
    nr_stations = len(time_matrix)

    final_cost_matrix = np.zeros((time_steps * nr_stations, time_steps * nr_stations))
    for t_pred in range(time_steps):
        for t_gt in range(time_steps):
            start_x, end_x = (t_pred * nr_stations, (t_pred + 1) * nr_stations)
            start_y, end_y = (t_gt * nr_stations, (t_gt + 1) * nr_stations)
            if t_pred > t_gt:
                waiting_time = (t_pred - t_gt) * backward_cost
                final_cost_matrix[start_x:end_x, start_y:end_y] = np.maximum(
                    time_matrix, waiting_time * np.ones(time_matrix.shape)
                )
            else:
                waiting_time = (t_gt - t_pred) * forward_cost
                final_cost_matrix[start_x:end_x, start_y:end_y] = np.maximum(
                    time_matrix, waiting_time * np.ones(time_matrix.shape)
                )
    return final_cost_matrix
