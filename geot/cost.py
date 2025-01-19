import numpy as np
import argparse
import collections
from scipy.spatial.distance import cdist

QUADRATIC_TIME = 0.1  # at 0.1h, so at 6min, the perceived time is higher than
# the actual time (quadratic function at (1,1))


def space_cost_matrix(coords1, coords2=None, speed_factor=None, scale_function=None):
    """
    coords: spatial coordinates (projected, distances in m)
    speed_factor: relocation speed of users (in km/h)
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

    # convert to perceived time
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
    Cell i,j is the cost from timeslot=i//nr_stations and station=i%nr_stations
    to timeslot=j//nr_stations and station=j%nr_stations

    dist_matrix: pairwise distances in m
    forward_cost: cost for using demand that was originally allocated for the
    preceding timestep (usually low) - in hours
    backward_cost: cost for using demand that was allocated for the next timestep - in hours
    """
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
