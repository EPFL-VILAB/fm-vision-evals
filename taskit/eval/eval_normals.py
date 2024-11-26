import json
from typing import List, Optional, Union

import cvxpy as cp
import numpy as np
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from skimage.util import img_as_float
from skimage.segmentation import slic
from tqdm import tqdm

from taskit.mfm import MFMWrapper
from taskit.utils.data import find_adjacent_segments


def combine_normals(normal_map):
    arr_x, arr_y, arr_z = normal_map[0], normal_map[1], normal_map[2]
    arr_x = np.nan_to_num(arr_x, copy=True, nan=0.0, posinf=None, neginf=None)
    arr_y = np.nan_to_num(arr_y, copy=True, nan=0.0, posinf=None, neginf=None)
    arr_z = np.nan_to_num(arr_z, copy=True, nan=0.0, posinf=None, neginf=None)

    x_rescaled = arr_x * 2 - 1
    y_rescaled = arr_y * 2 - 1
    z_rescaled = arr_z * 2 - 1

    magnitude = np.sqrt(x_rescaled**2 + y_rescaled**2 + z_rescaled**2)
    x_normalized = x_rescaled / magnitude
    y_normalized = y_rescaled / magnitude
    z_normalized = z_rescaled / magnitude

    x_display = (x_normalized + 1) / 2
    y_display = (y_normalized + 1) / 2
    z_display = (z_normalized + 1) / 2

    normal_image = np.stack((x_display, y_display, z_display), axis=-1)
    return normal_image


def compute_correl_metrics(optimal_xs, average_norm, seg_map, gt_normals, file_name):
    pred_images = []
    gt_pixelated = []
    n_points = len(np.unique(seg_map))
    for opt_idx, optimal_x in enumerate(optimal_xs):
        if opt_idx == 1 or opt_idx == 2:
            optimal_x = -optimal_x
        optimal_x = (optimal_x - optimal_x.min()) / (optimal_x.max() - optimal_x.min()) * 255
        optimal_x = optimal_x.astype(int)
        pred_image, gt_pix = np.zeros_like(seg_map), np.zeros_like(seg_map)
        for i, segment in enumerate(np.unique(seg_map)):
            gt_pix[seg_map == segment] = average_norm[segment-1, opt_idx]
            pred_image[seg_map == segment] = optimal_x[i]

        pred_images.append(pred_image)
        gt_pixelated.append(gt_pix)

    norm_gts = [gt_normals[:, :, 0], gt_normals[:, :, 1], gt_normals[:, :, 2]]

    average_norm = np.zeros((n_points, 3))
    for i in range(n_points):
        average_norm[i] = np.mean(gt_normals[seg_map == (i + 1)], axis=0)

    local_correl_tau, local_correl_rho = [], []
    for i in range(3):
        optimal_x_direction = optimal_xs[i]
        if i == 1 or i == 2:
            optimal_x_direction = -optimal_x_direction
        local_correl_tau_direction = kendalltau(optimal_x_direction, average_norm[:, i]).correlation
        local_correl_rho_direction = spearmanr(optimal_x_direction, average_norm[:, i]).correlation
        local_correl_tau.append(local_correl_tau_direction)
        local_correl_rho.append(local_correl_rho_direction)

    global_correl_tau, global_correl_rho = [], []
    for i in range(3):
        pred_image = pred_images[i]
        global_correl_tau_direction = kendalltau(pred_image.flatten(), norm_gts[i].flatten()).correlation
        global_correl_rho_direction = spearmanr(pred_image.flatten(), norm_gts[i].flatten()).correlation
        global_correl_tau.append(global_correl_tau_direction)
        global_correl_rho.append(global_correl_rho_direction)

    return local_correl_tau, local_correl_rho, global_correl_tau, global_correl_rho


def find_relative_normals(normal_orders_preds, seg_pairs, segment_map, smoothness_weight):
    normal_orders = []

    for i, dic in enumerate(normal_orders_preds):
        orders, directions = [], ['right', 'up', 'out']
        for direction in directions:
            ans = dic[direction]
            orders.append(0 if ans == 'red' else 1 if ans == 'blue' else 2)

        normal_orders.append(orders)

    # Eliminate duplicate all_segments, and only keep the corresponding depth_order
    all_segments_unique = np.unique(seg_pairs, axis=0)
    normal_order_segment_map = {}
    normal_order_segment_map = {tuple(segment): normal_orders[i] for i, segment in enumerate(seg_pairs) if tuple(segment) not in normal_order_segment_map}
    normal_order_unique = np.array([normal_order_segment_map[tuple(segment)] for segment in all_segments_unique])

    optimal_xs = []
    adjacency_matrix = find_adjacent_segments(segment_map)
    for i in range(3):
        normal_order = np.array(normal_order_unique)[:, i]

        all_segments_unique_gt = all_segments_unique[normal_order == 0]
        all_segments_unique_lt = all_segments_unique[normal_order == 1]
        all_segments_unique_eq = all_segments_unique[normal_order == 2]

        n_edges = len(all_segments_unique)
        n_points = len(np.unique(seg_pairs))
        A_gt = np.zeros((n_edges, n_points + n_edges))

        for idx, segment in enumerate(all_segments_unique_gt):
            A_gt[idx, segment[0]-1] = 1
            A_gt[idx, segment[1]-1] = -1
            A_gt[idx, n_points + idx] = -1

        A_lt = np.zeros((n_edges, n_points + n_edges))

        for idx, segment in enumerate(all_segments_unique_lt):
            A_lt[idx, segment[0]-1] = -1
            A_lt[idx, segment[1]-1] = 1
            A_lt[idx, n_points + idx] = -1

        A_eq = np.zeros((n_edges, n_points + n_edges))

        for idx, segment in enumerate(all_segments_unique_eq):
            A_eq[idx, segment[0]-1] = 1
            A_eq[idx, segment[1]-1] = -1
            A_eq[idx, n_points + idx] = 0

        adj_pairs = np.argwhere(adjacency_matrix)
        adj_pairs = adj_pairs[adj_pairs[:, 0] < adj_pairs[:, 1]]  # To ensure each pair is only counted once

        # Number of adjacency pairs
        n_adjacency_pairs = len(adj_pairs)
        A_adj = np.zeros((n_adjacency_pairs, n_points))

        for idx, (i, j) in enumerate(adj_pairs):
            A_adj[idx, i] = 1
            A_adj[idx, j] = -1

        # Optimise the function x_gt^T A_gt^T A_gt x_gt + x_lt^T A_lt^T A_lt x_lt
        x = cp.Variable(n_points + n_edges)

        # Define the constraints to fix the last n_edges values of x to 1
        constraints = [x[n_points:] == 1]

        # Define the objective function
        eps_noise = np.eye(n_points + n_edges) * 1e-8
        objective = cp.Minimize(cp.quad_form(x, A_gt.T @ A_gt + eps_noise) + cp.quad_form(x, A_lt.T @ A_lt + eps_noise) + cp.quad_form(x, A_eq.T @ A_eq + eps_noise) + smoothness_weight*cp.quad_form(x[:n_points], A_adj.T @ A_adj + eps_noise[:n_points, :n_points]))

        prob = cp.Problem(objective, constraints)
        result = prob.solve()  # noqa

        # Get the optimal value of x
        optimal_x = x.value[:n_points]
        optimal_xs.append(optimal_x)

    return optimal_xs


@MFMWrapper.register_eval('eval_normals')
def eval_normals(
    predictions: Union[List, str],
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    n_segments: int = 200,
    visualise: bool = False,
    smoothness_weight: float = 5,
):
    """ Returns Kendall's tau and Spearman's rank correlation for normals after reading outputs from 'predictions'

        Args:
            predictions: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
            invalid_files: List of invalid files
            read_from_file: bool, whether to read data_file_names from file
            data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored
            n_segments: int, number of segments to use for SLIC
            visualise: bool, whether to output surface normal maps instead of metrics
            smoothness_weight: float, weight for the smoothness term in the optimization problem

        Returns:
            (If visualise is False)
            avg_accuracy: percentage of correctly predicted pairwise rankings
            avg_local_kendall_tau: float, average local Kendall's tau  (local: ranking correlation of the superpixels wrt ground-truth superpixels)
            avg_local_spearman_rank: float, average local Spearman's rank correlation (ranking of superpixels)
            avg_global_kendall_tau: float, average global Kendall's tau (global: ranking correlation of all the pixels wrt ground-truth pixels)
            avg_global_spearman_rank: float, average global Spearman's rank correlation (ranking of pixels)

            OR

            (If visualise is True)
            normal_maps: List of normal maps normalized to 0-1 (np.ndarray)
    """

    if isinstance(predictions, list):
        outputs = {'data': predictions}
    else:
        with open(predictions, 'r') as f:
            outputs = json.load(f)

    if read_from_file:
        with open(data_file_names) as f:
            rgb_data_files = f.readlines()
    else:
        rgb_data_files = [output['file_name'] for output in outputs['data']]
    rgb_data_files = [file_name for file_name in rgb_data_files if file_name not in invalid_files]  # Remove invalid files

    if not visualise:
        gt_normals = []
        groundtruth = json.load(open('./taskit/utils/metadata/hypersim-normals.json'))  # dict mapping file_name to gt file_path
        for rgb_file in rgb_data_files:
            gt_normals.append(np.array(Image.open(groundtruth[rgb_file])))

    # --Form the predicted images------------------------------------------------
    normal_maps = []
    avg_accuracy, avg_local_kendall_tau, avg_local_spearman_rank, avg_global_kendall_tau, avg_global_spearman_rank = {}, {}, {}, {}, {}
    for direction in ['right', 'up', 'out']:
        avg_accuracy[direction], avg_local_kendall_tau[direction], avg_local_spearman_rank[direction], avg_global_kendall_tau[direction], avg_global_spearman_rank[direction] = [], [], [], [], []
    for file_idx, output_dict in tqdm(enumerate(outputs['data']), disable=visualise):
        if output_dict['file_name'] not in rgb_data_files:
            continue

        rgb_img = Image.open(output_dict['file_name']).convert('RGB')
        img = img_as_float(rgb_img)
        segments = slic(img, n_segments=n_segments, sigma=5)

        # Get the normal orders
        normal_orders_preds = output_dict['normal_orders']
        seg_pairs = output_dict['segment_pairs']
        optimal_xs = find_relative_normals(normal_orders_preds, seg_pairs, segments, smoothness_weight)

        if visualise:
            normal_arrays = []
            for opt_idx, optimal_x in enumerate(optimal_xs):
                if opt_idx == 1 or opt_idx == 2:
                    optimal_x = -optimal_x
                optimal_x = (optimal_x - optimal_x.min()) / (optimal_x.max() - optimal_x.min()) * 255
                optimal_x = optimal_x.astype(int)
                pred_image = np.zeros_like(segments)
                for i, segment in enumerate(np.unique(segments)):
                    pred_image[segments == segment] = optimal_x[i]
                pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min())
                normal_arrays.append(pred_image)

            normal_maps.append(combine_normals(normal_arrays))

        else:
            # --Compute metrics------------------------------------------------
            accuracies = {"right": [], "up": [], "out": []}
            n_points = len(np.unique(segments))
            normal_img = gt_normals[file_idx]

            average_norm = np.zeros((n_points, 3))
            average_stds = np.zeros((n_points, 3))
            for point_idx in range(n_points):
                average_norm[point_idx] = np.mean(normal_img[segments == (point_idx + 1)], axis=0)
                average_stds[point_idx] = np.std(normal_img[segments == (point_idx + 1)], axis=0)

            for seg_idx, (seg1, seg2) in enumerate(seg_pairs):
                for dir_idx, direction in enumerate(['right', 'up', 'out']):
                    if (normal_orders_preds[seg_idx][direction] == 'equal') and (np.abs(average_norm[seg1-1, dir_idx] - average_norm[seg2-1, dir_idx]) <= max(average_stds[seg1-1, dir_idx], average_stds[seg2-1, dir_idx])):
                        accuracies[direction].append(1)
                    elif ((normal_orders_preds[seg_idx][direction] == 'red') and (average_norm[seg1-1, dir_idx] < average_norm[seg2-1, dir_idx]) or (normal_orders_preds[seg_idx][direction] == 'blue') and (average_norm[seg1-1, dir_idx] > average_norm[seg2-1, dir_idx])) and direction != 'right':
                        accuracies[direction].append(1)
                    elif ((normal_orders_preds[seg_idx][direction] == 'red') and (average_norm[seg1-1, dir_idx] > average_norm[seg2-1, dir_idx]) or (normal_orders_preds[seg_idx][direction] == 'blue') and (average_norm[seg1-1, dir_idx] < average_norm[seg2-1, dir_idx])) and direction == 'right':
                        accuracies[direction].append(1)
                    else:
                        accuracies[direction].append(0)
            for direction in ['right', 'up', 'out']:
                accuracies[direction] = sum(accuracies[direction]) / len(accuracies[direction])
                avg_accuracy[direction].append(accuracies[direction])

            local_correl_tau, local_correl_rho, global_correl_tau, global_correl_rho = compute_correl_metrics(optimal_xs, average_norm, segments, normal_img, file_name=rgb_data_files[file_idx].split('/')[-1].split('.')[0])
            for direction in ['right', 'up', 'out']:
                avg_local_kendall_tau[direction].append(local_correl_tau[["right", "up", "out"].index(direction)])
                avg_local_spearman_rank[direction].append(local_correl_rho[["right", "up", "out"].index(direction)])
                avg_global_kendall_tau[direction].append(global_correl_tau[["right", "up", "out"].index(direction)])
                avg_global_spearman_rank[direction].append(global_correl_rho[["right", "up", "out"].index(direction)])

    if visualise:
        return normal_maps

    return_dict = {}
    return_dict['avg_accuracy'] = {direction: np.mean(avg_accuracy[direction]) for direction in ['right', 'up', 'out']}
    return_dict['avg_local_kendall_tau'] = {direction: np.mean(avg_local_kendall_tau[direction]) for direction in ['right', 'up', 'out']}
    return_dict['avg_local_spearman_rank'] = {direction: np.mean(avg_local_spearman_rank[direction]) for direction in ['right', 'up', 'out']}
    return_dict['avg_global_kendall_tau'] = {direction: np.mean(avg_global_kendall_tau[direction]) for direction in ['right', 'up', 'out']}
    return_dict['avg_global_spearman_rank'] = {direction: np.mean(avg_global_spearman_rank[direction]) for direction in ['right', 'up', 'out']}

    return return_dict
