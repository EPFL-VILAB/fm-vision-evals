import json
from typing import List, Optional, Union

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from skimage.util import img_as_float
from skimage.segmentation import slic
from tqdm import tqdm

from taskit.mfm import TextMFMWrapper, ImageMFMWrapper
from taskit.utils.data import find_adjacent_segments, sample_segments


def delta_error(pred_depths, gt_depths):
    deltas = [1.25, 1.25**2, 1.25**3]
    errors = [0, 0, 0]
    total = 0
    for i in range(len(pred_depths)):
        for j in range(len(deltas)):
            errors[j] += np.sum(
                np.maximum(pred_depths[i] / gt_depths[i], gt_depths[i] / pred_depths[i])
                < deltas[j]
            )
        total += np.prod(pred_depths[i].shape)

    return [error / total for error in errors]


def abs_rel_error(pred_depths, gt_depths):
    errors = [
        np.sum(np.abs(pred_depths[i] - gt_depths[i]) / gt_depths[i])
        for i in range(len(pred_depths))
    ]
    totals = [np.prod(pred_depths[i].shape) for i in range(len(pred_depths))]
    return np.sum(errors) / np.sum(totals)


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, axis=(1, 2))
    a_01 = np.sum(mask * prediction, axis=(1, 2))
    a_11 = np.sum(mask, axis=(1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, axis=(1, 2))
    b_1 = np.sum(mask * target, axis=(1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det != 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    # If scale is negative, set it to 0 and shift to target mean
    negative_scale = x_0 < 0
    if np.any(negative_scale):
        x_1[negative_scale] = np.sum(mask * target, axis=(1, 2)) / np.sum(
            mask, axis=(1, 2)
        )
        x_0[negative_scale] = 0

    return x_0, x_1


def compute_correl_metrics(optimal_x, average_depth, seg_map, gt_depth):
    local_correl_tau = kendalltau(optimal_x, average_depth).correlation
    local_correl_rho = spearmanr(optimal_x, average_depth).correlation

    sup_pix_depth = np.zeros_like(seg_map).astype(float)
    for i in range(len(np.unique(seg_map))):
        sup_pix_depth[seg_map == (i + 1)] = optimal_x[i]

    global_correl_tau = kendalltau(
        sup_pix_depth.flatten(), gt_depth.flatten()
    ).correlation
    global_correl_rho = spearmanr(
        sup_pix_depth.flatten(), gt_depth.flatten()
    ).correlation

    return -local_correl_tau, -local_correl_rho, -global_correl_tau, -global_correl_rho


def compute_correl_metrics_dense(dense_map, gt_depth):

    global_correl_tau = kendalltau(dense_map.flatten(), gt_depth.flatten()).correlation
    global_correl_rho = spearmanr(dense_map.flatten(), gt_depth.flatten()).correlation

    return global_correl_tau, global_correl_rho


def find_relative_depths(depth_orders_preds, seg_pairs, segment_map, smoothness_weight):
    depth_orders, logprobs = [], []

    for i, ord_dict in enumerate(depth_orders_preds):
        ans = ord_dict["depth_order"]
        depth_orders.append(0 if ans[0] == "red" else 1)
        if "logprob" in ord_dict:
            logprobs.append(ord_dict["logprob"])

    all_segments_unique = np.unique(seg_pairs, axis=0)
    depth_order_segment_map = {}
    depth_order_segment_map = {
        tuple(segment): depth_orders[i]
        for i, segment in enumerate(seg_pairs)
        if tuple(segment) not in depth_order_segment_map
    }
    depth_order_unique = np.array(
        [depth_order_segment_map[tuple(segment)] for segment in all_segments_unique]
    )

    optimal_x = []
    adjacency_matrix = find_adjacent_segments(segment_map)
    all_segments_unique_gt = all_segments_unique[depth_order_unique == 0]
    all_segments_unique_lt = all_segments_unique[depth_order_unique == 1]

    n_edges = len(all_segments_unique)
    n_points = len(np.unique(seg_pairs))
    A_gt = np.zeros((n_edges, n_points + n_edges))
    W_logprob_gt = np.zeros((n_edges, n_edges))
    for idx, segment in enumerate(all_segments_unique_gt):
        A_gt[idx, segment[0] - 1] = 1
        A_gt[idx, segment[1] - 1] = -1
        A_gt[idx, n_points + idx] = -1
        if len(logprobs) > 0:
            W_logprob_gt[idx, idx] = max(2 * np.exp(logprobs[idx]) - 1, 1e-8)

    A_lt = np.zeros((n_edges, n_points + n_edges))
    W_logprob_lt = np.zeros((n_edges, n_edges))
    for idx, segment in enumerate(all_segments_unique_lt):
        A_lt[idx, segment[0] - 1] = -1
        A_lt[idx, segment[1] - 1] = 1
        A_lt[idx, n_points + idx] = -1
        if len(logprobs) > 0:
            W_logprob_lt[idx, idx] = max(2 * np.exp(logprobs[idx]) - 1, 1e-8)

    adj_pairs = np.argwhere(adjacency_matrix)
    adj_pairs = adj_pairs[
        adj_pairs[:, 0] < adj_pairs[:, 1]
    ]  # To ensure each pair is only counted once

    # Number of adjacency pairs
    n_adjacency_pairs = len(adj_pairs)
    A_adj = np.zeros((n_adjacency_pairs, n_points))

    for idx, (i, j) in enumerate(adj_pairs):
        A_adj[idx, i] = 1
        A_adj[idx, j] = -1

    x = cp.Variable(n_points + n_edges)
    constraints = [x[n_points:] == 1]

    eps_noise = np.eye(n_points + n_edges) * 1e-8

    if len(logprobs) == 0:
        objective = cp.Minimize(
            cp.quad_form(x, A_gt.T @ A_gt + eps_noise)
            + cp.quad_form(x, A_lt.T @ A_lt + eps_noise)
            + smoothness_weight
            * cp.quad_form(
                x[:n_points], A_adj.T @ A_adj + eps_noise[:n_points, :n_points]
            )
        )  # objective function
    else:
        objective = cp.Minimize(
            cp.quad_form(x, A_gt.T @ W_logprob_gt @ A_gt + eps_noise)
            + cp.quad_form(x, A_lt.T @ W_logprob_lt @ A_lt + eps_noise)
            + smoothness_weight
            * cp.quad_form(
                x[:n_points], A_adj.T @ A_adj + eps_noise[:n_points, :n_points]
            )
        )

    prob = cp.Problem(objective, constraints)
    result = prob.solve()  # noqa
    optimal_x = x.value[:n_points]

    return optimal_x


@TextMFMWrapper.register_eval("eval_depth")
def eval_depth(
    predictions: Union[List, str],
    ground_truth: str = None,
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    n_segments: int = 100,
    visualise: bool = False,
    smoothness_weight: float = 10,
):
    """Returns several depth metrics after reading outputs from 'predictions' using depth orders

    Args:
        predictions: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
        ground_truth: str, path to JSON file containing ground truth
        invalid_files: list, list of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored
        n_segments: int, number of segments to use for SLIC
        visualise: bool, whether to output depth maps instead of metrics
        smoothness_weight: float, weight for the smoothness term in the optimization

    Returns:
        (If visualise is False)
        avg_accuracy: percentage of correctly predicted pairwise rankings
        avg_local_kendall_tau: float, average local Kendall's tau (local: ranking correlation of the superpixels wrt ground-truth superpixels)
        avg_local_spearman_rank: float, average local Spearman's rank correlation (ranking of superpixels)
        avg_global_kendall_tau: float, average global Kendall's tau (global: ranking correlation of all the pixels wrt ground-truth pixels)
        avg_global_spearman_rank: float, average global Spearman's rank correlation (ranking of pixels)
        delta_errors: The images are scaled and shifted based on the ground truth. The delta errors (<1.25, <1.25^2, <1.25^3) are computed for the scaled predictions
        abs_rel_errors: The absolute relative errors for the scaled predictions

        OR

        (If visualise is True)
        depth_maps: list of depth maps normalized to 0-1 (np.ndarray)
    """

    if isinstance(predictions, list):
        outputs = {"data": predictions}
    else:
        with open(predictions, "r") as f:
            outputs = json.load(f)

    if read_from_file:
        with open(data_file_names) as f:
            rgb_data_files = f.readlines()
    else:
        rgb_data_files = [output["file_name"] for output in outputs["data"]]
    rgb_data_files = [
        file_name for file_name in rgb_data_files if file_name not in invalid_files
    ]  # Remove invalid files
    rgb_data_files = [file_name.strip() for file_name in rgb_data_files]

    if not visualise:
        euclidean_depth_arrays = []
        groundtruth = json.load(
            open(ground_truth)
        )  # dict mapping file_name to gt file_path
        for rgb_file in [output["file_name"] for output in outputs["data"]]:
            if rgb_file in rgb_data_files:
                euclidean_file = groundtruth[rgb_file]
                euclidean_depth_arrays.append(np.array(Image.open(euclidean_file)))

    # --Form the predicted images------------------------------------------------
    depth_maps, superpixel_relative_imgs = [], []
    (
        avg_accuracy,
        avg_local_kendall_tau,
        avg_local_spearman_rank,
        avg_global_kendall_tau,
        avg_global_spearman_rank,
    ) = ([], [], [], [], [])
    gt_counter = 0
    for output_dict in tqdm(outputs["data"], disable=visualise):
        if output_dict["file_name"] not in rgb_data_files:
            continue

        rgb_img = Image.open(output_dict["file_name"]).convert("RGB")
        img = img_as_float(rgb_img)
        segments = slic(img, n_segments=n_segments, sigma=5)

        depth_orders_preds = output_dict["depth_orders"]
        seg_pairs = output_dict["segment_pairs"]

        optimal_x = find_relative_depths(
            depth_orders_preds, seg_pairs, segments, smoothness_weight
        )

        superpixel_relative_img = np.zeros_like(segments, dtype=np.float32)
        for idx, segment in enumerate(np.unique(segments)):
            superpixel_relative_img[segments == segment] = optimal_x[idx]
        superpixel_relative_imgs.append(-superpixel_relative_img)

        if visualise:
            superpixel_relative_img = -superpixel_relative_img
            superpixel_relative_img = (
                superpixel_relative_img - np.min(superpixel_relative_img)
            ) / (np.max(superpixel_relative_img) - np.min(superpixel_relative_img))
            viridis_img = plt.cm.viridis(superpixel_relative_img)
            depth_maps.append(viridis_img)

        else:
            average_depth = np.zeros(len(np.unique(segments)))
            for seg_idx in range(len(np.unique(segments))):
                average_depth[seg_idx] = np.mean(
                    euclidean_depth_arrays[gt_counter][segments == (seg_idx + 1)]
                )

            acc = 0
            for seg_idx, (seg1, seg2) in enumerate(seg_pairs):
                if (depth_orders_preds[seg_idx]["depth_order"][0] == "red") == (
                    average_depth[seg1 - 1] < average_depth[seg2 - 1]
                ):
                    acc += 1
            avg_accuracy.append(acc / len(seg_pairs))

            # Compute correlation metrics
            local_correl_tau, local_correl_rho, global_correl_tau, global_correl_rho = (
                compute_correl_metrics(
                    optimal_x,
                    average_depth,
                    segments,
                    euclidean_depth_arrays[gt_counter],
                )
            )
            (
                avg_local_kendall_tau,
                avg_local_spearman_rank,
                avg_global_kendall_tau,
                avg_global_spearman_rank,
            ) = (
                avg_local_kendall_tau + [local_correl_tau],
                avg_local_spearman_rank + [local_correl_rho],
                avg_global_kendall_tau + [global_correl_tau],
                avg_global_spearman_rank + [global_correl_rho],
            )
        gt_counter += 1

    # --Compute Depth Metrics------------------------------------------------
    if not visualise:
        scales, shifts = [], []
        for i in range(len(superpixel_relative_imgs)):
            scale, shift = compute_scale_and_shift(
                superpixel_relative_imgs[i].astype(np.float32)[np.newaxis, ...],
                euclidean_depth_arrays[i].astype(np.float32)[np.newaxis, ...],
                np.ones_like(euclidean_depth_arrays[i], dtype=bool)[np.newaxis, ...],
            )
            scales.append(scale)
            shifts.append(shift)
        scaled_imgs = [
            scales[i] * superpixel_relative_imgs[i] + shifts[i]
            for i in range(len(superpixel_relative_imgs))
        ]
        # metrics for scaled images
        delta_errors_scaled = delta_error(scaled_imgs, euclidean_depth_arrays)
        abs_rel_errors_scaled = abs_rel_error(scaled_imgs, euclidean_depth_arrays)

        return {
            "avg_accuracy": np.mean(avg_accuracy),
            "avg_local_kendall_tau": np.mean(avg_local_kendall_tau),
            "avg_local_spearman_rank": np.mean(avg_local_spearman_rank),
            "avg_global_kendall_tau": np.mean(avg_global_kendall_tau),
            "avg_global_spearman_rank": np.mean(avg_global_spearman_rank),
            "delta_errors": delta_errors_scaled,
            "abs_rel_errors": abs_rel_errors_scaled,
        }

    return depth_maps


@ImageMFMWrapper.register_eval("eval_dense_depth")
def eval_dense_depth(
    predictions: Union[List, str],
    ground_truth: str = None,
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    n_segments: int = 100,
):
    """Returns several metrics after reading outputs from 'predictions'. An evaluation function for dense predictions.

    Args:
        predictions: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
        ground_truth: str, path to JSON file containing ground truth
        invalid_files: list, list of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored
        n_segments: int, number of segments to use for SLIC

    Returns:
        (metrics after superpixelation, for consistency with eval_depth)
        avg_accuracy: percentage of correctly predicted pairwise rankings
        avg_local_kendall_tau: float, average local Kendall's tau (local: ranking correlation of the superpixels wrt ground-truth superpixels)
        avg_local_spearman_rank: float, average local Spearman's rank correlation (ranking of superpixels)
        avg_global_kendall_tau: float, average global Kendall's tau (global: ranking correlation of all the pixels wrt ground-truth pixels)
        avg_global_spearman_rank: float, average global Spearman's rank correlation (ranking of pixels)
        delta_errors: The images are scaled and shifted based on the ground truth. The delta errors (<1.25, <1.25^2, <1.25^3) are computed for the scaled predictions
        abs_rel_errors: The absolute relative errors for the scaled predictions

        (metrics computed using the dense predictions)
        avg_dense_kendall_tau: float,
        avg_dense_spearman_rank: float,
        dense_delta_errors: float,
        dense_abs_rel_errors: float
    """

    if isinstance(predictions, list):
        outputs = {"data": predictions}
    else:
        with open(predictions, "r") as f:
            outputs = json.load(f)

    if read_from_file:
        with open(data_file_names) as f:
            rgb_data_files = f.readlines()
    else:
        rgb_data_files = [output["file_name"] for output in outputs["data"]]
    rgb_data_files = [
        file_name for file_name in rgb_data_files if file_name not in invalid_files
    ]  # Remove invalid files
    rgb_data_files = [file_name.strip() for file_name in rgb_data_files]

    euclidean_depth_arrays = []
    groundtruth = json.load(
        open(ground_truth)
    )  # dict mapping file_name to gt file_path
    for rgb_file in [output["file_name"] for output in outputs["data"]]:
        if rgb_file in rgb_data_files:
            euclidean_file = groundtruth[rgb_file]
            euclidean_depth_arrays.append(np.array(Image.open(euclidean_file)))

    file_name_to_depth_image = {
        dic["file_name"]: dic["depth_image"] for dic in outputs["data"]
    }
    depth_maps = [
        np.array(Image.open(file_name_to_depth_image[file_name]).convert("L"))
        for file_name in rgb_data_files
    ]
    # --Form the predicted images------------------------------------------------
    superpixel_relative_imgs = []
    (
        avg_accuracy,
        avg_local_kendall_tau,
        avg_local_spearman_rank,
        avg_global_kendall_tau,
        avg_global_spearman_rank,
        avg_dense_kendall_tau,
        avg_dense_spearman_rank,
    ) = ([], [], [], [], [], [], [])
    gt_counter = 0
    for output_dict in tqdm(outputs["data"]):
        if output_dict["file_name"] not in rgb_data_files:
            continue

        rgb_img = Image.open(output_dict["file_name"]).convert("RGB")
        img = img_as_float(rgb_img)
        segments = slic(img, n_segments=n_segments, sigma=5)

        # generate 200 pairs randomly
        seg_pairs = sample_segments(segments, min_samples=200, shuffle=True, seed=25)

        optimal_x = []
        superpixel_relative_img = np.zeros_like(segments, dtype=np.float32)
        for idx, segment in enumerate(np.unique(segments)):
            optimal_x_value = -np.mean(depth_maps[gt_counter][segments == segment])
            superpixel_relative_img[segments == segment] = -optimal_x_value
            optimal_x.append(optimal_x_value)

        superpixel_relative_imgs.append(superpixel_relative_img)

        average_depth = np.zeros(len(np.unique(segments)))
        for seg_idx in range(len(np.unique(segments))):
            average_depth[seg_idx] = np.mean(
                euclidean_depth_arrays[gt_counter][segments == (seg_idx + 1)]
            )

        acc = 0
        for seg_idx, (seg1, seg2) in enumerate(seg_pairs):
            pred_depth_1 = np.mean(depth_maps[gt_counter][segments == seg1])
            pred_depth_2 = np.mean(depth_maps[gt_counter][segments == seg2])
            if (pred_depth_1 < pred_depth_2) == (
                average_depth[seg1 - 1] < average_depth[seg2 - 1]
            ):
                acc += 1
        avg_accuracy.append(acc / len(seg_pairs))

        # Compute correlation metrics
        local_correl_tau, local_correl_rho, global_correl_tau, global_correl_rho = (
            compute_correl_metrics(
                optimal_x,
                average_depth,
                segments,
                euclidean_depth_arrays[gt_counter],
            )
        )
        dense_correl_tau, dense_correl_rho = compute_correl_metrics_dense(
            depth_maps[gt_counter], euclidean_depth_arrays[gt_counter]
        )
        (
            avg_local_kendall_tau,
            avg_local_spearman_rank,
            avg_global_kendall_tau,
            avg_global_spearman_rank,
            avg_dense_kendall_tau,
            avg_dense_spearman_rank,
        ) = (
            avg_local_kendall_tau + [local_correl_tau],
            avg_local_spearman_rank + [local_correl_rho],
            avg_global_kendall_tau + [global_correl_tau],
            avg_global_spearman_rank + [global_correl_rho],
            avg_dense_kendall_tau + [dense_correl_tau],
            avg_dense_spearman_rank + [dense_correl_rho],
        )
        gt_counter += 1

    # --Compute Depth Metrics------------------------------------------------
    scales, shifts = [], []
    dense_scales, dense_shifts = [], []
    for i in range(len(superpixel_relative_imgs)):
        scale, shift = compute_scale_and_shift(
            superpixel_relative_imgs[i].astype(np.float32)[np.newaxis, ...],
            euclidean_depth_arrays[i].astype(np.float32)[np.newaxis, ...],
            np.ones_like(euclidean_depth_arrays[i], dtype=bool)[np.newaxis, ...],
        )
        dense_scale, dense_shift = compute_scale_and_shift(
            depth_maps[i].astype(np.float32)[np.newaxis, ...],
            euclidean_depth_arrays[i].astype(np.float32)[np.newaxis, ...],
            np.ones_like(euclidean_depth_arrays[i], dtype=bool)[np.newaxis, ...],
        )
        scales.append(scale)
        shifts.append(shift)
        dense_scales.append(dense_scale)
        dense_shifts.append(dense_shift)
    scaled_imgs = [
        scales[i] * superpixel_relative_imgs[i] + shifts[i]
        for i in range(len(superpixel_relative_imgs))
    ]
    dense_scaled_imgs = [
        dense_scales[i] * depth_maps[i] + dense_shifts[i]
        for i in range(len(depth_maps))
    ]
    delta_errors_scaled = delta_error(scaled_imgs, euclidean_depth_arrays)
    abs_rel_errors_scaled = abs_rel_error(scaled_imgs, euclidean_depth_arrays)
    dense_delta_errors_scaled = delta_error(dense_scaled_imgs, euclidean_depth_arrays)
    dense_abs_rel_errors_scaled = abs_rel_error(
        dense_scaled_imgs, euclidean_depth_arrays
    )

    return {
        "avg_accuracy": np.mean(avg_accuracy),
        "avg_local_kendall_tau": np.mean(avg_local_kendall_tau),
        "avg_local_spearman_rank": np.mean(avg_local_spearman_rank),
        "avg_global_kendall_tau": np.mean(avg_global_kendall_tau),
        "avg_global_spearman_rank": np.mean(avg_global_spearman_rank),
        "delta_errors": delta_errors_scaled,
        "abs_rel_errors": abs_rel_errors_scaled,
        "avg_dense_kendall_tau": np.mean(avg_dense_kendall_tau),
        "avg_dense_spearman_rank": np.mean(avg_dense_spearman_rank),
        "dense_delta_errors": dense_delta_errors_scaled,
        "dense_abs_rel_errors": dense_abs_rel_errors_scaled,
    }
