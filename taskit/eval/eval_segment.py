import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from taskit.mfm import MFMWrapper
from taskit.utils.data import flood_superpixels, colorize
from taskit.utils.data_constants import COCO_SEMSEG_LABELS, COCO_COLOR_MAP, COCO_LABEL_2_ID


def process_single_image(args):
    all_pred, color_map, label_to_id, rgb_data_file, n_segments, visualise, ignore_index = args
    img = Image.open(rgb_data_file)
    pred_array = np.array(flood_superpixels(img, all_pred, label_to_id, n_segments, ignore_index))
    if visualise:
        pred_img = colorize(pred_array, label_to_id, color_map)
        return pred_img
    return pred_array


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=float)
    total_area_union = np.zeros((num_classes, ), dtype=float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=float)
    total_area_label = np.zeros((num_classes, ), dtype=float)
    for i in tqdm(range(num_imgs)):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, _, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


@MFMWrapper.register_eval('eval_segment')
def eval_segment(
    predictions: Union[List, str],
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    n_segments: int = 400,
    labels: list = COCO_SEMSEG_LABELS,
    color_map: Dict[str, list] = COCO_COLOR_MAP,
    label_to_id: Dict[str, int] = COCO_LABEL_2_ID,
    visualise: bool = False,
    n_threads: int = 4,
    ignore_index: int = 255,
):
    """Returns pixel accuracy and mIoU after reading outputs from 'predictions'

    Args:

        predictions: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
        invalid_files: list, list of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored
        n_segments: int, number of segments to use for SLIC
        labels: list, list of segment labels
        color_map: dict, mapping from segment labels to RGB colors
        label_to_id: dict, mapping from segment labels to segment ids (for evaluation). The ID 'ignore_idx' is reserved for invalid segments
        visualise: bool, whether to output segmentation visualisations instead of metrics
        n_threads: int, number of threads to use for processing images
        ignore_index: int, index to ignore in evaluation

    Returns:
        (If visualise is False)
        eval_metrics: dict containing pixel accuracy and mIoU

        OR

        (If visualise is True)
        seg_maps: List of segmentation maps
    """
    for label in labels:
        if label not in color_map:
            print(f"Warning: {label} not present in color map. Adding with black color.")
            color_map[label] = [0, 0, 0]

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
        gt_arrays = []
        groundtruth = json.load(open('./taskit/utils/metadata/coco-segment.json'))  # dict mapping file_name to gt segmentation file_path
        for file_name in rgb_data_files:
            gt_arrays.append(np.array(Image.open(groundtruth[file_name])))
    else:
        label_to_id = {cls: i for i, cls in enumerate(labels)}

    # --Form the predicted images------------------------------------------------
    all_preds = []
    for file_idx, output_dict in enumerate(outputs['data']):
        if output_dict['file_name'] not in rgb_data_files:
            continue

        output_dict = dict(sorted(output_dict.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')))  # sort by int of key
        preds = []
        for j, (k, v) in enumerate(output_dict.items()):
            if k != 'file_name':
                preds.append(v)
        all_preds.append(preds)

    n_images = len(outputs['data'])
    pred_arrays = [None] * n_images
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_index = {
            executor.submit(process_single_image, (all_pred, color_map, label_to_id, rgb_data_files[i], n_segments, visualise, ignore_index)): i
            for i, all_pred in enumerate(all_preds)
        }
        display_pbar = not visualise
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), disable=not display_pbar):
            idx = future_to_index[future]
            pred_arrays[idx] = future.result()

    if visualise:
        return pred_arrays

    pixel_acc, _, ious = mean_iou(pred_arrays, gt_arrays, len(labels), ignore_index)
    mIoU = np.nanmean(ious)
    return {"pixel_accuracy": pixel_acc, "mIoU": mIoU}
