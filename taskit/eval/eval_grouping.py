import json
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from taskit.mfm import TextMFMWrapper, ImageMFMWrapper


predefined_colors_sam = [
    [255, 105, 97],  # Coral Pink
    [97, 168, 255],  # Light Blue
    [178, 255, 102],  # Lime Green
    [255, 179, 71],  # Mango
    [163, 122, 255],  # Lavender
    [255, 117, 224],  # Hot Pink
    [82, 236, 255],  # Cyan
    [255, 243, 92],  # Lemon Yellow
    [255, 133, 82],  # Tangerine
    [130, 255, 213],  # Mint
    [255, 92, 214],  # Magenta
    [103, 255, 169],  # Spring Green
    [255, 214, 102],  # Marigold
    [186, 104, 255],  # Purple
    [255, 92, 92],  # Vermilion
    [79, 255, 176],  # Aquamarine
]


def get_deterministic_color(file_name: str) -> list:

    file_name = file_name.split("/")[-1]  # Get the file name from the path
    # for paper figures
    hardcoded_colors = {
        "000000017029.jpg": [255, 214, 102],  # Marigold
        "000000555705.jpg": [255, 214, 102],  # Marigold
        "000000029393.jpg": [255, 92, 214],  # Magenta
        "000000474164.jpg": [255, 243, 92],  # Lemon Yellow
        "000000572408.jpg": [255, 179, 71],  # Mango
        "000000002149.jpg": [255, 133, 82],  # Tangerine
        "000000064499.jpg": [186, 104, 255],  # Purple
        "000000067896.jpg": [79, 255, 176],  # Aquamarine
    }
    if file_name in hardcoded_colors:
        return hardcoded_colors[file_name]

    # Deterministic fallback based on alphanumeric sum
    alnum_sum = sum(ord(c) for c in file_name if c.isalnum())
    return predefined_colors_sam[alnum_sum % len(predefined_colors_sam)]


@TextMFMWrapper.register_eval("eval_group")
def eval_group(
    predictions: Union[List, str],
    ground_truth: str = None,
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    visualise: bool = False,
    overlay_on_same_image: bool = False,
):
    """
    Finds mIoU of predicted masks wrt ground truth masks.

    Args:
        predictions: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
        ground_truth: str, path to JSON file containing ground truth
        invalid_files: list, list of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored
        visualise: bool, whether to output images with masks overlaid instead of metrics
        overlay_on_same_image: bool, whether to overlay masks on the same image or separate images (if there are multiple masks per image)

    Returns:
        (If visualise is False)
        float: mIoU score

        OR

        (If visualise is True)
        mask_list: list of images with overlaid masks (np.ndarray)
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
        groundtruth = json.load(open(ground_truth))

    all_imgs, gt_mious = [], []
    for output_dict in outputs["data"]:
        fn = output_dict["file_name"]
        if fn not in rgb_data_files:
            continue

        # sort output_dict by key
        output_dict = dict(
            sorted(
                output_dict.items(),
                key=lambda item: item[0] if isinstance(item[0], int) else float("inf"),
            )
        )
        img = Image.open(fn).convert("RGB")
        img_array = np.array(img)
        colored_masks, masks = [], []
        for k, v in output_dict.items():
            if k == "file_name":
                continue
            if visualise:
                color = get_deterministic_color(fn)
                colored_mask = np.zeros_like(img_array)
                colored_mask[np.array(v["prediction"]) > 0] = color
                colored_masks.append(colored_mask)
                masks.append(np.array(v["prediction"]) > 0)
            else:
                gt_map = np.array(groundtruth[fn][k]["gt"])
                pred_map = np.array(v["prediction"])
                gt_miou = np.sum(gt_map & pred_map) / np.sum(gt_map | pred_map)
                gt_mious.append(gt_miou)

        if visualise:
            if overlay_on_same_image:
                overlayed_img = img_array.copy()
                for cm_idx, cm in enumerate(colored_masks):
                    overlayed_img = np.where(
                        masks[cm_idx][..., np.newaxis],
                        (0.4 * overlayed_img + 0.6 * cm).astype(np.uint8),
                        overlayed_img,
                    )
                all_imgs.append(overlayed_img)

            else:
                for cm_idx, cm in enumerate(colored_masks):
                    overlayed_img = np.where(
                        masks[cm_idx][..., np.newaxis],
                        (0.4 * img_array + 0.6 * cm).astype(np.uint8),
                        img_array,
                    )
                    all_imgs.append(overlayed_img)

    if not visualise:
        print(f"mIoU: {np.mean(gt_mious)}")
        return {"mIoU": np.mean(gt_mious)}

    return all_imgs


@ImageMFMWrapper.register_eval("eval_dense_group")
def eval_dense_group(
    predictions: Union[List, str],
    ground_truth: str = None,
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
):
    """
    Finds mIoU of predicted (dense) masks wrt ground truth masks. Takes an image with the predicted object marked in red, and extracts the mask from it.

    Args:
        predictions: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
        ground_truth: str, path to JSON file containing ground truth
        invalid_files: list, list of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored

    Returns:
        float: mIoU score
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
    groundtruth = json.load(open(ground_truth))

    gt_mious, per_file_mious = [], []
    for output_dict in outputs["data"]:
        file_miou = {"mIoU": [], "file_name": output_dict["file_name"]}
        fn = output_dict["file_name"]
        if fn not in rgb_data_files:
            continue

        # sort output_dict by key
        output_dict = dict(
            sorted(
                output_dict.items(),
                key=lambda item: item[0] if isinstance(item[0], int) else float("inf"),
            )
        )
        for k, v in output_dict.items():
            if k == "file_name":
                continue

            gt_map = np.array(groundtruth[fn][k]["gt"])

            # --Logic for extracting predicted mask from the image-----------------------
            pred_img = cv2.imread(v["prediction"])
            hsv = cv2.cvtColor(pred_img, cv2.COLOR_BGR2HSV)

            # wrap-around means red lives in two places on the hue wheel
            lower1 = np.array([0, 80, 50])  # 0-10°
            upper1 = np.array([3, 255, 255])
            lower2 = np.array([177, 80, 50])  # 160-180°
            upper2 = np.array([180, 255, 255])

            pred_map = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(
                hsv, lower2, upper2
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            pred_map = cv2.morphologyEx(
                pred_map, cv2.MORPH_CLOSE, kernel, 2
            )  # fill small holes
            pred_map = cv2.morphologyEx(
                pred_map, cv2.MORPH_OPEN, kernel, 1
            )  # drop tiny specks

            num, labels, stats, _ = cv2.connectedComponentsWithStats(pred_map, 8)
            if num > 1:  # 0 is background
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                pred_map = np.uint8(labels == largest) * 255

            # print numpy data type
            pred_map = np.array(pred_map > 0)
            gt_map = np.array(gt_map > 0)
            gt_miou = np.sum(gt_map & pred_map) / np.sum(gt_map | pred_map)
            gt_mious.append(gt_miou)
            file_miou["mIoU"].append(gt_miou)
        per_file_mious.append(file_miou)

    return {"mIoU": np.mean(gt_mious)}
