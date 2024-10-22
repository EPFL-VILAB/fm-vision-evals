import json
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from taskit.mfm import MFMWrapper


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
    [255, 92, 92],   # Vermilion
    [79, 255, 176],  # Aquamarine
]


@MFMWrapper.register_eval('eval_group')
def eval_group(
    output_file: Union[List, str],
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    n_segments: int = 400,
    visualise: bool = False,
    overlay_on_same_image: bool = False,
):
    """
    Finds mIoU of predicted masks wrt ground truth masks.

    Args:
        output_file: Union[List, str], output file containing the model predictions
        invalid_files: list, list of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored
        n_segments: int, number of segments to use for SLIC
        visualise: bool, whether to output images with masks overlaid instead of metrics
        overlay_on_same_image: bool, whether to overlay masks on the same image or separate images (if there are multiple masks per image)

    Returns:
        (If visualise is False)
        float: mIoU score

        OR

        (If visualise is True)
        mask_list: list of images with overlaid masks
    """

    if isinstance(output_file, list):
        outputs = {'data': output_file}
    else:
        with open(output_file, 'r') as f:
            outputs = json.load(f)

    if read_from_file:
        with open(data_file_names) as f:
            rgb_data_files = f.readlines()
    else:
        rgb_data_files = [output['file_name'] for output in outputs['data']]
    rgb_data_files = [file_name for file_name in rgb_data_files if file_name not in invalid_files]  # Remove invalid files

    if not visualise:
        groundtruth = json.load(open('./taskit/utils/metadata/coco-group.json'))

    all_imgs, gt_mious = [], []
    for file_idx, output_dict in enumerate(outputs['data']):
        fn = output_dict['file_name']
        if fn not in rgb_data_files:
            continue

        # sort output_dict by key
        output_dict = dict(sorted(output_dict.items(), key=lambda item: item[0] if isinstance(item[0], int) else float('inf')))
        img = Image.open(fn).convert('RGB')
        img_array = np.array(img)
        colored_masks, masks = [], []
        for k, v in output_dict.items():
            if k == 'file_name':
                continue
            if visualise:
                color = predefined_colors_sam[np.random.randint(len(predefined_colors_sam))]
                colored_mask = np.zeros_like(img_array)
                colored_mask[np.array(v['prediction']) > 0] = color
                colored_masks.append(colored_mask)
                masks.append(np.array(v['prediction']) > 0)
            else:
                gt_map = np.array(groundtruth[fn][k]['gt'])
                pred_map = np.array(v['prediction'])
                gt_miou = np.sum(gt_map & pred_map) / np.sum(gt_map | pred_map)
                gt_mious.append(gt_miou)

        if visualise:
            if overlay_on_same_image:
                overlayed_img = img_array.copy()
                for cm_idx, cm in enumerate(colored_masks):
                    overlayed_img = np.where(masks[cm_idx][..., np.newaxis], (0.4 * overlayed_img + 0.6 * cm).astype(np.uint8), overlayed_img)
                all_imgs.append(overlayed_img)

            else:
                for cm_idx, cm in enumerate(colored_masks):
                    overlayed_img = np.where(masks[cm_idx][..., np.newaxis], (0.4 * img_array + 0.6 * cm).astype(np.uint8), img_array)
                    all_imgs.append(overlayed_img)

    if not visualise:
        print(f"mIoU: {np.mean(gt_mious)}")
        return {'mIoU': np.mean(gt_mious)}

    return all_imgs
