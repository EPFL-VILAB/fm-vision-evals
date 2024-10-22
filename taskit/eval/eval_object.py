import json
import os
from typing import List, Optional, Union

import cv2
from PIL import Image, ImageDraw, ImageFont

from taskit.mfm import MFMWrapper
from taskit.eval.eval_utils import get_map


def draw_and_save_bounding_boxes(image_path, pred_dict):

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans-Bold.ttf'), 15)
    label_positions = []
    for idx, (cls, bbox) in enumerate(pred_dict.items()):
        if cls == 'file_name':
            continue

        bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]] if len(bbox) == 2 else bbox[:4]
        x1, y1, x2, y2 = [coord for coord in bbox]

        img_width, img_height = img.size
        x1, x2 = max(0, x1), min(img_width - 1, x2)
        y1, y2 = max(0, y1), min(img_height - 1, y2)

        color = 'red'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        text_bbox = draw.textbbox((0, 0), cls, font=font)
        label_width = text_bbox[2] - text_bbox[0]
        label_height = text_bbox[3] - text_bbox[1]

        label_x, label_y = x1, y1
        text_padding = 4
        text_background = [(label_x, label_y), (label_x + label_width + text_padding * 2, label_y + label_height + text_padding * 2)]
        draw.rectangle(text_background, fill=color)

        draw.text((label_x + text_padding, label_y + text_padding), cls, font=font, fill='white')
        label_positions.append((label_x, label_y))

    return img


@MFMWrapper.register_eval('eval_detect')
def eval_detect(
    output_file: Union[List, str],
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    visualise: bool = False,
    verbose: bool = True
):
    """Returns mAP@0.1, 0.5, 0.75 and 0.5:0.95:0.05 for object detection.
    Only works for single instance detection.

    Args:
        output_file: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
        invalid_files: List of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files. If read_from_file is False, this is ignored
        visualise: bool, whether to output images with detected bounding boxes instead of metrics

    Returns:
        (If visualise is False)
        mAP@0.1, 0.5, 0.75 and 0.5:0.95:0.05

        OR

        (If visualise is True)
        bbox_imgs: List of PIL Images with bounding boxes drawn
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

    if visualise:
        bbox_imgs = []
        for output in outputs['data']:
            if output['file_name'] in rgb_data_files:
                bbox_img = draw_and_save_bounding_boxes(output['file_name'], output['coords'])
                bbox_imgs.append(bbox_img)
        return bbox_imgs

    else:
        pred_list = {'data': [{**pred_dict["coords"], 'file_name': pred_dict['file_name'], 'scores': pred_dict['scores']} for pred_dict in outputs['data']]}
        mAP = get_map(pred_list, rgb_data_files, verbose=verbose)
        return {"mAP@0.1": mAP[0], "mAP@0.5": mAP[1], "mAP@0.75": mAP[2], "mAP@0.5:0.95:0.05": mAP[3]}
