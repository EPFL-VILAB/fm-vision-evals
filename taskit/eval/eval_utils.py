import contextlib
import json
import os
from typing import List, Dict

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def get_gt(img_pths: List[str], groundtruth) -> List[Dict[str, int]]:
    """Returns ground truth bounding boxes (x1, y1, x2, y2, iscrowd) for the given image.
    The groundtruth has to be stored in the standard COCO annotations format.
    """

    img_idxs = [int(img_pth.split("/")[-1].split(".")[0]) for img_pth in img_pths]
    category_id_2_name = {ann["id"]: ann["name"] for ann in groundtruth["categories"]}
    all_gt_boxes = [{} for _ in range(len(img_pths))]

    for ann in groundtruth["annotations"]:
        if ann["image_id"] in img_idxs:
            idx = img_idxs.index(ann["image_id"])
            class_name = category_id_2_name[ann["category_id"]]
            if class_name not in all_gt_boxes[idx]:
                all_gt_boxes[idx][class_name] = []
                x, y, w, h = ann["bbox"]
                x1, y1 = x, y
                x2, y2 = x + w, y + h
            all_gt_boxes[idx][class_name].extend([x1, y1, x2, y2, ann["iscrowd"]])

    return all_gt_boxes


def get_map(
    outputs: dict, rgb_data_files: list, ground_truth: str, verbose: bool = True
):
    """Returns the mAP @0.1, 0.5, 0.75 and 0.5:0.95 for the given output file."""

    groundtruth = json.load(
        open(ground_truth)
    )  # json file, in the same format as coco/annotations/instances_val2017.json
    category_id_2_name = {ann["id"]: ann["name"] for ann in groundtruth["categories"]}
    name_2_category_id = {
        name: category_id for category_id, name in category_id_2_name.items()
    }

    pred_list = outputs["data"]
    dataset = {"info": {}, "licenses": [], "images": [], "annotations": []}

    predictions, category, image_id = [], [], 0
    gts = get_gt(rgb_data_files, groundtruth)

    for i, image_file in tqdm(enumerate(rgb_data_files)):
        file_name = os.path.basename(image_file)
        img = Image.open(image_file)
        width, height = img.size

        image_dict = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        }  # Add the image to the COCO dataset
        dataset["images"].append(image_dict)

        gt_boxes = gts[i]

        # - Add gt_bboxes to dataset-----------------------------------------------------------------
        for name, bbox in gt_boxes.items():
            category_id = name_2_category_id[name]
            x1, y1, x2, y2, iscrowd = bbox
            width = x2 - x1
            height = y2 - y1

            ann_dict = {
                "id": len(dataset["annotations"]) + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": iscrowd,
                "score": 1,
            }

            if {"id": category_id, "name": name} not in category:
                category.append({"id": category_id, "name": name})

            dataset["annotations"].append(ann_dict)

        # - Add pred_bboxes to dataset-----------------------------------------------------------------
        pred_dict = pred_list[i]
        scores = pred_dict["scores"]
        for pred_bbox_idx, (name, bbox) in enumerate(pred_dict.items()):
            if name == "file_name" or name not in name_2_category_id:
                continue

            category_id = name_2_category_id[name]
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            width, height = x2 - x1, y2 - y1
            ann_dict = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": scores[pred_bbox_idx],
            }

            predictions.append(ann_dict)

        image_id += 1

    dataset["categories"] = category

    # - Evaluate-----------------------------------------------------------------
    rand_n = np.random.randint(0, 1000)
    with open(f"dataset_{rand_n}.json", "w") as f:
        json.dump(dataset, f)
    with open(f"predictions_{rand_n}.json", "w") as f:
        json.dump(predictions, f)

    gt = COCO(f"dataset_{rand_n}.json")
    dt = gt.loadRes(f"predictions_{rand_n}.json")
    eval = COCOeval(gt, dt, iouType="bbox")

    results = []
    for iou in [0.1, *[0.5 + 0.05 * i for i in range(10)]]:
        eval.params.iouThrs = [iou]
        if verbose:
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                eval.evaluate()
                eval.accumulate()
                eval.summarize()

        results.append(eval.stats[0])

    # delete json files
    if os.path.exists(f"dataset_{rand_n}.json"):
        os.remove(f"dataset_{rand_n}.json")
    if os.path.exists(f"predictions_{rand_n}.json"):
        os.remove(f"predictions_{rand_n}.json")

    map_01, map_05, map_075, map_05_095 = (
        results[0],
        results[1],
        results[6],
        np.mean(results[1:]),
    )
    return map_01, map_05, map_075, map_05_095
