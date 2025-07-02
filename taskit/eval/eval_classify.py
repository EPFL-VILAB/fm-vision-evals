import json
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from taskit.mfm import TextMFMWrapper
from taskit.utils.data_constants import IMAGENET_LABELS


@TextMFMWrapper.register_eval("eval_classify")
def eval_classify(
    predictions: Union[List, str],
    ground_truth: str = None,
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    labels: list = IMAGENET_LABELS,
) -> Dict[str, float]:
    """Returns top-1 accuracy after reading outputs from 'predictions'

    Args:
        predictions: Union[List, str], Path to output JSON file containing the model predictions, or a list of dictionaries with the model predictions
        ground_truth: str, path to JSON file containing ground truth labels
        invalid_files: list, list of invalid files
        read_from_file: bool, whether to read data_file_names from file
        data_file_names: str, path to file containing all the data files
        labels: list, list of labels

    Returns:
        accuracy: float, top-1 accuracy
    """

    groundtruth = json.load(open(ground_truth))  # dict mapping file_name to label

    if isinstance(predictions, list):
        outputs = {"data": predictions}
    else:
        with open(predictions, "r") as f:
            outputs = json.load(f)

    acc = 0
    # read files in data_file_names
    if read_from_file:
        with open(data_file_names) as f:
            data_files = f.readlines()
    else:
        data_files = [output["file_name"] for output in outputs["data"]]

    # Remove invalid files
    data_files = [
        file_name for file_name in data_files if file_name not in invalid_files
    ]
    data_files = [file_name.strip() for file_name in data_files]

    for dic in tqdm(outputs["data"]):
        if dic["file_name"].strip() not in data_files:
            continue
        if dic["class"].strip() == labels[groundtruth[dic["file_name"]]]:
            acc += 1

    acc /= len(data_files)
    return {"accuracy": acc}
