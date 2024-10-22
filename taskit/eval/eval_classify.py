import json
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from taskit.mfm import MFMWrapper
from taskit.utils.data_constants import IMAGENET_LABELS


@MFMWrapper.register_eval('eval_classify')
def eval_classify(
    output_file: Union[List, str],
    invalid_files: list = [],
    read_from_file: bool = False,
    data_file_names: Optional[str] = None,
    dataset: str = 'imagenet',
    labels: list = IMAGENET_LABELS
) -> Dict[str, float]:
    """ Returns top-1 accuracy after reading outputs from 'output_file'

        Args:
            output_file: Union[Dict, str], output file containing the model predictions
            data_file_names: str, path to file containing all the data files
            invalid_files: list, list of invalid files
            read_from_file: bool, whether to read data_file_names from file
            dataset: str, dataset used for evaluation

        Returns:
            accuracy: float, top-1 accuracy
    """

    valid_datasets = ['imagenet-r', 'imagenet-robustbench-3dcc', 'imagenet-robustbench-2dcc', 'imagenet-sketch', 'imagenet-v2', 'imagenet']

    if dataset in valid_datasets:
        groundtruth = json.load(open(f'./taskit/utils/metadata/{dataset}.json'))  # dict mapping file_name to label
    else:
        raise ValueError(f"Dataset {dataset} not supported by eval_classify")

    if isinstance(output_file, list):
        outputs = {'data': output_file}
    else:
        with open(output_file, 'r') as f:
            outputs = json.load(f)

    acc = 0
    # read files in data_file_names
    if read_from_file:
        with open(data_file_names) as f:
            data_files = f.readlines()
    else:
        data_files = [output['file_name'] for output in outputs['data']]

    # Remove invalid files
    data_files = [file_name for file_name in data_files if file_name not in invalid_files]

    for dic in tqdm(outputs['data']):
        if dic['file_name'] not in data_files:
            continue
        if dic['class'].strip() == labels[groundtruth[dic['file_name']]]:
            acc += 1

    acc /= len(data_files)
    return {"accuracy": acc}
