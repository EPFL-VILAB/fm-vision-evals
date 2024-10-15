import json
from typing import Dict, List, Optional, Tuple

from PIL import Image

from taskit.mfm import MFMWrapper
from taskit.prompts import full_prompt_cls
from utils.data import replace_images_in_prompt, crop_img


@MFMWrapper.register_task('classify')
def classify(model: MFMWrapper, file_name: List, prompt: Optional[Dict] = None, prompt_no: int = -1, crop: bool = True, dataset: str = 'imagenet') -> Tuple[List[Dict], Tuple[int, int], bool]:
    """Classify a batch of images using the MFM.

    Args:
        model: The MFM model to use.
        prompt: The prompt to use for the classification.
        crop: Whether to resize and crop the image before classification.
        dataset: The dataset to use for classification.

    Returns:
        resp_list: List of dicts, each containing the "class"
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred
    """

    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]
    if crop:
        imgs = [crop_img(img) for img in imgs]
    imagenet_labels = json.load(open('/scratch/rahul/4o-dev/data/metadata/imagenet-actual-labels.json'))  # list of 1000 imagenet labels

    if not prompt:
        prompt = full_prompt_cls(prompt_no, imagenet_labels, len(imgs), model.name, len(imagenet_labels), dataset)

    prompt = replace_images_in_prompt(prompt, imgs)

    resp_dict, tokens, error_status = model.send_message(prompt)
    if error_status:
        return None, tokens, error_status

    resp_list = []
    for i in range(len(resp_dict)):
        resp_list.append({"class": resp_dict[str(i+1)], "file_name": file_name[i].strip()})

    return resp_list, tokens, error_status
