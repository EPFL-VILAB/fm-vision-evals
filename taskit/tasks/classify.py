from typing import Dict, List, Optional, Union

from PIL import Image

from taskit.mfm import MFMWrapper
from taskit.prompts import full_prompt_cls
from taskit.utils.data import replace_images_in_prompt, crop_img
from taskit.utils.data_constants import IMAGENET_LABELS


@MFMWrapper.register_task('classify')
def classify(model: MFMWrapper, file_name: Union[List[str], str], prompt: Optional[Dict] = None, prompt_no: int = -1, crop: bool = True, labels: List[str] = IMAGENET_LABELS, return_dict=False):
    """Classify a batch of images using the MFM.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to classify.
        prompt: The prompt to use for the classification.
        prompt_no: The prompt number to use for the classification (if prompt is None).
        crop: Whether to resize and crop the image before classification.
        labels: The list of labels to use for classification.
        return_dict: Whether to return the result as a list of dictionaries.

    Returns:
        (if return_dict is True)
        resp_list: List of dicts, each containing the "class"
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        resp_list: List of the predicted classes
        tokens: A tuple containing the completion tokens and the prompt tokens
    """

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    if crop:
        imgs = [crop_img(img) for img in imgs]
    imagenet_labels = labels

    if not prompt:
        prompt = full_prompt_cls(prompt_no, imagenet_labels, len(imgs), model.name)

    prompt = replace_images_in_prompt(prompt, imgs)

    resp_dict, tokens, error_status = model.send_message(prompt)
    if error_status:
        return None, tokens, error_status

    resp_list = []
    for i in range(len(resp_dict)):
        resp_list.append({"class": resp_dict[str(i+1)], "file_name": file_name[i].strip()})

    if return_dict:
        return resp_list, tokens, error_status
    else:
        return [dic["class"] for dic in resp_list], tokens
