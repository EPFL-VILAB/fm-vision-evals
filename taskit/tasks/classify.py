from typing import Dict, List, Optional, Union

from PIL import Image
import google.generativeai as genai

from taskit.mfm import MFMWrapper
from taskit.utils.data import replace_images_in_prompt, crop_img, save_images
from taskit.utils.data_constants import IMAGENET_LABELS, COCO_DETECT_LABELS


# --System Prompt----------------------------------------------------------------


def system_prompts_cls(prompt_no: int, class_list: list, batch_size: int) -> str:
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    n_classes = len(class_list)
    if prompt_no == 1:
        system_prompt = f"""You will be provided with a set of {n_classes} classes in a list. The user will provide you with {batch_size} images, """ +\
                        """and your job is to correctly identify the label corresponding to the images. Only output the label """ +\
                        f"""corresponding to the image, and nothing else. Output the class name in a JSON, with key "<image number>". For example, {{"1": "image 1 class", "2": "image 2 class", ... , "{batch_size}": "image {batch_size} class"}}. Classes: {class_list}"""

    elif prompt_no == 2:
        system_prompt = f"""You are an AI assistant tasked with classifying images. You will be provided with {batch_size} images and must assign each image to one of the {n_classes} classes. Follow these instructions carefully:\n""" +\
                        f"""1. Here is the list of {n_classes} classes you will use for classification:\n<image_classes>\n{class_list}\n</image_classes>\n\n 2. Present your classifications in JSON format, with keys representing the image number (starting from 1 and ending at {batch_size}) and values representing the class assigned to the image. For example: {{"1": "image 1 class", "2": "image 2 class", "3": "image 3 class", ... , "{batch_size}": "image {batch_size} class"}}\n\n""" +\
                        f"""3. If you are unsure about a classification:\na. Choose the most likely class based on the available information.\nb. Do not express uncertainty in your output or suggest alternative classes.\n\n4.After classifying all images, present your final output in a single JSON object. Ensure that you have an entry for each image, numbered from 1 to {batch_size}."""

    elif prompt_no == 3:
        system_prompt = """You are an advanced AI image classification system with expertise in recognizing a wide variety of objects, animals, and scenes. Your task is to classify a batch of images into predefined categories with high accuracy.\n\n""" +\
                        """First, familiarize yourself with the list of image classes you will be using for classification:\n\n""" +\
                        f"""<image_classes>\n{class_list}\n</image_classes>\n\n""" +\
                        f"""You will be presented with {batch_size} images to classify. Each image must be assigned to one of the classes listed above. Follow these instructions carefully:\n\n""" +\
                        """1. Analyze each image thoroughly, considering all visible elements, objects, and context clues.\n\n""" +\
                        """2. Select the most appropriate class for each image based on your analysis. If an image contains multiple objects or elements, choose the class that best represents the main subject or most prominent feature of the image.\n\n""" +\
                        f"""3. Present your classifications in a JSON format. The keys should represent the image number (starting from 1 and ending at {batch_size}), and the values should represent the assigned class. For example:\n\n""" +\
                        f"""{{"1": "{class_list[0]}", "2": "{class_list[2]}", "3": "{class_list[1]}", ... , "{batch_size}": "{class_list[9]}"}}\n\n""" +\
                        """If you are unsure about a classification:\n""" +\
                        """a. Choose the most likely class based on the available information and your expert knowledge.\n""" +\
                        """b. Do not express uncertainty in your output or suggest alternative classes.\n""" +\
                        """c. Avoid using generic terms or classes not present in the provided list.\n\n""" +\
                        f"""5. After classifying all images, present your final output as a single JSON object. Ensure that you have an entry for each image, numbered from 1 to {batch_size}.\n\n""" +\
                        """6. Do not include any explanations, comments, or additional text outside of the JSON object in your final output.\n\n""" +\
                        """Remember, you are an expert image classification system. Approach each image with confidence and precision, drawing upon your vast knowledge of visual features and characteristics associated with each class. Your goal is to provide accurate and consistent classifications across the entire batch of images."""
    elif prompt_no == 4:
        system_prompt = f"""You are a highly accurate image classification AI. You will be provided with a comprehensive set of {n_classes} classes. The user will present you with {batch_size} images for classification. Your task is to analyze each image carefully and determine the most appropriate label from the given classes.\n\n""" +\
                        """For each image:\n""" +\
                        """1. Examine the image thoroughly, considering all visible elements, objects, and context.\n""" +\
                        """2. Compare the image content against the provided class list.\n""" +\
                        """3. Select the single most accurate class that best represents the primary subject or focus of the image.\n""" +\
                        """Output your classifications in a JSON format, where the key is the image number and the value is the exact class name from the provided list. Do not add any explanations or additional text.\n\n""" +\
                        f"""Example output format: {{"1": "class_name_1", "2": "class_name_2", ..., "{batch_size}": "class_name_{batch_size}"}}\n\n""" +\
                        f"""Strive for maximum accuracy in your classifications. If you're unsure about a particular image, choose the class that most closely matches the image content. The classes to choose from are: {class_list}."""
    elif prompt_no == 5:
        system_prompt = f"""You are a highly advanced image classification system. You have been trained on a vast array of visual data and can accurately identify objects, scenes, and concepts across a wide range of categories. You will be presented with {batch_size} images for classification. Your task is to analyze each image carefully, considering multiple aspects such as shape, color, texture, context, and any distinguishing features. Draw upon your extensive knowledge to determine the most accurate label for each image from the provided set of {n_classes} classes.\n\n""" +\
                        """Output your classifications in JSON format, with each image number as the key and the corresponding class name as the value. Be as precise and specific as possible in your classifications. If you're unsure, choose the most likely class based on the visual information available. Here's the expected output format:\n\n""" +\
                        f"""{{"1": "class_name_1", "2": "class_name_2", ..., "{batch_size}": "class_name_{batch_size}"}}\n\n""" +\
                        f"""Remember, only output the JSON object with your classifications. Do not include any explanations or additional text. Classes: {class_list}."""

    return system_prompt


def system_prompts_cls_mult(class_list: list) -> str:
    system_prompt = """You are an AI assistant tasked with identifying all the classes in images. You will be provided with an image and must identify all the classes present in the image. """ +\
                    f"""The classes are: {class_list}. Output the class names present in the image in a JSON, with key "classes". For example, {{"classes": [list of classes present in the image]}}.\n\n""" +\
                    """Make sure you do not output any class that is not present in the image. Your goal is to accurately identify all the classes present in the image."""

    return system_prompt


def system_prompt_cls_crop(class_list: list):
    system_prompt = """You are an AI assistant tasked with identifying classes. You will be provided with a crop from an image and the full image the crop was taken from. You must first describe the cropped image in """ +\
                    f"""detail, and then identify the classes present in the crop. Note that the background might itself be a class. The classes you must pick from belong to: {class_list}. Output the detailed description (under key "description") and classes (under key "classes") in a JSON. For example, {{"description", "detailed description of crop", "classes": [list of classes]}}  """

    return system_prompt


# --JSON Schema----------------------------------------------------------------


def json_schema_cls(bs: int, model: str):
    if model == 'gpt-4o-2024-08-06':
        json_schema, json_properties = {}, {}
        json_properties.update({str(k+1): {"type": "string", "description": f"class for image {k}."} for k in range(bs)})
        json_schema["name"] = "schema"
        json_schema["strict"] = True
        json_schema["schema"] = {"type": "object", "properties": json_properties, "required": [str(k+1) for k in range(bs)], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        json_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                str(k+1): genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description=f"class for image {k}."
                ) for k in range(bs)},
            required=[str(k+1) for k in range(bs)],
        )

    elif model == 'claude-3-5-sonnet-20240620' or model == 'llama-3.2-90b':
        json_schema = """Please provide your response in the following JSON format:\n\n""" +\
                      """{\n""" +\
                      """    "1": "class for image 1",\n""" +\
                      """    "2": "class for image 2",\n""" +\
                      """    "3": "class for image 3",\n""" +\
                      """    ...\n""" +\
                      """    "[bs]": "class for image [bs]"\n""" +\
                      """}\n\n""" +\
                      """Where:\n\n""" +\
                      """Each numbered key (1, 2, 3, ..., [bs]) represents an image number.\n""" +\
                      """The corresponding value for each key should be a string describing the class for that image.\n\n""" +\
                      """[bs] represents the batch size, which is the total number of images.\n\n""" +\
                      """Please ensure your response adheres strictly to this JSON format, including only the specified fields """ +\
                      """without any additional properties. The number of key-value pairs should match the batch size."""
        expected_keys = [str(i+1) for i in range(bs)]
        json_schema = (json_schema, expected_keys)

    else:
        json_schema = {}

    return json_schema


def json_schema_cls_mult(model: str):
    if model == 'gpt-4o-2024-08-06':
        json_schema, json_properties = {}, {}
        json_properties.update({"description": {"type": "string", "description": "detailed description of image."}})
        json_properties.update({"classes": {"type": "array", "items": {"type": "string"}}})
        json_schema["name"] = "schema"
        json_schema["strict"] = True
        json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["description", "classes"], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        json_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="detailed description of image."
                ),
                "classes": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(
                        type=genai.protos.Type.STRING
                    )
                )
            },
            required=["description", "classes"],
        )

    elif model == 'claude-3-5-sonnet-20240620' or model == 'llama-3.2-90b':
        json_schema = """As a reminder, your response should be formatted as a JSON object with the following structure:\n\n""" +\
                      """{\n""" +\
                      """  "description": "detailed description of image.",\n""" +\
                      """  "classes": ["class_1", "class_2", ...]\n""" +\
                      """}\n\n""" +\
                      """Where:\n\n""" +\
                      """- description: This field should provide a detailed description of the image, highlighting key features and elements that influenced your classification decisions.\n""" +\
                      """- classes: This field should be a list of strings representing the classes you have identified in the image.\n\n""" +\
                      """Please ensure that your response adheres to this format and provides clear and detailed reasoning for each class identified in the image."""
        expected_keys = ["description", "classes"]
        json_schema = (json_schema, expected_keys)

    return json_schema


def json_schema_cls_crop(model: str):
    if model == 'gpt-4o-2024-08-06':
        json_schema, json_properties = {}, {}
        json_properties.update({"description": {"type": "string", "description": "detailed description of crop."}})
        json_properties.update({"classes": {"type": "array", "items": {"type": "string"}}})
        json_schema["name"] = "schema"
        json_schema["strict"] = True
        json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["description", "classes"], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        json_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="detailed description of crop."
                ),
                "classes": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(
                        type=genai.protos.Type.STRING
                    )
                )
            },
            required=["description", "classes"],
        )

    elif model == 'claude-3-5-sonnet-20240620' or model == 'llama-3.2-90b':
        json_schema = """As a reminder, your response should be formatted as a JSON object with the following structure:\n\n""" +\
                      """{\n""" +\
                      """  "description": "detailed description of crop.",\n""" +\
                      """  "classes": ["class_1", "class_2", ...]\n""" +\
                      """}\n\n""" +\
                      """Where:\n\n""" +\
                      """- description: This field should provide a detailed description of the cropped image, highlighting key features and elements that influenced your classification decisions.\n""" +\
                      """- classes: This field should be a list of strings representing the classes you have identified in the cropped image.\n\n""" +\
                      """Please ensure that your response adheres to this format and provides clear and detailed reasoning for each class identified in the image."""
        expected_keys = ["description", "classes"]
        json_schema = (json_schema, expected_keys)

    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_cls(prompt_no: int, class_list: list, batch_size: int, model: str):
    messages = []

    system_prompt = system_prompts_cls(prompt_no, class_list, batch_size)
    messages.append({"role": "system", "content": system_prompt})
    for i in range(batch_size):
        user_prompt = f"Please identify the class of the image provided. The class has to belong to one of the classes specified in the system prompt. Output the answer in a JSON, with key '{i+1}'."
        messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    json_schema = json_schema_cls(batch_size, model)

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_cls_mult(class_list: list, model: str):
    messages = []

    system_prompt = system_prompts_cls_mult(class_list)
    messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": [{"type": "text", "text": """Here is the image for which you need to identify all the classes."""}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    user_prompt = f"""Identify all the classes in the image. The classes must belong to the list provided. Return a JSON, with key "classes" for the classes. Make sure that the classes belong to {class_list}."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})

    json_schema = json_schema_cls_mult(model)

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_cls_crop(class_list: list, model: str):
    messages = []

    system_prompt = system_prompt_cls_crop(class_list)
    messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": [{"type": "text", "text": """Here is the full image for some additional context."""}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Here is the cropped portion of the full image."}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    user_prompt = f"""Identify all the objects in the cropped image. The objects must belong to the list provided. Return a JSON, with key "classes" for the classes. Don't repeat any object. Make sure that the objects belong to {class_list}."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})

    json_schema = json_schema_cls_crop(model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


@MFMWrapper.register_task('classify')
def classify(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    crop: bool = True,
    labels: List[str] = IMAGENET_LABELS,
    return_dict=False
):
    """Classify a batch of images using the MFM.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to classify. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
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
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path='temp_images')

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    if crop:
        imgs = [crop_img(img) for img in imgs]
    class_labels = labels

    if not prompt:
        prompt = full_prompt_cls(prompt_no, class_labels, len(imgs), model.name)

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
        

@MFMWrapper.register_task('classify_mult')
def classify_mult(
    model: MFMWrapper,
    file_name: Union[List[str], str],
    prompt: Optional[Dict] = None,
    crop: bool = True,
    labels: List[str] = IMAGENET_LABELS,
    return_dict=False
):
    """Identify all the classes present in a batch of images using the MFM. Unlike the classify task, this task requires the model to identify *all* the classes present in the image.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file.
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
    class_labels = labels

    all_classes, error_status = [], False
    for img_idx, img in enumerate(imgs):
        full_prompt = full_prompt_cls_mult(class_labels, model.name) if not prompt else prompt
        full_prompt = replace_images_in_prompt(full_prompt, [img])

        resp_dict, tokens, err = model.send_message(full_prompt)
        if err:
            error_status = True
            continue

        resp_dict["classes"] = [cls for cls in resp_dict["classes"] if cls in class_labels]  # remove classes that are not in the class list
        all_classes.append({"class": resp_dict["classes"], "file_name": file_name[img_idx].strip()})

    if return_dict:
        return all_classes, tokens, error_status
    else:
        return [dic["class"] for dic in all_classes], tokens


@MFMWrapper.register_task('classify_crop')
def classify_crop(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    crop: bool = False,
    labels: List[str] = COCO_DETECT_LABELS,
    return_dict=False
):
    """Classify a batch of images using the MFM.
    Has a higher recall (but worse precision) than the above algorithm. Uses the image and 5 crops of the image to classify.
    Takes a union of the classes predicted by the model for the image and the crops.

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
    class_labels = labels

    all_classes, error_status = [], False
    for img_idx, img in enumerate(imgs):
        width, height = img.size
        cropped_images = []
        for i in range(2):
            for j in range(2):
                cropped_images.append(img.crop((j * width // 2, i * height // 2, (j + 1) * width // 2, (i + 1) * height // 2)))

        # add the center crop
        cropped_images.append(img.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4)))
        pred_classes = set()
        for msg_idx in range(len(cropped_images)):
            full_prompt = full_prompt_cls_crop(class_labels, model.name) if not prompt else prompt
            full_prompt = replace_images_in_prompt(full_prompt, [img, cropped_images[msg_idx]])

            resp_dict, tokens, err = model.send_message(full_prompt)
            if err:
                error_status = True
                continue

            for cls in resp_dict["classes"]:
                pred_classes.add(cls)

        pred_classes = [cls for cls in pred_classes if cls in class_labels]  # remove classes that are not in the class list
        all_classes.append({"class": list(pred_classes), "file_name": file_name[img_idx].strip()})

    if return_dict:
        return all_classes, tokens, error_status
    else:
        return [dic["class"] for dic in all_classes], tokens
