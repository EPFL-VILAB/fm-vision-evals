from typing import Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.segmentation import slic
from tqdm import tqdm

from taskit.eval import eval_segment
from taskit.mfm import MFMWrapper
from taskit.utils.data import replace_images_in_prompt, draw_around_superpixel
from taskit.utils.data_constants import COCO_SEMSEG_LABELS, COCO_COLOR_MAP


# --System Prompt----------------------------------------------------------------


def system_prompts_segment(prompt_no: int, seg_classes: list, batch_size: int, shape: str, classes_guessed=[]):
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    if prompt_no == 1:
        system_prompt = f"""You are an advanced classifier. You will be provided with a number of {shape}s, each representing a part of an object or background that belongs to a semantic class. Your task is to determine the semantic class of the underlying object or background to which each {shape} belongs. The available semantic classes are: {seg_classes}.\n\n""" +\
                    """Your response should follow this format:\n""" +\
                    f"""{{"reasoning_steps": [the reasoning steps leading to the final conclusion], "1": "class 1", "2": "class 2", ... "{batch_size}": "class {batch_size}"}}.\n\n""" +\
                    f"""It is important to note that you are not identifying the class the {shape} demarcates itself, but rather the class of the object or background that the {shape} is on."""
    elif prompt_no == 2 and (len(classes_guessed) > 0):
        system_prompt = f"""You are an advanced image classifier AI. Your task is to determine the semantic class of objects or backgrounds marked by {shape}s in an image. Each {shape} marks a part of an object or background, and you need to identify the class to which the entire object or background belongs.\n\n""" +\
                        f"""<seg_classes>{seg_classes}</seg_classes>\nThese are the available semantic classes from which you must choose when classifying each {shape}.\n\n""" +\
                        f"""<classes_guessed>{classes_guessed}</classes_guessed>\nThese are the classes you have already guessed in previous segments. Use them if they are relevant to the current segment, otherwise, consider other classes from <seg_classes>.\n\n""" +\
                        f"""<batch_size>{batch_size}</batch_size>\nThis is the number of {shape}s you need to classify in this task.\n\n""" +\
                        """To complete this task effectively, follow these steps:\n\n""" +\
                        f"""1. For each {shape}, consider its location, size, and any other visual cues that might help identify the object or background it's part of.\n\n""" +\
                        f"""2. Think about which of the available semantic classes in <seg_classes> best matches the object or background represented by the {shape}.\n\n""" +\
                        """3. Use your knowledge of common objects, their parts, and typical backgrounds to make informed decisions.\n\n""" +\
                        """4. If you're unsure, consider multiple possibilities and explain your reasoning for choosing one over the others.\n\n""" +\
                        """Provide your response in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "1": "class 1",\n""" +\
                        """ "2": "class 2",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        """Important notes:\n""" +\
                        f"""- You are identifying the class of the entire object or background, not just the part demarcated by the {shape}.\n""" +\
                        f"""- Ensure that your classification for each {shape} is one of the classes provided in <seg_classes>.\n""" +\
                        """- Provide clear and concise reasoning steps that explain your classification process.\n""" +\
                        """- If you're uncertain about a classification, explain your thought process and why you chose one class over others.\n\n""" +\
                        f"""Remember, your goal is to provide accurate classifications based on the limited information given by the {shape}s, using your understanding of common objects and backgrounds."""
    elif prompt_no == 2:
        system_prompt = f"""You are an advanced image classifier AI. Your task is to determine the semantic class of objects or backgrounds marked by {shape}s in an image. Each {shape} marks a part of an object or background, and you need to identify the class to which the entire object or background belongs.\n\n""" +\
                        f"""<seg_classes>{seg_classes}</seg_classes>\nThese are the available semantic classes from which you must choose when classifying each {shape}.\n\n""" +\
                        f"""<batch_size>{batch_size}</batch_size>\nThis is the number of {shape}s you need to classify in this task.\n\n""" +\
                        """To complete this task effectively, follow these steps:\n\n""" +\
                        f"""1. For each {shape}, consider its location, size, and any other visual cues that might help identify the object or background it's part of.\n\n""" +\
                        f"""2. Think about which of the available semantic classes in <seg_classes> best matches the object or background represented by the {shape}.\n\n""" +\
                        """3. Use your knowledge of common objects, their parts, and typical backgrounds to make informed decisions.\n\n""" +\
                        """4. If you're unsure, consider multiple possibilities and explain your reasoning for choosing one over the others.\n\n""" +\
                        """Provide your response in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "1": "class 1",\n""" +\
                        """ "2": "class 2",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        """Important notes:\n""" +\
                        f"""- You are identifying the class of the entire object or background, not just the part demarcated by the {shape}.\n""" +\
                        f"""- Ensure that your classification for each {shape} is one of the classes provided in <seg_classes>.\n""" +\
                        """- Provide clear and concise reasoning steps that explain your classification process.\n""" +\
                        """- If you're uncertain about a classification, explain your thought process and why you chose one class over others.\n\n""" +\
                        f"""Remember, your goal is to provide accurate classifications based on the limited information given by the {shape}s, using your understanding of common objects and backgrounds."""

    elif prompt_no == 3:
        system_prompt = """You are an advanced image classifier specializing in identifying semantic classes of objects or backgrounds marked by specific shapes in images. Your task is to analyze a full image, examine zoomed-in crops of that image, and classify the semantic class of the object or background that each shape demarcates.\n""" +\
                        """1. First, examine the full image.\n""" +\
                        f"""2. After that, you will be provided with zoomed-in crops from the full image. Each crop is associated with a {shape} that marks a specific area of interest.\n""" +\
                        """3. Lastly, you will be provided with some additional context surrounding these zoomed-in crops.\n""" +\
                        f"""Your task is to determine the semantic class of the object or background that each {shape} is marking. The available semantic classes are:\n""" +\
                        f"""<semantic_classes>\n{seg_classes}\n</semantic_classes>\n\n""" +\
                        """Guidelines for classification:\n""" +\
                        f"""1. Focus on the object or background that the {shape} is marking, not the {shape} itself.\n""" +\
                        """2. Consider the context provided by the full image and the zoomed-in crop.\n""" +\
                        """3. Use the additional context to inform your decision.\n""" +\
                        """4. If you're unsure, explain your reasoning and provide your best guess.\n""" +\
                        """Present your analysis and classification in the following format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "1": "class 1",\n""" +\
                        """ "2": "class 2",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        f"""Ensure that you provide classifications for all {batch_size} shapes. If you cannot confidently classify a shape, explain your reasoning and provide your best estimate based on the available information."""

    elif prompt_no == 4:
        system_prompt = """You are an expert image classifier tasked with identifying the semantic classes of objects or backgrounds marked by specific shapes in images. Your job is to analyze a full image, examine its zoomed-in crops, and classify the semantic class of the object or background that each shape marks.\n\n""" +\
                        """### Instructions:\n\n""" +\
                        """1. **Examine the Full Image**: Start by analyzing the entire image to understand the overall context.\n""" +\
                        f"""2. **Analyze Zoomed-In Crops**: You will then be provided with several zoomed-in crops of the image, each containing a {shape} that highlights a specific area of interest.\n""" +\
                        """3. **Consider Additional Context**: Finally, review any additional context around these zoomed-in crops to help inform your classification.\n\n""" +\
                        """### Task:\n\n""" +\
                        f"""Determine the semantic class of the object or background associated with each {shape}. The possible semantic classes are listed below:\n\n""" +\
                        f"""<semantic_classes>\n{seg_classes}\n</semantic_classes>\n\n""" +\
                        """### Guidelines for Classification:\n\n""" +\
                        f"""1. Focus on the object or background marked by each {shape}, not the {shape} itself.\n""" +\
                        """2. Use the context from the full image and the zoomed-in crops to make your decision.\n""" +\
                        """3. Leverage the additional surrounding context to refine your classification.\n""" +\
                        """4. If uncertain, provide your reasoning and your best guess based on the information available.\n\n""" +\
                        """### Response Format:\n\n""" +\
                        """Present your analysis and classifications in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "1": "class 1",\n""" +\
                        """ "2": "class 2",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        f"""Ensure that you provide a classification for each of the {batch_size} shapes. If you cannot confidently classify a shape, explain your reasoning and provide your best estimate based on the given context."""
    elif prompt_no == 5:
        system_prompt = f"""You are an expert semantic classifier. You will be given a set of {shape}s, each highlighting a segment of an object or background within an image. Your task is to identify the semantic class of the underlying object or background marked by each {shape}.\n\n""" +\
                        """### Available Semantic Classes:\n""" +\
                        f"""{seg_classes}\n\n""" +\
                        """### Task Instructions:\n\n""" +\
                        f"""1. **Analyze the Image Segments**: For each {shape} provided, determine the semantic class of the object or background it represents, using the list of available semantic classes.\n""" +\
                        f"""2. **Focus on the Underlying Content**: Do not classify the {shape} itself; instead, focus on the object or background that the {shape} highlights.\n\n""" +\
                        """### Response Format:\n\n""" +\
                        """Provide your response in the following structured format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "1": "class 1",\n""" +\
                        """ "2": "class 2",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        """### Important Notes:\n\n""" +\
                        """- Make sure your reasoning steps are clear and detailed, leading logically to your final classification.\n""" +\
                        """- If uncertain about any classification, provide your best judgment along with a brief explanation of your reasoning.\n\n""" +\
                        f"""- Make sure the predicted classes belong in {seg_classes}\n\n""" +\
                        """Follow these guidelines to accurately classify each shape and ensure your output matches the format specified above."""

    return system_prompt


# --JSON Schema----------------------------------------------------------------


def json_schema_segment(shape: str, batch_size: int, model: str):
    if model == 'gpt-4o-2024-08-06':
        json_schema, json_properties = {}, {}
        json_properties["reasoning_steps"] = {"type": "array", "items": {"type": "string"}, "description": f"The step-by-step reasoning process leading to the final conclusion. Begin by describing the full image. Then, analyze each {shape}, and infer which object or background class in the image each {shape} represents. Use the semantic classes provided in the system prompt for your reasoning."}
        json_properties.update({str(k+1): {"type": "string", "description": f"The semantic class of the object or background in the full image to which {shape} {k+1} belongs. The output must be one of the classes listed in the system prompt."} for k in range(batch_size)})
        json_schema["name"] = "reasoning_schema"
        json_schema["strict"] = True
        json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["reasoning_steps"] + [str(k+1) for k in range(batch_size)], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        json_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning_steps": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING),
                    description=f"The step-by-step reasoning process leading to the final conclusion. Begin by describing the full image. Then, analyze each {shape}, and infer which object or background class in the image each {shape} represents. Use the semantic classes provided in the system prompt for your reasoning."
                ),
                **{str(k+1): genai.protos.Schema(type=genai.protos.Type.STRING, description=f"The semantic class of the object or background in the full image to which {shape} {k+1} belongs. The output must be one of the classes listed in the system prompt.") for k in range(batch_size)}
            },
            required=["reasoning_steps"] + [str(k+1) for k in range(batch_size)],
        )

    elif model == 'claude-3-5-sonnet-20240620':
        reasoning_steps_description = """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """ +\
                                  f"""leading to the final conclusion. Begin by describing the full image context. Then, analyze each {shape} provided, """ +\
                                  """discussing its characteristics and inferring which object or background class it belongs to. Use the semantic classes provided in the system prompt for your reasoning."""

        segment_descriptions = "\n".join([
            f'"{k+1}": This field should contain a string representing the semantic class of the object or background in the full image to which {shape} {k+1} belongs. '
            "The output must be one of the classes listed in the system prompt."
            for k in range(batch_size)
        ])

        json_schema = (
            "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
            "{\n"
            '  "reasoning_steps": [list of reasoning steps],\n'
            + "\n".join([f'  "{k+1}": "class name",' for k in range(batch_size)]) +
            "\n}\n\n"
            "Where:\n"
            f"- {reasoning_steps_description}\n"
            f"- {segment_descriptions}\n"
            "Please ensure that the output follows this format strictly, without additional fields or changes in structure."
        )

        expected_keys = ["reasoning_steps"] + [str(k+1) for k in range(batch_size)]
        json_schema = (json_schema, expected_keys)

    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_segment(prompt_no: int, class_labels: list, shape: str, batch_size: int, model: str, start_idx: int, n_unique_segments: int, classes_guessed: list = []):
    messages = []

    system_prompt = system_prompts_segment(prompt_no, class_labels, batch_size, shape, classes_guessed)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_1 = """Here is the full image provided for some context."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_1}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    user_prompt_2 = f"""Here are {batch_size} {shape}s for analysis."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_2}]})
    for j in range(start_idx, start_idx + batch_size):
        if j > n_unique_segments:
            break
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here is {shape} {j-(start_idx-1)}"}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})

    user_prompt_3 = f"""Here is additional surrounding context for the {shape}s. Note that other semantic classes might be present in this additional context; however, these should not be considered when determining the class of the object or background marked by the {shape}."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_3}]})
    for j in range(start_idx, start_idx + batch_size):
        if j > n_unique_segments:
            break
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here is the context for {shape} {j-(start_idx-1)}"}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})

    json_schema = json_schema_segment(shape, batch_size, model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


@MFMWrapper.register_task('segment')
def segment(
    model: MFMWrapper,
    file_name: Union[List[str], str],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_segments: int = 400,
    batch_size: int = 16,
    shape: str = "point",
    labels: List[str] = COCO_SEMSEG_LABELS,
    color_map: Dict[str, list] = COCO_COLOR_MAP,
    return_dict: bool = False
):
    """Segments image(s) using an MFM.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to segment.
        prompt: The prompt to use for segmentation.
        prompt_no: The prompt number to use (if prompt is None).
        n_segments: The number of segments to split the image into (using SLIC). The actual number of segments will be close but may be different.
        batch_size: The number of segments to classify in each batch.
        shape: The shape of the visual marker.
        labels: The list of labels to use for segmentation.
        color_map: The color map for the segmentation. (dict mapping class names to list of RGB (0-255) values)
        return_dict: Whether to return the result as a list of dictionaries.

    Returns:
        (if return_dict is True)
        resp_list: List of dicts, each containing the segment_idx and the corresponding class
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        resp_list: List of segment images (display using plt.imshow())
        tokens: A tuple containing the completion tokens and the prompt tokens
    """

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    compl_tokens, prompt_tokens = 0, 0
    resp_dict_list, error_status = [], False

    for img_idx, img in enumerate(imgs):
        segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
        n_unique_segments = len(np.unique(segments))

        all_results, classes_guessed = {}, []
        display_pbar = True if return_dict else False
        for start_idx in tqdm(range(1, n_unique_segments + 1, batch_size), disable=not display_pbar):
            prompt_imgs = [img]
            for j in range(start_idx, start_idx + batch_size):
                if j > n_unique_segments:
                    break
                segment = draw_around_superpixel(img, segments, j, shape, crop_width=1, radius=4)
                prompt_imgs.append(segment)

            for j in range(start_idx, start_idx + batch_size):
                if j > n_unique_segments:
                    break
                segment_2 = draw_around_superpixel(img, segments, j, shape, crop_width=40, radius=4)
                prompt_imgs.append(segment_2)

            full_prompt = full_prompt_segment(prompt_no, labels, shape, batch_size, model.name, start_idx, n_unique_segments, classes_guessed) if not prompt else prompt
            full_prompt = replace_images_in_prompt(full_prompt, prompt_imgs)

            resp_dict, tokens, error_status = model.send_message(full_prompt)
            if not error_status:
                resp_dict.pop("reasoning_steps", None)

            for k in range(batch_size):
                if start_idx + k > n_unique_segments:
                    break
                all_results[str(start_idx + k)] = resp_dict.get(str(k + 1), "unknown")

            classes_guessed += [v for k, v in resp_dict.items() if k != "reasoning_steps"]
            classes_guessed = list(set(classes_guessed))
            compl_tokens += tokens[0]
            prompt_tokens += tokens[1]

        all_results["file_name"] = file_name[img_idx].strip()
        resp_dict_list.append(all_results)

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        seg_maps = model.eval(resp_dict_list, eval='eval_segment', n_segments=n_segments, labels=labels, color_map=color_map, visualise=True)
        return seg_maps, (compl_tokens, prompt_tokens)
