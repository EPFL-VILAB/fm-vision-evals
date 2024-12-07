from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.segmentation import slic
from tqdm import tqdm

from taskit.eval import eval_normals
from taskit.mfm import MFMWrapper
from taskit.utils.data import replace_images_in_prompt, sample_segments, draw_around_superpixel, save_images


# --System Prompt----------------------------------------------------------------


def system_prompts_normals(prompt_no: int, shape: str) -> str:
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    if prompt_no == 1:
        system_prompt = f"""In this task, two regions are demarcated by blue and red {shape}s in an indoor scene. Your job is to figure out the direction in which these surfaces face relative to the viewer or camera. These directions are not limited to fixed categories (e.g., "right," "up," "outward") but exist on a continuous spectrum, so evaluate them based on the degree to which each surface faces these directions.\n\n""" +\
                        """You need to determine the following directions for these regions based on a continuous ranking of orientations:\n\n""" +\
                        """1) **Which region is facing more to the right (from the viewer's perspective)?**\n""" +\
                        """   - Ranking guideline: Surfaces facing more towards the right should rank higher. For instance, surfaces facing directly right should rank the highest, followed by surfaces with components of upward, downward, or outward directions, and surfaces facing left should rank the lowest.\n\n""" +\
                        """2) **Which region is facing more upwards (from the viewer's perspective)?**\n""" +\
                        """   - Ranking guideline: Surfaces facing more upwards should rank higher. Surfaces directly facing upward should be ranked highest, followed by those with components of left, right, or outward, and those facing downward should rank the lowest.\n\n""" +\
                        """3) **Which region is facing more outward towards the viewer or camera?**\n""" +\
                        """   - Ranking guideline: Surfaces facing more outward (towards the viewer) should rank highest. Surfaces facing left, right, upward, or downward should rank lower in terms of how much they face outward.\n\n""" +\
                        """For each of these three tasks, output the answer as a JSON with the keys "right," "up," and "out" and the values as the region colors ("red" or "blue") that face more in the respective directions. If both regions face equally in one direction, use the value "equal" for that direction.\n\n""" +\
                        """Example output format: \n{"right": "color", "up": "color", "out": "color"}."""

    elif prompt_no == 2:
        system_prompt = """You are an expert in computer vision and surface normal prediction. Your task is to analyze an image of an indoor scene and determine the orientation of two specific regions marked by colored shapes. This task requires a nuanced understanding of 3D space as represented in 2D images.\n\n""" +\
                        f"""You will be provided with an image in which two regions are demarcated by blue and red {shape}s. Your job is to determine the direction in which these surfaces face relative to the viewer or camera. You will assess three primary directions:\n\n""" +\
                        """1. **Right-facing orientation**\n""" +\
                        """2. **Upward-facing orientation**\n""" +\
                        """3. **Outward-facing orientation (towards the viewer/camera)**\n\n""" +\
                        """For each of these directions, you need to determine which region (blue or red) faces more in that direction. Remember, these orientations exist on a continuous spectrum and are not limited to fixed categories.\n\n""" +\
                        """**Guidelines for each direction assessment:**\n\n""" +\
                        """1. **Right-facing orientation**:\n""" +\
                        """   - Surfaces facing directly to the right should be ranked highest.\n""" +\
                        """   - Surfaces with components of upward, downward, or outward directions should be ranked next.\n""" +\
                        """   - Surfaces facing left should be ranked lowest.\n""" +\
                        """   - Consider the degree to which each surface faces right, not just whether it faces right or not.\n\n""" +\
                        """2. **Upward-facing orientation**:\n""" +\
                        """   - Surfaces facing directly upward should be ranked highest.\n""" +\
                        """   - Surfaces with components of left, right, or outward directions should be ranked next.\n""" +\
                        """   - Surfaces facing downward should be ranked lowest.\n""" +\
                        """   - Consider the degree to which each surface faces upward, not just whether it faces up or not.\n\n""" +\
                        """3. **Outward-facing orientation (towards the viewer/camera)**:\n""" +\
                        """   - Surfaces facing directly outward should be ranked highest.\n""" +\
                        """   - Surfaces with components of left, right, upward, or downward directions should be ranked lower.\n""" +\
                        """   - Consider the degree to which each surface faces outward, not just whether it faces out or not.\n\n""" +\
                        """Provide your final assessment in JSON format:\n\n""" +\
                        """<output>\n""" +\
                        """{\n  "right": "[color that faces more right]",\n  "up": "[color that faces more up]",\n  "out": "[color that faces more out]"\n}\n</output>\n\n""" +\
                        """If both regions face equally in one direction, use the value "equal" for that direction.\n\n""" +\
                        """Example output:\n\n""" +\
                        """<output>\n""" +\
                        """{\n  "right": "blue",\n  "up": "red",\n  "out": "equal"\n}\n</output>\n\n""" +\
                        """Remember, this task requires a nuanced assessment of surface orientations in 3D space as represented in a 2D image. Take into account perspective, shadows, and other visual cues that might indicate the orientation of each surface. Your analysis should reflect the continuous nature of these orientations rather than binary classifications.\n\n""" +\
                        """Carefully examine the image and provide a thorough analysis before making your final assessment."""

    elif prompt_no == 3:
        system_prompt = f"""In this task, two regions are demarcated by blue and red {shape}s in an indoor scene. Your job is to figure out the direction in which these surfaces face relative to the viewer or camera. These directions are not limited to fixed categories (e.g., "right," "up," "outward") but exist on a continuous spectrum, so evaluate them based on the degree to which each surface faces these directions.\n\n""" +\
                        """You need to determine the following directions for these regions based on a continuous ranking of orientations:\n\n""" +\
                        """1) **Which region is facing more to the right (from the viewer's perspective)?**\n""" +\
                        """   - Ranking guideline: Surfaces facing more towards the right should rank higher. For instance, surfaces facing directly right should rank the highest, followed by surfaces with components of upward, downward, or outward directions, and surfaces facing left should rank the lowest.\n\n""" +\
                        """2) **Which region is facing more upwards (from the viewer's perspective)?**\n""" +\
                        """   - Ranking guideline: Surfaces facing more upwards should rank higher. Surfaces directly facing upward should be ranked highest, followed by those with components of left, right, or outward, and those facing downward should rank the lowest.\n\n""" +\
                        """3) **Which region is facing more outward towards the viewer or camera?**\n""" +\
                        """   - Ranking guideline: Surfaces facing more outward (towards the viewer) should rank highest. Surfaces facing left, right, upward, or downward should rank lower in terms of how much they face outward.\n\n""" +\
                        """In addition to making these determinations, you are required to explain the reasoning behind your answers. For each question, describe the following reasoning steps:\n\n""" +\
                        """- **Surface orientation**: Describe the visual clues that helped determine the orientation of each surface in relation to the viewer.\n""" +\
                        """- **Comparison between regions**: Explain why one region faces more towards a specific direction (e.g., right, up, outward) compared to the other. If both regions face a direction equally, state the reasons behind this conclusion.\n""" +\
                        """- **Spatial understanding**: A good way to determine if the predicted direction of a surface's orientation is correct, is to consider moving in the predicted direction from the surface. If moving in that direction takes you away from the surface, then the prediction is correct. On the other hand, if moving in the opposite direction takes you away from the surface instead of towards it, then the prediction is incorrect. Use this diagnostic test to validate your predictions in the reasoning steps.\n\n""" +\
                        """For each of these three tasks, output the answer as a JSON with the keys "right," "up," and "out" and the values as the region colors ("red" or "blue") that face more in the respective directions. If both regions face equally in one direction, use the value "equal" for that direction.\n\n""" +\
                        """Example output format: \n{"reasoning_steps": ["The reasoning steps leading to the final conclusion."], "right": "color", "up": "color", "out": "color"}."""

    elif prompt_no == 4:
        system_prompt = """You are an expert in computer vision and surface normal prediction. Your task is to analyze an image of an indoor scene and determine the orientation of two specific regions marked by colored shapes. This task requires a nuanced understanding of 3D space as represented in 2D images.\n\n""" +\
                        f"""You will be provided with an image in which two regions are demarcated by blue and red {shape}s. Your job is to determine the **surface normal** of these regions relative to the viewer or camera. The surface normal is a unit vector that points perpendicularly outward from the surface.\n\n""" +\
                        """You will assess the **degree of orientation** of each region in three primary directions:\n\n""" +\
                        """1. **Right-facing orientation** (positive x-axis direction)\n""" +\
                        """2. **Upward-facing orientation** (positive y-axis direction)\n""" +\
                        """3. **Outward-facing orientation** (towards the viewer/camera, positive z-axis direction)\n\n""" +\
                        """For each of these directions, determine the **degree of orientation** of each region (blue or red) in that direction. Remember, these orientations exist on a continuous spectrum and are not limited to fixed categories.\n\n""" +\
                        """**Guidelines for each direction assessment:**\n\n""" +\
                        """1. **Right-facing orientation:**\n""" +\
                        """   - A surface that faces directly to the right has a surface normal pointing directly to the right (x-axis). Its degree of orientation in the right direction is 1.\n""" +\
                        """   - A surface that faces slightly upward and to the right has a surface normal pointing slightly upward and to the right. Its degree of orientation in the right direction is less than 1 but greater than 0.\n""" +\
                        """   - A surface that faces directly left has a surface normal pointing directly to the left (negative x-axis). Its degree of orientation in the right direction is -1.\n\n""" +\
                        """2. **Upward-facing orientation:**\n""" +\
                        """   - Similar guidelines apply to upward-facing orientation, considering the y-axis.\n\n""" +\
                        """3. **Outward-facing orientation:**\n""" +\
                        """   - A surface that faces directly outward towards the viewer has a surface normal pointing directly towards the viewer (z-axis). Its degree of orientation in the outward direction is 1.\n""" +\
                        """   - A surface that faces slightly downward and outward has a surface normal pointing slightly downward and outward. Its degree of orientation in the outward direction is less than 1 but greater than 0.\n\n""" +\
                        """In addition to making these determinations, you are required to explain the reasoning behind your answers. For each question, describe the following reasoning steps:\n\n""" +\
                        """- **Surface orientation**: Describe the visual clues that helped determine the orientation of each surface in relation to the viewer.\n""" +\
                        """- **Comparison between regions**: Explain why one region has a higher degree of orientation in a specific direction compared to the other. If both regions have similar degrees of orientation in a direction, state the reasons behind this conclusion.\n""" +\
                        """- **Spatial reasoning**: To validate your predictions, consider moving in the predicted direction from the surface. If moving in that direction takes you away from the surface, then the prediction is correct. On the other hand, if moving in the opposite direction takes you away from the surface instead of towards it, then the prediction is incorrect. Use this diagnostic test to validate your predictions in the reasoning steps.\n\n""" +\
                        """Provide your final assessment in JSON format:\n\n""" +\
                        """<output>\n""" +\
                        """{\n  "reasoning_steps": ["Detail your reasoning here."],\n  "right": "[color with higher degree of orientation in the right direction]",\n  "up": "[color with higher degree of orientation in the upward direction]",\n  "out": "[color with higher degree of orientation in the outward direction]"\n}\n</output>\n\n""" +\
                        """Remember, this task requires a nuanced assessment of surface orientations in 3D space as represented in a 2D image. Take into account perspective, shadows, and other visual cues that might indicate the orientation of each surface. Your analysis should reflect the continuous nature of these orientations rather than binary classifications.\n\n""" +\
                        """Carefully examine the image and provide a thorough analysis before making your final assessment."""

    elif prompt_no == 5:
        system_prompt = """**Surface Orientation Analysis**\n\n""" +\
                        f"""You are tasked with analyzing an indoor scene featuring two regions marked by blue and red {shape}s. Your objective is to determine the orientation of these surfaces relative to the viewer or camera. Please note that orientations are not limited to fixed categories such as "left" or "right"; they exist along a continuum.\n\n""" +\
                        """### Objectives:\n\n""" +\
                        """1. **Identify Rightward Orientation:**  \n""" +\
                        """   - **Question:** Which region is oriented more towards the right?  \n""" +\
                        """   - **Guideline:** Rank the surfaces based on how directly they face right. The most right-facing surfaces should receive the highest ranking, while those facing left should rank lower.\n\n""" +\
                        """2. **Determine Upward Orientation:**  \n""" +\
                        """   - **Question:** Which region is oriented more upwards?  \n""" +\
                        """   - **Guideline:** Rank the surfaces according to their upward orientation. Surfaces pointing directly upward should receive the highest ranking, while downward-facing surfaces should rank lower.\n\n""" +\
                        """3. **Assess Outward Orientation:**  \n""" +\
                        """   - **Question:** Which region faces outward towards the viewer?  \n""" +\
                        """   - **Guideline:** Surfaces that face directly outward should rank highest, while those oriented in other directions should rank lower.\n\n""" +\
                        """### Reasoning Breakdown:\n\n""" +\
                        """For each of the three orientation questions, please provide a detailed explanation:\n\n""" +\
                        """1. **Surface Insights:** Describe the visual cues that guided your decisions.\n""" +\
                        """2. **Comparative Analysis:** Explain why one region is oriented more towards a specific direction than the other. If both regions are equal, clarify this conclusion.\n""" +\
                        """3. **Spatial Validation:** Consider how moving in the predicted direction affects your position relative to the surface. Use this insight to confirm your predictions.\n\n""" +\
                        """### Output Format:\n\n""" +\
                        """Present your conclusions in JSON format, with keys "right," "up," and "out" indicating the colors of the regions facing in those respective directions. If both regions are equal in a direction, indicate this with "equal." \n\n""" +\
                        """{\n""" +\
                        """    "reasoning_steps": ["Detail your reasoning here."],\n""" +\
                        """    "right": "color",\n""" +\
                        """    "up": "color",\n""" +\
                        """    "out": "color"\n""" +\
                        """}\n""" +\
                        """You are encouraged to engage thoroughly with the task at hand and provide thoughtful analysis.\n"""

    return system_prompt


# --JSON Schema----------------------------------------------------------------


def json_schema_normals(prompt_no: int, model: str):
    if model == 'gpt-4o-2024-08-06':
        if prompt_no in [1, 2]:
            json_schema, json_properties = {}, {}
            json_properties["right"] = {"type": "string", "enum": ["red", "blue", "equal"], "description": "The color of the region that faces right (from the point of view of the viewer). If both are regions equally face right, output 'equal'. If a region faces left, output the color of the other region."}
            json_properties["up"] = {"type": "string", "enum": ["red", "blue", "equal"], "description": "The color of the region that faces up. If both are regions equally face up, output 'equal'. If a region faces down, output the color of the other region."}
            json_properties["out"] = {"type": "string", "enum": ["red", "blue", "equal"], "description": "The color of the region that faces out of the image. If both are regions equally face out of the image, output 'equal'. Note that a region facing inwards won't be visible to the camera. Therefore, output the color of the region that faces outwards, or output 'equal' if no region faces outwards."}
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["right", "up", "out"], "additionalProperties": False}

        elif prompt_no in [3, 4, 5]:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {"type": "array", "items": {"type": "string"}, "description": "The reasoning steps leading to the final conclusion."}
            json_properties["right"] = {"type": "string", "enum": ["red", "blue", "equal"], "description": "The color of the region that faces right (from the point of view of the viewer). If both regions equally face right, output 'equal'. If a region faces left, output the color of the other region."}
            json_properties["up"] = {"type": "string", "enum": ["red", "blue", "equal"], "description": "The color of the region that faces up. If both regions equally face up, output 'equal'. If a region faces down, output the color of the other region."}
            json_properties["out"] = {"type": "string", "enum": ["red", "blue", "equal"], "description": "The color of the region that faces out of the image. If both regions equally face out of the image, output 'equal'. Note that a region facing inwards won't be visible to the camera. Therefore, output the color of the region that faces outwards, or output 'equal' if no region faces outwards."}
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["reasoning_steps", "right", "up", "out"], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        if prompt_no in [1, 2]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "right": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue", "equal"],
                        description="The color of the region that faces right (from the point of view of the viewer). If both regions equally face right, output 'equal'. If a region faces left, output the color of the other region."
                    ),
                    "up": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue", "equal"],
                        description="The color of the region that faces up. If both regions equally face up, output 'equal'. If a region faces down, output the color of the other region."
                    ),
                    "out": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue", "equal"],
                        description="The color of the region that faces out of the image. If both regions equally face out of the image, output 'equal'. Note that a region facing inwards won't be visible to the camera. Therefore, output the color of the region that faces outwards, or output 'equal' if no region faces outwards."
                    )
                },
                required=["right", "up", "out"],
            )

        elif prompt_no in [3, 4, 5]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "reasoning_steps": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="The reasoning steps leading to the final conclusion. Use your understanding of 3D scene geometry to reason."
                    ),
                    "right": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue", "equal"],
                        description="The color of the region that faces right (from the point of view of the viewer). If both regions equally face right, output 'equal'. If a region faces left, output the color of the other region."
                    ),
                    "up": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue", "equal"],
                        description="The color of the region that faces up. If both regions equally face up, output 'equal'. If a region faces down, output the color of the other region."
                    ),
                    "out": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue", "equal"],
                        description="The color of the region that faces out of the image. If both regions equally face out of the image, output 'equal'. Note that a region facing inwards won't be visible to the camera. Therefore, output the color of the region that faces outwards, or output 'equal' if no region faces outwards."
                    )
                },
                required=["reasoning_steps", "right", "up", "out"],
            )

    elif model == 'claude-3-5-sonnet-20240620' or model == 'llama-3.2-90b':
        if prompt_no in [1, 2]:
            json_schema = "Please provide your response in the following JSON format:\n\n" +\
                "{\n" +\
                '    "right": "color",\n' +\
                '    "up": "color",\n' +\
                '    "out": "color"\n' +\
                "}\n\n" +\
                "Where:\n\n" +\
                "right: This field should contain a string representing the color of the region that faces right (from the " +\
                "point of view of the viewer). If both regions equally face right, output 'equal'. If a region faces left, " +\
                "output the color of the other region. The output must be one of: 'red', 'blue', or 'equal'.\n\n" +\
                "up: This field should contain a string representing the color of the region that faces up. If both regions " +\
                "equally face up, output 'equal'. If a region faces down, output the color of the other region. The output " +\
                "must be one of: 'red', 'blue', or 'equal'.\n\n" +\
                "out: This field should contain a string representing the color of the region that faces out of the image. " +\
                "If both regions equally face out of the image, output 'equal'. Note that a region facing inwards won't be " +\
                "visible to the camera. Therefore, output the color of the region that faces outwards, or output 'equal' if " +\
                "no region faces outwards. The output must be one of: 'red', 'blue', or 'equal'.\n\n" +\
                "Please ensure your response adheres strictly to this JSON format, including only the specified fields " +\
                "without any additional properties. To summarise, the keys in the JSON object should be 'right', 'up', and 'out'."
            expected_keys = ["right", "up", "out"]

        elif prompt_no in [3, 4, 5]:
            json_schema = "Please provide your response in the following JSON format:\n\n" +\
                "{\n" +\
                '    "reasoning_steps": ["The reasoning steps leading to the final conclusion."],\n' +\
                '    "right": "color",\n' +\
                '    "up": "color",\n' +\
                '    "out": "color"\n' +\
                "}\n\n" +\
                "Where:\n\n" +\
                "reasoning_steps: This field should contain an array of strings representing the reasoning steps leading to " +\
                "the final conclusion. Use your understanding of 3D scene geometry to reason.\n\n" +\
                "right: This field should contain a string representing the color of the region that faces right (from the " +\
                "point of view of the viewer). If both regions equally face right, output 'equal'. If a region faces left, " +\
                "output the color of the other region. The output must be one of: 'red', 'blue', or 'equal'.\n\n" +\
                "up: This field should contain a string representing the color of the region that faces up. If both regions " +\
                "equally face up, output 'equal'. If a region faces down, output the color of the other region. The output " +\
                "must be one of: 'red', 'blue', or 'equal'.\n\n" +\
                "out: This field should contain a string representing the color of the region that faces out of the image. " +\
                "If both regions equally face out of the image, output 'equal'. Note that a region facing inwards won't be " +\
                "visible to the camera. Therefore, output the color of the region that faces outwards, or output 'equal' if " +\
                "no region faces outwards. The output must be one of: 'red', 'blue', or 'equal'.\n\n" +\
                "Please ensure your response adheres strictly to this JSON format, including only the specified fields " +\
                "without any additional properties. To summarise, the keys in the JSON object should be 'reasoning_steps', " +\
                "'right', 'up', and 'out'."
            expected_keys = ["reasoning_steps", "right", "up", "out"]

        json_schema = (json_schema, expected_keys)

    else:
        if prompt_no in [1, 2]:
            json_schema = """{"right": """
        elif prompt_no in [3, 4, 5]:
            json_schema = """{"reasoning_steps": ["""
            
    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_normals(prompt_no: int, model: str, shape: str):
    messages = []

    system_prompt = system_prompts_normals(prompt_no, shape)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_1 = f"""Here is the full image with the two {shape}s. """
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_1}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    json_schema = json_schema_normals(prompt_no, model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


@MFMWrapper.register_task('normals')
def normals(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_samples: int = 200,
    n_segments: int = 200,
    shape: str = "rectangle",
    n_threads: int = 20,
    return_dict: bool = False
):
    """Finds the surface normals of an image using the MFM.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to estimate normals. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
        prompt: The prompt to use for normal estimation
        prompt_no: The prompt number to use (if prompt is None).
        n_samples: The number of samples to generate.
        n_segments: The number of segments to split the image into (using SLIC). The actual number of segments will be close but may be different.
        shape: The shape of the visual marker to use for demarcating the regions.
        n_threads: The number of threads to use for parallel processing.
        return_dict: Whether to return the result as a list of dictionaries.

    Returns:
        (if return_dict is True)
        resp_list: List of dicts, each containing the "right", "up", and "out" values.
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        resp_list: List of normal maps normalized to 0-1 (np.ndarray) (display using plt.imshow())
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    print(f'{shape}')
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path='temp_images')

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    compl_tokens, prompt_tokens = 0, 0
    resp_dict_list, error_status = [], False

    for img_idx, img in enumerate(imgs):
        segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
        segment_pairs = sample_segments(segments, min_samples=n_samples)

        def process_segment_pair(index, seg_pair):
            img_boundaries = deepcopy(img)
            for color, seg in zip(["red", "blue"], seg_pair):
                seg_mask = np.zeros_like(segments)
                seg_mask[segments == seg] = 1
                img_boundaries = draw_around_superpixel(img_boundaries, segments, seg, shape, color)

            full_prompt = full_prompt_normals(prompt_no, model.name, shape) if not prompt else prompt
            full_prompt = replace_images_in_prompt(full_prompt, [img_boundaries])

            resp_dict, tokens, error_status = model.send_message(full_prompt)

            if not error_status:
                resp_dict.pop("reasoning_steps", None)
            else:
                resp_dict = {"right": "red", "up": "red", "out": "red"}
                return index, resp_dict, tokens, error_status

            return index, resp_dict, tokens, error_status

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(process_segment_pair, i, seg_pair) for i, seg_pair in enumerate(segment_pairs)]
            results = [None] * len(futures)
            display_pbar = True if return_dict else False

            for future in tqdm(as_completed(futures), total=len(futures), disable=not display_pbar):
                idx, result, (res_compl_tokens, res_prompt_tokens), err = future.result()
                results[idx] = result
                compl_tokens += res_compl_tokens
                prompt_tokens += res_prompt_tokens
                if err:
                    error_status = True

        seg_pairs = [[int(seg) for seg in seg_pair] for seg_pair in segment_pairs]
        resp_dict_list.append({"normal_orders": results, "segment_pairs": seg_pairs, "file_name": file_name[img_idx].strip()})

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        normal_maps = model.eval(eval='eval_normals', predictions=resp_dict_list, n_segments=n_segments, visualise=True)
        return normal_maps, (compl_tokens, prompt_tokens)
