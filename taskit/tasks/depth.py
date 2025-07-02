import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from io import BytesIO
from typing import Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.segmentation import slic
from tqdm import tqdm

from taskit.mfm import (
    TextMFMWrapper,
    ImageMFMWrapper,
    OPENAI_MODELS,
    GEMINI_MODELS,
    CLAUDE_MODELS,
    TOGETHER_MODELS,
    QWEN2_MODELS,
)
from taskit.utils.data import (
    replace_images_in_prompt,
    sample_segments,
    draw_around_superpixel,
    save_images,
    crop_img,
)


# --System Prompt----------------------------------------------------------------


def system_prompts_depth(prompt_no: int, model: str) -> str:
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    if model not in GEMINI_MODELS:
        if prompt_no == 1:
            system_prompt = (
                """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """
                + """Output your answer as a JSON in the following format:\n"""
                + """{\n"""
                + """  "reasoning_steps": [\n"""
                + """  "Step 1: ...",\n"""
                + """  "Step 2: ..."\n"""
                + """ ...],\n"""
                + """  "depth_order": ["color 1", "color 2"]\n"""
                + """}\n\n"""
                + """Important Notes:\n"""
                + """1. The depth_order should list the nearest region first and the farthest region last.\n"""
                + """2. The output color names should be either 'red' or 'blue'.\n"""
                + """3. The reasoning_steps should clearly outline the logical process leading to your final conclusion.\n"""
                + """4. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n"""
                + """5. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n"""
                + """6. If the depth order is unclear, explain why in your reasoning steps and provide your best estimate.\n"""
                + """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""
            )

        elif prompt_no == 2:
            system_prompt = (
                """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n"""
                + """### Steps to Approach the Task\n"""
                + """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n"""
                + """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n"""
                + """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your reasoning strictly on visual evidence.\n\n"""
                + """### Provide Your Conclusion\n"""
                + """Once you have analyzed the images, output your reasoning and conclusion in the following JSON format:\n"""
                + """{\n"""
                + """  "reasoning_steps": [\n"""
                + """    "Step 1: ...",\n"""
                + """    "Step 2: ...",\n"""
                + """    ...\n"""
                + """  ],\n"""
                + """  "depth_order": ["nearest region", "farthest region"]\n"""
                + """}\n\n"""
                + """### Guidelines\n"""
                + """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n"""
                + """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n"""
                + """- **Uncertainty**: If it is not clear which region is closer, explain your reasoning and provide your best estimate based on the visual information you have.\n\n"""
                + """Ensure that your final decision is supported by visual evidence and that your reasoning is clearly articulated in each step."""
            )

        elif prompt_no == 3:
            system_prompt = (
                """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """
                + """Output your answer as a JSON in the following format:\n"""
                + """{\n"""
                + """  "near": "color of the nearer region",\n"""
                + """}\n\n"""
                + """Important Notes:\n"""
                + """1. The output color name should be either 'red' or 'blue'.\n"""
                + """2. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n"""
                + """3. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n"""
                + """4. If the depth order is unclear, make an educated guess based on the available evidence.\n"""
                + """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""
            )
        elif prompt_no == 4:
            system_prompt = (
                """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """
                + """Output your answer as a JSON in the following format:\n"""
                + """{\n"""
                + """  "depth_order": ["color 1", "color 2"]\n"""
                + """}\n\n"""
                + """Important Notes:\n"""
                + """1. The depth_order should list the nearest region first and the farthest region last.\n"""
                + """2. The output color names should be either 'red' or 'blue'.\n"""
                + """3. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n"""
                + """4. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n"""
                + """5. If the depth order is unclear, provide your best estimate.\n"""
                + """Provide your response in the specified JSON format."""
            )
        elif prompt_no == 5:
            system_prompt = (
                """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n"""
                + """### Steps to Approach the Task\n"""
                + """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n"""
                + """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n"""
                + """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your decision strictly on visual evidence.\n\n"""
                + """### Provide Your Conclusion\n"""
                + """Once you have analyzed the images, output your conclusion in the following JSON format:\n"""
                + """{\n"""
                + """  "depth_order": ["nearest region", "farthest region"]\n"""
                + """}\n\n"""
                + """### Guidelines\n"""
                + """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n"""
                + """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n"""
                + """- **Uncertainty**: If it is not clear which region is closer, provide your best estimate based on the visual information you have."""
            )

    else:
        if prompt_no == 1:
            system_prompt = (
                """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """
                + """Output your answer as a JSON in the following format:\n"""
                + """{\n"""
                + """  "reasoning_steps": [\n"""
                + """  "Step 1: ...",\n"""
                + """  "Step 2: ..."\n"""
                + """ ...],\n"""
                + """  "near": "color of the nearer region"\n"""
                + """}\n\n"""
                + """Important Notes:\n"""
                + """1. The "near" field should list the nearest region.\n"""
                + """2. The output color should be either 'red' or 'blue'.\n"""
                + """3. The reasoning_steps should clearly outline the logical process leading to your final conclusion.\n"""
                + """4. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n"""
                + """5. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n"""
                + """6. If the depth order is unclear, explain why in your reasoning steps and provide your best estimate.\n"""
                + """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""
            )

        elif prompt_no == 2:
            system_prompt = (
                """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n"""
                + """### Steps to Approach the Task\n"""
                + """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n"""
                + """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n"""
                + """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your reasoning strictly on visual evidence.\n\n"""
                + """### Provide Your Conclusion\n"""
                + """Once you have analyzed the images, output your reasoning and conclusion in the following JSON format:\n"""
                + """{\n"""
                + """  "reasoning_steps": [\n"""
                + """    "Step 1: ...",\n"""
                + """    "Step 2: ...",\n"""
                + """    ...\n"""
                + """  ],\n"""
                + """  "near": "color of the nearer region"\n"""
                + """}\n\n"""
                + """### Guidelines\n"""
                + """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n"""
                + """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n"""
                + """- **Uncertainty**: If it is not clear which region is closer, explain your reasoning and provide your best estimate based on the visual information you have.\n\n"""
                + """Ensure that your final decision is supported by visual evidence and that your reasoning is clearly articulated in each step."""
            )

        elif prompt_no == 3:
            system_prompt = (
                """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """
                + """Output your answer as a JSON in the following format:\n"""
                + """{\n"""
                + """  "near": "color of the nearer region"\n"""
                + """}\n\n"""
                + """Important Notes:\n"""
                + """1. The output color should be either 'red' or 'blue'.\n"""
                + """2. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n"""
                + """3. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n"""
                + """4. If the depth order is unclear, make an educated guess based on the available evidence.\n"""
                + """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""
            )

        elif prompt_no == 4:
            system_prompt = (
                """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """
                + """Output your answer as a JSON in the following format:\n"""
                + """{\n"""
                + """  "near": "color of the nearer region"\n"""
                + """}\n\n"""
                + """Important Notes:\n"""
                + """1. The "near" field should list the nearest region.\n"""
                + """2. The output color names should be either 'red' or 'blue'.\n"""
                + """3. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n"""
                + """4. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n"""
                + """5. If the depth order is unclear, provide your best estimate.\n"""
                + """Provide your response in the specified JSON format."""
            )

        elif prompt_no == 5:
            system_prompt = (
                """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n"""
                + """### Steps to Approach the Task\n"""
                + """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n"""
                + """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n"""
                + """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your decision strictly on visual evidence.\n\n"""
                + """### Provide Your Conclusion\n"""
                + """Once you have analyzed the images, output your conclusion in the following JSON format:\n"""
                + """{\n"""
                + """  "near": "color of the nearer region"\n"""
                + """}\n\n"""
                + """### Guidelines\n"""
                + """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n"""
                + """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n"""
                + """- **Uncertainty**: If it is not clear which region is closer, provide your best estimate based on the visual information you have."""
            )

    return system_prompt


# --JSON Schema----------------------------------------------------------------


def json_schema_depth(prompt_no: int, model: str):
    if model in OPENAI_MODELS:
        if prompt_no in [1, 2]:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": "The step-by-step reasoning process leading to the final conclusion. Use your understanding of 3D scene geometry to reason.",
            }
            json_properties["depth_order"] = {
                "type": "array",
                "items": {"type": "string", "enum": ["red", "blue"]},
                "description": "The order of the regions based on their depth (nearest region first). The list should contain the color names 'red' and 'blue' only.",
            }
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": ["reasoning_steps", "depth_order"],
                "additionalProperties": False,
            }

        elif prompt_no in [3]:
            json_schema, json_properties = {}, {}
            json_properties["near"] = {
                "type": "string",
                "enum": ["red", "blue"],
                "description": "The color of the region that is closer to the camera (at a shallower depth).",
            }
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": ["near"],
                "additionalProperties": False,
            }

        if prompt_no in [4, 5]:
            json_schema, json_properties = {}, {}
            json_properties["depth_order"] = {
                "type": "array",
                "items": {"type": "string", "enum": ["red", "blue"]},
                "description": "The order of the regions based on their depth (nearest region first). The list should contain the color names 'red' and 'blue' only.",
            }
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": ["depth_order"],
                "additionalProperties": False,
            }

    elif model in GEMINI_MODELS:
        if prompt_no in [1, 2]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "reasoning_steps": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="The step-by-step reasoning process leading to the final conclusion. Use your understanding of 3D scene geometry to reason.",
                    ),
                    "near": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue"],
                        description="The color of the region that is closer to the camera (at a shallower depth).",
                    ),
                },
                required=["reasoning_steps", "near"],
            )

        elif prompt_no in [3, 4, 5]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "near": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue"],
                        description="The color of the region that is closer to the camera (at a shallower depth).",
                    ),
                },
                required=["near"],
            )

    elif model in CLAUDE_MODELS or model in TOGETHER_MODELS:
        if prompt_no in [1, 2]:
            json_schema = (
                "Please provide your response in the following JSON format:\n\n"
                + "{\n"
                + '    "reasoning_steps": [\n'
                + '        "Step 1: Description of the reasoning process",\n'
                + '        "Step 2: Further analysis",\n'
                + '        "..."\n'
                + "    ],\n"
                + '    "depth_order": ["color1", "color2"]\n'
                + "}\n\n"
                + "Where:\n\n"
                + "reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process "
                + "leading to the final conclusion. Use your understanding of 3D scene geometry to reason about the depth "
                + "relationships between the red and blue regions in the image.\n\n"
                + "depth_order: This field should contain a list of two strings representing the order of the regions based on "
                + "their depth, with the nearest region first. The list should only contain the color names 'red' and 'blue'.\n\n"
                + "Please ensure your response adheres strictly to this JSON format, including only the specified fields "
                + "without any additional properties."
            )
            expected_keys = ["reasoning_steps", "depth_order"]

        elif prompt_no in [3]:
            json_schema = (
                "Please provide your response in the following JSON format:\n\n"
                + "{\n"
                + '    "near": "color"\n'
                + "}\n\n"
                + "Where:\n\n"
                + "near: This field should contain a string representing the color of the region that is closer to the camera "
                + "(at a shallower depth). The color should be either 'red' or 'blue'.\n\n"
                + "Please ensure your response adheres strictly to this JSON format, including only the specified fields "
                + "without any additional properties."
            )
            expected_keys = ["near"]

        elif prompt_no in [4, 5]:
            json_schema = (
                "Please provide your response in the following JSON format:\n\n"
                + "{\n"
                + '    "depth_order": ["color1", "color2"]\n'
                + "}\n\n"
                + "Where:\n\n"
                + "depth_order: This field should contain a list of two strings representing the order of the regions based on "
                + "their depth, with the nearest region first. The list should only contain the color names 'red' and 'blue'.\n\n"
                + "Please ensure your response adheres strictly to this JSON format, including only the specified fields "
                + "without any additional properties."
            )
            expected_keys = ["depth_order"]

        json_schema = (json_schema, expected_keys)

    elif model in QWEN2_MODELS:
        if prompt_no in [1, 2]:
            json_schema = """{"reasoning_steps": ["""
        elif prompt_no in [3]:
            json_schema = """{"near": """
        elif prompt_no in [4, 5]:
            json_schema = """{"depth_order": ["""
    else:
        raise ValueError(f"Model {model} not supported.")

    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_depth(prompt_no: int, model: str):
    messages = []

    system_prompt = system_prompts_depth(prompt_no, model)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_1 = (
        "Here is the full image with the two regions (demarcated in red and blue)."
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}},
                {"type": "text", "text": user_prompt_1},
            ],
        }
    )
    user_prompt_2 = "Output the regions in the order of depth (near to far)"
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": user_prompt_2}]}
    )

    json_schema = json_schema_depth(prompt_no, model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


@TextMFMWrapper.register_task("depth")
def depth(
    model: TextMFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_samples: int = 200,
    n_segments: int = 100,
    shape: str = "point",
    shuffle: bool = True,
    n_threads: int = 20,
    return_dict: bool = False,
):
    """Find the depth map of an image using the MFM.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to estimate the depth. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
        prompt: The prompt to use for depth prediction.
        prompt_no: The prompt number to use (if prompt is None).
        n_samples: The total number of samples in the depth map.
        n_segments: The number of segments to split the image into (using SLIC). The actual number of segments will be close but may be different.
        shape: The shape of the visual marker.
        shuffle: Whether to shuffle the order of the segments.
        n_threads: The number of threads to use for parallel processing.
        return_dict: Whether to return the result as a list of dictionaries.

    Returns:
        (if return_dict is True)
        resp_list: List of dicts, each containing "depth_orders" and the corresponding "segment_pairs"
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        resp_list: List of depth maps normalized to 0-1 (np.ndarray) (display using plt.imshow())
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path="temp_images")

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert("RGB") for fn in file_name]

    compl_tokens, prompt_tokens = 0, 0
    resp_dict_list, error_status = [], False

    for img_idx, img in enumerate(imgs):
        segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
        segment_pairs = sample_segments(
            segments, min_samples=n_samples, shuffle=shuffle
        )

        def process_segment_pair(index, seg_pair):
            img_boundaries = deepcopy(img)
            for color, seg in zip(["red", "blue"], seg_pair):
                seg_mask = np.zeros_like(segments)
                seg_mask[segments == seg] = 1
                img_boundaries = draw_around_superpixel(
                    img_boundaries, segments, seg, shape, color
                )

            full_prompt = (
                full_prompt_depth(prompt_no, model.name) if not prompt else prompt
            )
            full_prompt = replace_images_in_prompt(full_prompt, [img_boundaries])

            resp_dict, tokens, error_status = model.send_message(full_prompt)

            if not error_status:
                resp_dict.pop("reasoning_steps", None)
                if "near" in resp_dict:
                    resp_dict["depth_order"] = [
                        resp_dict["near"],
                        "red" if resp_dict["near"] == "blue" else "blue",
                    ]
                    resp_dict.pop("near")
            if error_status:
                resp_dict = {"depth_order": ["red", "blue"]}
                return index, resp_dict, tokens, error_status

            return index, resp_dict, tokens, error_status

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(process_segment_pair, i, seg_pair)
                for i, seg_pair in enumerate(segment_pairs)
            ]
            results = [None] * len(futures)
            display_pbar = True if return_dict else False

            for future in tqdm(
                as_completed(futures), total=len(futures), disable=not display_pbar
            ):
                idx, result, (res_compl_tokens, res_prompt_tokens), err = (
                    future.result()
                )
                results[idx] = result
                compl_tokens += res_compl_tokens
                prompt_tokens += res_prompt_tokens
                if err:
                    error_status = True

        seg_pairs = [[int(seg) for seg in seg_pair] for seg_pair in segment_pairs]
        resp_dict_list.append(
            {
                "depth_orders": results,
                "segment_pairs": seg_pairs,
                "file_name": file_name[img_idx].strip(),
            }
        )

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        depth_maps = model.eval(
            eval="eval_depth",
            predictions=resp_dict_list,
            n_segments=n_segments,
            visualise=True,
        )
        return depth_maps, (compl_tokens, prompt_tokens)


@ImageMFMWrapper.register_task("dense_depth")
def dense_depth(
    model: ImageMFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    *,
    center_crop: bool = False,
    save_dir: str = "./4o-imagegen/depth-preds/",
    return_dict: bool = False,
    reverse: bool = True,
):
    """
    Create a dense depth map of an image using the MFM.

    Args:
        model: The MFM model to use.
        file_name: Path(s) or PIL.Image(s) to run depth on.
        center_crop: if True, center-crop before sending.
        save_dir: Directory to save the generated depth maps.
        return_dict: If True, returns a list of dicts with "depth_image" and "file_name".
        reverse: If True, reverses the depth map (white = near, black = far). If False, black = near, white = far.

    Returns:
        (if return_dict is True)
        outputs: List of dicts, each containing "depth_image" and the corresponding "file_name".
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        depth_imgs: List of depth maps
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path="temp_images")
    file_list = file_name if isinstance(file_name, list) else [file_name]

    if not reverse:
        depth_prompt = """
        Generate a **pure grayscale depth map** from the input image. The grayscale values must encode depth as follows:

        - **Black (0 intensity)** represents points that are **closest** to the camera (minimum depth).
        - **White (255 intensity)** represents points that are **farthest** from the camera (maximum depth).
        - All other points must be shaded **monotonically between black and white**, based solely on their distance from the camera.

        This map must not contain any colors, textures, or artistic effects — only smooth grayscale transitions that accurately reflect increasing depth, with lighter shades at greater distances.
        """
    else:
        depth_prompt = """
        Generate a **pure grayscale depth map** from the input image. The grayscale values must encode depth as follows:

        - **White (255 intensity)**  represents points that are **closest** to the camera (minimum depth).
        - **Black (0 intensity)** represents points that are **farthest** from the camera (maximum depth).
        - All other points must be shaded **monotonically between black and white**, based solely on their distance from the camera.

        This map must not contain any colors, textures, or artistic effects — only smooth grayscale transitions that accurately reflect increasing depth, with darker shades at greater distances.
        """

    compl_tokens = prompt_tokens = 0
    outputs, depth_imgs = [], []
    error_status = False

    for fn in file_list:
        img = Image.open(fn.strip()).convert("RGB")

        # either center-crop
        if center_crop:
            shortest_side = img.size[0] if img.size[0] < img.size[1] else img.size[1]
            crop_size = shortest_side
            proc = crop_img(img, crop_size=crop_size, shortest_side=shortest_side)
        else:
            # or pad to 1024×1024
            proc = Image.new("RGB", (1024, 1024), (0, 0, 0))
            proc.paste(img, (0, 128))

        buf = BytesIO()
        proc.save(buf, format="PNG")
        buf.name = "image.png"
        buf.seek(0)

        resp_b64, (ct, pt), err = model.send_message(prompt=depth_prompt, image=[buf])
        compl_tokens += ct
        prompt_tokens += pt
        error_status |= err

        # decode result
        if not err:
            depth_img = Image.open(BytesIO(base64.b64decode(resp_b64))).convert("L")
            if reverse:
                depth_img = -np.array(depth_img) + 255
                depth_img = Image.fromarray(depth_img, mode="L")
            if not center_crop:
                depth_img = depth_img.crop((0, 128, 1024, 896))
        else:
            depth_img = Image.new(
                "RGB", (crop_size, crop_size) if center_crop else (1024, 768), (0, 0, 0)
            )

        out_name = os.path.basename(fn.strip())
        folder_name = fn.strip().split("/")[-2]
        os.makedirs(save_dir, exist_ok=True)  # Ensure save_dir exists
        out_path = f"{save_dir}{folder_name}-{out_name}"
        depth_img.save(out_path)

        outputs.append({"depth_image": out_path, "file_name": fn.strip()})
        depth_imgs.append(depth_img)

    if return_dict:
        return outputs, (compl_tokens, prompt_tokens), error_status
    else:
        if center_crop:
            depth_imgs = [np.array(img, dtype=np.float32) / 255.0 for img in depth_imgs]
            return depth_imgs, (compl_tokens, prompt_tokens)
        else:
            return depth_imgs, (compl_tokens, prompt_tokens)
