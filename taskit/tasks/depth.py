from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.segmentation import slic
from tqdm import tqdm

from taskit.eval import eval_depth
from taskit.mfm import MFMWrapper
from taskit.utils.data import replace_images_in_prompt, sample_segments, draw_around_superpixel, save_images


# --System Prompt----------------------------------------------------------------


def system_prompts_depth(prompt_no: int, model: str) -> str:
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    if model != 'gemini-1.5-pro':
        if prompt_no == 1:
            system_prompt = """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """ +\
                            """Output your answer as a JSON in the following format:\n""" +\
                            """{\n""" +\
                            """  "reasoning_steps": [\n""" +\
                            """  "Step 1: ...",\n""" +\
                            """  "Step 2: ..."\n""" +\
                            """ ...],\n""" +\
                            """  "depth_order": ["color 1", "color 2"]\n""" +\
                            """}\n\n""" +\
                            """Important Notes:\n""" +\
                            """1. The depth_order should list the nearest region first and the farthest region last.\n""" +\
                            """2. The output color names should be either 'red' or 'blue'.\n""" +\
                            """3. The reasoning_steps should clearly outline the logical process leading to your final conclusion.\n""" +\
                            """4. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n""" +\
                            """5. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n""" +\
                            """6. If the depth order is unclear, explain why in your reasoning steps and provide your best estimate.\n""" +\
                            """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""

        elif prompt_no == 2:
            system_prompt = """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n""" +\
                            """### Steps to Approach the Task\n""" +\
                            """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n""" +\
                            """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n""" +\
                            """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your reasoning strictly on visual evidence.\n\n""" +\
                            """### Provide Your Conclusion\n""" +\
                            """Once you have analyzed the images, output your reasoning and conclusion in the following JSON format:\n""" +\
                            """{\n""" +\
                            """  "reasoning_steps": [\n""" +\
                            """    "Step 1: ...",\n""" +\
                            """    "Step 2: ...",\n""" +\
                            """    ...\n""" +\
                            """  ],\n""" +\
                            """  "depth_order": ["nearest region", "farthest region"]\n""" +\
                            """}\n\n""" +\
                            """### Guidelines\n""" +\
                            """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n""" +\
                            """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n""" +\
                            """- **Uncertainty**: If it is not clear which region is closer, explain your reasoning and provide your best estimate based on the visual information you have.\n\n""" +\
                            """Ensure that your final decision is supported by visual evidence and that your reasoning is clearly articulated in each step."""

        elif prompt_no == 3:
            system_prompt = """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """ +\
                            """Output your answer as a JSON in the following format:\n""" +\
                            """{\n""" +\
                            """  "near": "color of the nearer region",\n""" +\
                            """}\n\n""" +\
                            """Important Notes:\n""" +\
                            """1. The output color name should be either 'red' or 'blue'.\n""" +\
                            """2. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n""" +\
                            """3. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n""" +\
                            """4. If the depth order is unclear, make an educated guess based on the available evidence.\n""" +\
                            """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""
        elif prompt_no == 4:
            system_prompt = """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """ +\
                            """Output your answer as a JSON in the following format:\n""" +\
                            """{\n""" +\
                            """  "depth_order": ["color 1", "color 2"]\n""" +\
                            """}\n\n""" +\
                            """Important Notes:\n""" +\
                            """1. The depth_order should list the nearest region first and the farthest region last.\n""" +\
                            """2. The output color names should be either 'red' or 'blue'.\n""" +\
                            """3. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n""" +\
                            """4. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n""" +\
                            """5. If the depth order is unclear, provide your best estimate.\n""" +\
                            """Provide your response in the specified JSON format."""
        elif prompt_no == 5:
            system_prompt = """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n""" +\
                            """### Steps to Approach the Task\n""" +\
                            """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n""" +\
                            """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n""" +\
                            """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your decision strictly on visual evidence.\n\n""" +\
                            """### Provide Your Conclusion\n""" +\
                            """Once you have analyzed the images, output your conclusion in the following JSON format:\n""" +\
                            """{\n""" +\
                            """  "depth_order": ["nearest region", "farthest region"]\n""" +\
                            """}\n\n""" +\
                            """### Guidelines\n""" +\
                            """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n""" +\
                            """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n""" +\
                            """- **Uncertainty**: If it is not clear which region is closer, provide your best estimate based on the visual information you have."""

    else:
        if prompt_no == 1:
            system_prompt = """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """ +\
                            """Output your answer as a JSON in the following format:\n""" +\
                            """{\n""" +\
                            """  "reasoning_steps": [\n""" +\
                            """  "Step 1: ...",\n""" +\
                            """  "Step 2: ..."\n""" +\
                            """ ...],\n""" +\
                            """  "near": "color of the nearer region"\n""" +\
                            """}\n\n""" +\
                            """Important Notes:\n""" +\
                            """1. The "near" field should list the nearest region.\n""" +\
                            """2. The output color should be either 'red' or 'blue'.\n""" +\
                            """3. The reasoning_steps should clearly outline the logical process leading to your final conclusion.\n""" +\
                            """4. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n""" +\
                            """5. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n""" +\
                            """6. If the depth order is unclear, explain why in your reasoning steps and provide your best estimate.\n""" +\
                            """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""

        elif prompt_no == 2:
            system_prompt = """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n""" +\
                            """### Steps to Approach the Task\n""" +\
                            """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n""" +\
                            """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n""" +\
                            """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your reasoning strictly on visual evidence.\n\n""" +\
                            """### Provide Your Conclusion\n""" +\
                            """Once you have analyzed the images, output your reasoning and conclusion in the following JSON format:\n""" +\
                            """{\n""" +\
                            """  "reasoning_steps": [\n""" +\
                            """    "Step 1: ...",\n""" +\
                            """    "Step 2: ...",\n""" +\
                            """    ...\n""" +\
                            """  ],\n""" +\
                            """  "near": "color of the nearer region"\n""" +\
                            """}\n\n""" +\
                            """### Guidelines\n""" +\
                            """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n""" +\
                            """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n""" +\
                            """- **Uncertainty**: If it is not clear which region is closer, explain your reasoning and provide your best estimate based on the visual information you have.\n\n""" +\
                            """Ensure that your final decision is supported by visual evidence and that your reasoning is clearly articulated in each step."""

        elif prompt_no == 3:
            system_prompt = """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """ +\
                            """Output your answer as a JSON in the following format:\n""" +\
                            """{\n""" +\
                            """  "near": "color of the nearer region"\n""" +\
                            """}\n\n""" +\
                            """Important Notes:\n""" +\
                            """1. The output color should be either 'red' or 'blue'.\n""" +\
                            """2. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n""" +\
                            """3. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n""" +\
                            """4. If the depth order is unclear, make an educated guess based on the available evidence.\n""" +\
                            """Provide your response in the specified JSON format, ensuring that your reasoning is clear and well-supported by the visual evidence in the image."""

        elif prompt_no == 4:
            system_prompt = """You will be provided with a full image containing two regions, one demarcated in red and the other in blue. Your task is to determine which region is closer to the camera, or in other words, which region is at a shallower depth.\n """ +\
                            """Output your answer as a JSON in the following format:\n""" +\
                            """{\n""" +\
                            """  "near": "color of the nearer region"\n""" +\
                            """}\n\n""" +\
                            """Important Notes:\n""" +\
                            """1. The "near" field should list the nearest region.\n""" +\
                            """2. The output color names should be either 'red' or 'blue'.\n""" +\
                            """3. Be sure to distinguish between depth (the distance of an object perpendicular to the image plane) and lateral distance (the distance between objects along the image plane). Focus on spatial relationships to deduce depth accurately.\n""" +\
                            """4. Consider visual cues such as occlusion, relative size, perspective, and other indicators of depth. Avoid assumptions about specific regions and instead rely on objective evidence from the image.\n""" +\
                            """5. If the depth order is unclear, provide your best estimate.\n""" +\
                            """Provide your response in the specified JSON format."""

        elif prompt_no == 5:
            system_prompt = """You are tasked with determining which of two regions in an image, one marked in red and the other in blue, is closer to the camera (i.e., at a shallower depth). Your decision should be based on a thorough analysis of visual depth cues in the provided images, without making assumptions about object types or real-world scene layouts.\n\n""" +\
                            """### Steps to Approach the Task\n""" +\
                            """1. **Analyze the Full Image**: Start by observing the full image that contains both regions. Look for any depth-related visual cues such as perspective lines, lighting and shadows, and relative sizes. Consider whether either region appears to be positioned closer based on these factors.\n""" +\
                            """2. **Examine Individual Regions**: Next, review the close-up views of the red and blue regions. Look at details like the texture, sharpness, or clarity of each region, as closer objects may appear more detailed. Consider whether these factors provide additional information about which region might be nearer to the camera.\n""" +\
                            """3. **Synthesize Visual Information**: Combine the insights from the full image and the close-ups to form a complete picture of the scene's depth. Pay attention to how lighting, size, and perspective contribute to the spatial relationships between the two regions. Avoid relying on typical assumptions about scene layouts (e.g., "ceilings are far away" or "floors are closer")—base your decision strictly on visual evidence.\n\n""" +\
                            """### Provide Your Conclusion\n""" +\
                            """Once you have analyzed the images, output your conclusion in the following JSON format:\n""" +\
                            """{\n""" +\
                            """  "near": "color of the nearer region"\n""" +\
                            """}\n\n""" +\
                            """### Guidelines\n""" +\
                            """- **Depth Cues**: Focus on factors like perspective, relative size, texture sharpness, lighting, and shading to determine which region is closer.\n""" +\
                            """- **Depth Definition**: Remember, depth refers to distance along the line perpendicular to the camera's view (closer or farther from the camera), not just side-to-side position in the image.\n""" +\
                            """- **Uncertainty**: If it is not clear which region is closer, provide your best estimate based on the visual information you have."""

    return system_prompt


# --JSON Schema----------------------------------------------------------------


def json_schema_depth(prompt_no: int, model: str):
    if model == 'gpt-4o-2024-08-06':
        if prompt_no in [1, 2]:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {"type": "array", "items": {"type": "string"}, "description": "The step-by-step reasoning process leading to the final conclusion. Use your understanding of 3D scene geometry to reason."}
            json_properties["depth_order"] = {"type": "array", "items": {"type": "string", "enum": ["red", "blue"]}, "description": "The order of the regions based on their depth (nearest region first). The list should contain the color names 'red' and 'blue' only."}
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["reasoning_steps", "depth_order"], "additionalProperties": False}

        elif prompt_no in [3]:
            json_schema, json_properties = {}, {}
            json_properties["near"] = {"type": "string", "enum": ["red", "blue"], "description": "The color of the region that is closer to the camera (at a shallower depth)."}
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["near"], "additionalProperties": False}

        if prompt_no in [4, 5]:
            json_schema, json_properties = {}, {}
            json_properties["depth_order"] = {"type": "array", "items": {"type": "string", "enum": ["red", "blue"]}, "description": "The order of the regions based on their depth (nearest region first). The list should contain the color names 'red' and 'blue' only."}
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["depth_order"], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        if prompt_no in [1, 2]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "reasoning_steps": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="The step-by-step reasoning process leading to the final conclusion. Use your understanding of 3D scene geometry to reason."
                    ),
                    "near": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue"],
                        description="The color of the region that is closer to the camera (at a shallower depth)."
                    ),
                },
                required=["reasoning_steps", "near"]
            )

        elif prompt_no in [3, 4, 5]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "near": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["red", "blue"],
                        description="The color of the region that is closer to the camera (at a shallower depth)."
                    ),
                },
                required=["near"]
            )

    elif model == 'claude-3-5-sonnet-20240620':
        if prompt_no in [1, 2]:
            json_schema = "Please provide your response in the following JSON format:\n\n" +\
                "{\n" +\
                '    "reasoning_steps": [\n' +\
                '        "Step 1: Description of the reasoning process",\n' +\
                '        "Step 2: Further analysis",\n' +\
                '        "..."\n' +\
                "    ],\n" +\
                '    "depth_order": ["color1", "color2"]\n' +\
                "}\n\n" +\
                "Where:\n\n" +\
                "reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process " +\
                "leading to the final conclusion. Use your understanding of 3D scene geometry to reason about the depth " +\
                "relationships between the red and blue regions in the image.\n\n" +\
                "depth_order: This field should contain a list of two strings representing the order of the regions based on " +\
                "their depth, with the nearest region first. The list should only contain the color names 'red' and 'blue'.\n\n" +\
                "Please ensure your response adheres strictly to this JSON format, including only the specified fields " +\
                "without any additional properties."
            expected_keys = ["reasoning_steps", "depth_order"]

        elif prompt_no in [3]:
            json_schema = "Please provide your response in the following JSON format:\n\n" +\
                "{\n" +\
                '    "near": "color"\n' +\
                "}\n\n" +\
                "Where:\n\n" +\
                "near: This field should contain a string representing the color of the region that is closer to the camera " +\
                "(at a shallower depth). The color should be either 'red' or 'blue'.\n\n" +\
                "Please ensure your response adheres strictly to this JSON format, including only the specified fields " +\
                "without any additional properties."
            expected_keys = ["near"]

        elif prompt_no in [4, 5]:
            json_schema = "Please provide your response in the following JSON format:\n\n" +\
                "{\n" +\
                '    "depth_order": ["color1", "color2"]\n' +\
                "}\n\n" +\
                "Where:\n\n" +\
                "depth_order: This field should contain a list of two strings representing the order of the regions based on " +\
                "their depth, with the nearest region first. The list should only contain the color names 'red' and 'blue'.\n\n" +\
                "Please ensure your response adheres strictly to this JSON format, including only the specified fields " +\
                "without any additional properties."
            expected_keys = ["depth_order"]

        json_schema = (json_schema, expected_keys)

    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_depth(prompt_no: int, model: str):
    messages = []

    system_prompt = system_prompts_depth(prompt_no, model)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_1 = "Here is the full image with the two regions (demarcated in red and blue)."
    messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}, {"type": "text", "text": user_prompt_1}]})
    user_prompt_2 = "Output the regions in the order of depth (near to far)"
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_2}]})

    json_schema = json_schema_depth(prompt_no, model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


@MFMWrapper.register_task('depth')
def depth(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_samples: int = 200,
    n_segments: int = 100,
    shape: str = "point",
    shuffle: bool = True,
    n_threads: int = 20,
    return_dict: bool = False
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
        file_name = save_images(file_name, save_path='temp_images')

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    compl_tokens, prompt_tokens = 0, 0
    resp_dict_list, error_status = [], False

    for img_idx, img in enumerate(imgs):
        segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
        segment_pairs = sample_segments(segments, min_samples=n_samples, shuffle=shuffle)

        def process_segment_pair(index, seg_pair):
            img_boundaries = deepcopy(img)
            for color, seg in zip(["red", "blue"], seg_pair):
                seg_mask = np.zeros_like(segments)
                seg_mask[segments == seg] = 1
                img_boundaries = draw_around_superpixel(img_boundaries, segments, seg, shape, color)

            full_prompt = full_prompt_depth(prompt_no, model.name) if not prompt else prompt
            full_prompt = replace_images_in_prompt(full_prompt, [img_boundaries])

            resp_dict, tokens, error_status = model.send_message(full_prompt)

            if not error_status:
                resp_dict.pop("reasoning_steps", None)
                if "near" in resp_dict:
                    resp_dict["depth_order"] = [resp_dict["near"], "red" if resp_dict["near"] == "blue" else "blue"]
                    resp_dict.pop("near")
            if error_status:
                return index, None, tokens, error_status

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
        resp_dict_list.append({"depth_orders": results, "segment_pairs": seg_pairs, "file_name": file_name[img_idx].strip()})

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        depth_maps = model.eval(eval='eval_depth', predictions=resp_dict_list, n_segments=n_segments, visualise=True)
        return depth_maps, (compl_tokens, prompt_tokens)
