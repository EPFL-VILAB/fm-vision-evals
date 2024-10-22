from copy import deepcopy
from typing import Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from PIL import Image

from taskit.eval import eval_object
from taskit.mfm import MFMWrapper
from taskit.tasks import classify
from taskit.utils.data import replace_images_in_prompt
from taskit.utils.data_constants import COCO_DETECT_LABELS


# --System Prompt----------------------------------------------------------------


def system_prompts_od(prompt_no: int, obj: str) -> str:

    if prompt_no == 1:
        system_prompt = f"""You are an advanced object detection model. You will be provided with a full image for context, followed by a specific grid cell from that image. Your task is to determine whether any part of the {obj} is present in the given grid cell.\n\n""" +\
                        """First, you'll see the full image to understand the overall context. Then, you'll be shown a specific grid cell, which is a section of the full image. You need to focus your analysis on this grid cell.\n\n""" +\
                        """Output your reasoning steps and your final conclusion in the following format:\n""" +\
                        """{\n""" +\
                        """    "reasoning_steps": [list the steps of your reasoning process],\n""" +\
                        f"""    "1": "answer" (respond with "yes" if any part of the {obj} is present in the grid cell, or "no" if it isn't)\n""" +\
                        """}\n\n""" +\
                        f"""Be thorough in your analysis, but focus solely on the presence or absence of the {obj} in the given grid cell. If you're not sure, or the grid-cell is too small, output "yes"."""

    elif prompt_no == 2:
        system_prompt = """You are an advanced object detection model. You will be provided with a full image for context, followed by a specific grid cell from that image. Your task is to determine whether any part of the specified object is present in the given grid cell.\n""" +\
                        """The object you are looking for is:\n""" +\
                        f"""<object>{obj}</object>\n\n""" +\
                        """First, you'll see the full image to understand the overall context. Then, you'll be shown a specific grid cell, which is a section of the full image. You need to focus your analysis on this grid cell.\n\n""" +\
                        """Follow these steps for the grid cell:\n""" +\
                        """1. Carefully examine the contents of the cell.\n""" +\
                        """2. Look for any features or parts that could belong to the target object.\n""" +\
                        """3. Consider partial appearances of the object, not just complete views.\n""" +\
                        """4. Make a decision based on your analysis.\n\n""" +\
                        """As you analyze the cell, document your reasoning process. This will help explain your decisions and ensure a thorough examination of the grid cell.\n""" +\
                        """Present your final output in the following JSON format:\n""" +\
                        """{\n""" +\
                        """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n""" +\
                        """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n""" +\
                        """}\n\n""" +\
                        """Ensure that your reasoning steps are clear, concise, and directly related to the presence or absence of the target object in the grid cell. Be as objective as possible in your analysis, basing your decisions on visual evidence present in the grid cell.\n""" +\
                        """If you're not sure, or the grid cell is too small, output "yes". Your goal is to accurately detect the presence of the specified object, providing a well-reasoned analysis for your decision.\n"""

    elif prompt_no == 3:
        system_prompt = """You are an advanced object detection model with exceptional analytical capabilities. Your task is to detect the presence of a specified object within a provided grid cell.\n\n""" +\
                        """**Target Object**:\n""" +\
                        f"""<object>{obj}</object>\n""" +\
                        """\n""" +\
                        """**Grid Information**:\n""" +\
                        """- You will analyze a grid cell representing a section of a larger image.\n""" +\
                        """- The cell may contain all, part, or none of the target object.\n\n""" +\
                        """**Your Objectives**:\n""" +\
                        """1. **Analyze The Cell**:\n""" +\
                        """   - Examine the visual content of the grid cell carefully.\n""" +\
                        """   - Look for any features, patterns, or fragments associated with the target object.\n""" +\
                        """   - Consider partial appearances, occlusions, rotations, scaling, and variations in lighting or perspective.\n\n""" +\
                        """2. **Determine Presence of the Object**:\n""" +\
                        """   - Decide whether any part of the target object is present in the cell.\n""" +\
                        """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n""" +\
                        """3. **Document Your Reasoning**:\n""" +\
                        """   - Provide clear and concise reasoning for each decision.\n""" +\
                        """   - Your reasoning should focus on the key visual evidence that supports your conclusion.\n\n""" +\
                        """**Output Format**:\n""" +\
                        """Present your findings in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """    "reasoning_steps": [\n""" +\
                        """        "Your reasoning for the cell",\n""" +\
                        """    ],\n""" +\
                        """    "1": "yes" or "no",\n""" +\
                        """}\n""" +\
                        """\n""" +\
                        """**Guidelines**:\n""" +\
                        """- **Be Objective**: Base your analysis solely on the visual content of the cell.\n""" +\
                        """- **Be Concise**: Keep your reasoning for the cell to a few sentences, emphasizing the most significant observations.\n""" +\
                        """- **Ensure Accuracy**: Double-check your conclusions to maintain high accuracy in object detection.\n""" +\
                        """- **Maintain Clarity**: Use clear and direct language in your reasoning.\n\n""" +\
                        """**Remember**:\n""" +\
                        """Your primary goal is to accurately detect the presence of the specified object in the grid cell and provide justifiable reasoning for your decisions. Avoid including any information not pertinent to the task."""

    elif prompt_no == 4:
        system_prompt = """You are an advanced object detection model.\n\n""" +\
                        """**Task**:\n""" +\
                        """Using the full image provided, determine whether any part of the specified object is present in a grid cell.\n\n""" +\
                        """**Object to Detect**:\n""" +\
                        f"""<object>{obj}</object>\n\n""" +\
                        """**Input**:\n""" +\
                        """- You will be given the full image containing the object.\n""" +\
                        """- The image is divided into a 3x3 grid, creating 9 cells numbered from 1 to 9 (left to right, top to bottom). You will be provided one such cell.\n\n""" +\
                        """- This cell may contain all, part, or none of the object.\n\n""" +\
                        """**Instructions**:\n\n""" +\
                        """1. **Analyze the Full Image**:\n""" +\
                        """ - Begin by examining the full image to understand the object's location, size, and features.\n\n""" +\
                        """2. **Evaluate The Grid Cell**:\n""" +\
                        """ - Determine if any part of the object is present within the cell.\n""" +\
                        """ - Look for distinguishing features such as shape, color, texture, or patterns.\n""" +\
                        """ - Consider partial appearances and overlapping regions.\n""" +\
                        """ - Decide whether to label the cell as containing the object ("yes") or not ("no").\n\n""" +\
                        """3. **Document Your Reasoning**:\n""" +\
                        """ - Provide a brief reasoning for each cell, focusing on key observations that led to your decision.\n\n""" +\
                        """**Output Format**:\n\n""" +\
                        """Present your findings in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n""" +\
                        """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n""" +\
                        """}\n\n""" +\
                        """**Example**:\n\n""" +\
                        """{\n""" +\
                        """    "reasoning_steps": [\n""" +\
                        """         "The full image shows the object ...",\n""" +\
                        """    ],\n""" +\
                        """    "1": "yes",\n""" +\
                        """}\n""" +\
                        """**Guidelines**:\n\n""" +\
                        """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n""" +\
                        """- **Clarity**: Keep your reasoning concise and focused on the most significant features.\n""" +\
                        """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n""" +\
                        """- **Consistency**: Ensure your reasoning aligns with your final decision for the cell.\n\n"""

    elif prompt_no == 5:
        system_prompt = """You are an advanced object detection model.\n\n""" +\
                        """**Task**:\n""" +\
                        """- Analyze the provided grid cell from the image to determine if any part of the specified object is present.\n""" +\
                        """- The cell is a section of the full image, and may contain all, part, or none of the object.\n\n""" +\
                        """**Object to Detect**:\n""" +\
                        f"""<object>{obj}</object>\n\n""" +\
                        """**Inputs**:\n""" +\
                        """- **Full Image**: You have the full image to understand the context and specifics of the object.\n""" +\
                        """- **Grid Cell**: You have an individual image of the grid cell extracted from the full image.\n\n""" +\
                        """**Instructions**:\n\n""" +\
                        """1. **Examine the Object in the Full Image**:\n""" +\
                        """   - Understand the object's features: shape, color, texture, patterns, and any distinctive marks.\n\n""" +\
                        """2. **Analyze The Grid Cell Thoroughly**:\n""" +\
                        """   - Look for any visual evidence of the object, even if it's a very small part or a tiny sliver.\n""" +\
                        """   - Consider that the object might be partially visible due to the division of the grid.\n\n""" +\
                        """3. **Decision Criteria**:\n""" +\
                        """   - **Label as "yes"** if any part of the object is present in the cell, regardless of how small.\n""" +\
                        """   - **Label as "no"** if there is no visual evidence of the object in the cell.\n""" +\
                        """   - Base your decision solely on the visual content of the cell image.\n\n""" +\
                        """4. **Document Your Reasoning**:\n""" +\
                        """   - Provide a brief reasoning for the cell.\n""" +\
                        """   - Mention specific features observed that led to your decision.\n""" +\
                        """   - Be precise and focus on the visual evidence.\n\n""" +\
                        """**Output Format**:\n\n""" +\
                        """Present your findings in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n""" +\
                        """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n""" +\
                        """}\n\n""" +\
                        """**Guidelines**:\n\n""" +\
                        """- **Detect Even Small Parts**: If any part of the object is present, no matter how small, label the cell as "yes".\n""" +\
                        """- **Precision**: Do not assume the object's presence without clear visual confirmation, even if adjacent cells contain the object.\n""" +\
                        """- **Clarity**: Keep your reasoning concise and focused on the key visual details observed in the cell.\n\n"""

    elif prompt_no == 6:
        system_prompt = """You are an advanced object detection model.\n\n""" +\
                        """**Task**:\n""" +\
                        """Using the full image provided, determine whether any part of the specified object is present in a grid cell.\n\n""" +\
                        """**Object to Detect**:\n""" +\
                        f"""<object>{obj}</object>\n\n""" +\
                        """**Input**:\n""" +\
                        """- You will be given the full image containing the object.\n""" +\
                        """- The image is divided into a 3x3 grid, creating 9 cells numbered from 1 to 9 (left to right, top to bottom). You will be provided one such cell.\n\n""" +\
                        """- This cell may contain all, part, or none of the object.\n\n""" +\
                        """**Instructions**:\n\n""" +\
                        """1. **Analyze the Full Image**:\n""" +\
                        """ - Begin by examining the full image to understand the object's location, size, and features.\n\n""" +\
                        """2. **Evaluate The Grid Cell**:\n""" +\
                        """ - Determine if any part of the object is present within the cell.\n""" +\
                        """ - Look for distinguishing features such as shape, color, texture, or patterns.\n""" +\
                        """ - Consider partial appearances and overlapping regions.\n""" +\
                        """ - Decide whether to label the cell as containing the object ("yes") or not ("no").\n\n""" +\
                        """**Output Format**:\n\n""" +\
                        """Present your findings in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n""" +\
                        """}\n\n""" +\
                        """**Guidelines**:\n\n""" +\
                        """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n""" +\
                        """- **Clarity**: Keep your analysis concise and focused on the most significant features.\n""" +\
                        """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n""" +\
                        """- **Consistency**: Ensure your analysis aligns with your final decision for the cell.\n\n"""

    elif prompt_no == 7:
        system_prompt = """You are an advanced object detection model. Your task is to detect the presence of a specified object within a grid cell.\n\n""" +\
                        """**Target Object**:\n""" +\
                        f"""<object>{obj}</object>\n""" +\
                        """\n""" +\
                        """**Cell Information**:\n""" +\
                        """- You will analyze a grid cell representing a section of a larger image.\n""" +\
                        """- The cell may contain all, part, or none of the target object.\n\n""" +\
                        """**Your Objectives**:\n""" +\
                        """1. **Analyze The Cell**:\n""" +\
                        """   - Examine the visual content of the grid cell carefully.\n""" +\
                        """   - Look for any features, patterns, or fragments associated with the target object.\n\n""" +\
                        """2. **Determine Presence of the Object**:\n""" +\
                        """   - Decide whether any part of the target object is present in the cell.\n""" +\
                        """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n""" +\
                        """**Output Format**:\n""" +\
                        """Present your findings in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n""" +\
                        """}"""

    return system_prompt


# --JSON Schema----------------------------------------------------------------


def json_schema_detect(prompt_no: int, obj: str, model: str):
    if model == 'gpt-4o-2024-08-06':
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {"type": "array", "items": {"type": "string"}, "description": f"The reasoning steps leading to the final conclusion. For each grid cell, consider the location of it in the image, and use this to figure out if any part of the {obj} is present."}
            json_properties.update({str(k+1): {"type": "string", "enum": ["yes", "no"], "description": f"""Whether any part of the {obj} belongs to the grid cell indexed {k+1}. The output must either be "yes" or "no"."""} for k in range(9)})
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["reasoning_steps"] + [str(k+1) for k in range(9)], "additionalProperties": False}

        elif prompt_no in [6, 7]:
            json_schema, json_properties = {}, {}
            json_properties.update({str(k+1): {"type": "string", "enum": ["yes", "no"], "description": f"""Whether any part of the {obj} belongs to the grid cell indexed {k+1}. The output must either be "yes" or "no"."""} for k in range(9)})
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": [str(k+1) for k in range(9)], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "reasoning_steps": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description=f"The reasoning steps leading to the final conclusion. For each grid cell, consider the location of it in the image, and use this to figure out if any part of the {obj} is present."
                    ),
                    **{str(k+1): genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["yes", "no"],
                        description=f"Whether any part of the {obj} belongs to the grid cell. The output must either be 'yes' or 'no'."
                    ) for k in range(1)}
                },
                required=["reasoning_steps"] + [str(k+1) for k in range(1)],
            )

        elif prompt_no in [6, 7]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    **{str(k+1): genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["yes", "no"],
                        description=f"Whether any part of the {obj} belongs to the grid cell. The output must either be 'yes' or 'no'."
                    ) for k in range(1)}
                },
                required=[str(k+1) for k in range(1)],
            )

    elif model == 'claude-3-5-sonnet-20240620':
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema = "Please provide your response in the following JSON format:\n\n" +\
                "{\n" +\
                '    "reasoning_steps": [\n' +\
                '        "Step 1: Description of the reasoning process",\n' +\
                '        "Step 2: Further analysis",\n' +\
                '        "..."\n' +\
                "    ],\n" +\
                ''.join(f'    "{i+1}": "yes/no",\n' for i in range(9))[:-2] +\
                "\n}\n\n" +\
                "Where:\n\n" +\
                "reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process " +\
                f"leading to the final conclusion. For each grid cell, consider the location of it in the image, and use this to figure out if any part of the {obj} is present.\n\n" +\
                "1 through 9: These fields should each contain a string that is either 'yes' or 'no', " +\
                f"indicating whether any part of the {obj} belongs to the grid cell indexed by that number. " +\
                "The grid cells are numbered from 1 to 9, starting from the top-left corner and moving right and down.\n\n" +\
                "Please ensure your response adheres strictly to this JSON format, including only the specified fields " +\
                "without any additional properties."
            expected_keys = ["reasoning_steps"] + [str(k+1) for k in range(9)]

        elif prompt_no in [6, 7]:
            json_schema = "Please provide your response in the following JSON format:\n\n" +\
                "{\n" +\
                ''.join(f'    "{i+1}": "yes/no",\n' for i in range(9))[:-2] +\
                "\n}\n\n" +\
                "Where:\n\n" +\
                "1 through 9: These fields should each contain a string that is either 'yes' or 'no', " +\
                f"indicating whether any part of the {obj} belongs to the grid cell indexed by that number. " +\
                "The grid cells are numbered from 1 to 9, starting from the top-left corner and moving right and down.\n\n" +\
                "Please ensure your response adheres strictly to this JSON format, including only the specified fields " +\
                "without any additional properties."
            expected_keys = [str(k+1) for k in range(9)]

        json_schema = (json_schema, expected_keys)

    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_detect(prompt_no: int, obj: str, model: str) -> str:
    messages = []
    rows, cols = 3, 3

    system_prompt = system_prompts_od(prompt_no, obj)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_0 = """Here is the full image for context."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_0}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    for i in range(rows):
        for j in range(cols):
            user_prompt_local = f"""Here is grid cell index {cols * i + j + 1}. Is any part of the {obj} present in this cell? """
            messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_local}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    json_schema = json_schema_detect(prompt_no, obj, model)

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_detect_independent(prompt_no: int, obj: str, model: str, location: str) -> str:
    messages = []

    system_prompt = system_prompts_od(prompt_no, obj)
    messages.append({"role": "system", "content": system_prompt})

    context_prompt = f"""Here is the full image for context. We will be focusing on the {location} section."""
    messages.append({"role": "user", "content": [{"type": "text", "text": context_prompt}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    user_prompt_local = f"""Here is a grid cell taken from the {location} of the full image. Is any part of the {obj} present in this cell?"""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_local}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    json_schema = json_schema_detect(prompt_no, obj, model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


def find_coords(np_img: np.ndarray, img_width, img_height, resolution: int) -> List[int]:
    coords = [-1, -1, -1, -1]  # left, top, right, bottom

    if resolution == 1:
        marks = [0, 1/4, 3/4, 1]
    elif resolution == 2:
        marks = [0, 1/8, 7/8, 1]
    else:
        raise ValueError("Unsupported resolution")

    # Calculate x and y positions based on marks
    x_positions = [int(mark * img_width) for mark in marks]
    y_positions = [int(mark * img_height) for mark in marks]

    left_indices = np.where(np.sum(np_img, axis=0) > 0)[0]
    if left_indices.size > 0:
        coords[0] = x_positions[left_indices[0]]

    right_indices = np.where(np.sum(np_img, axis=0) > 0)[0]
    if right_indices.size > 0:
        coords[2] = x_positions[right_indices[-1] + 1]

    top_indices = np.where(np.sum(np_img, axis=1) > 0)[0]
    if top_indices.size > 0:
        coords[1] = y_positions[top_indices[0]]

    bottom_indices = np.where(np.sum(np_img, axis=1) > 0)[0]
    if bottom_indices.size > 0:
        coords[3] = y_positions[bottom_indices[-1] + 1]

    return coords


def zoom(
    prompt,
    prompt_no,
    img,
    model: MFMWrapper,
    obj: str,
    global_coords,
    resolution
):
    rows, cols = 3, 3
    width, height = img.size
    np_img = np.zeros((rows, cols))

    marks = [0, 1/4, 3/4, 1] if resolution == 1 else [0, 1/8, 7/8, 1]
    x_positions = [int(width * marks[i]) for i in range(len(marks))]
    y_positions = [int(height * marks[i]) for i in range(len(marks))]
    if x_positions[0] == x_positions[1] or y_positions[0] == y_positions[1]:
        return global_coords, (0, 0), True, False

    img_list = [img]
    for i in range(rows):
        for j in range(cols):
            img_list.append(img.crop((x_positions[j], y_positions[i], x_positions[j + 1], y_positions[i + 1])))

    full_prompt = full_prompt_detect(prompt_no, obj, model.name) if not prompt else prompt
    full_prompt = replace_images_in_prompt(full_prompt, img_list)

    resp_dict, tokens, err = model.send_message(full_prompt)

    for i in range(9):
        if resp_dict[str(i+1)] == 'yes':
            np_img[i // cols, i % cols] = 1

    if np.sum(np_img) == 0:
        np_img = np.ones((rows, cols))
    coords = find_coords(np_img, width, height, resolution)
    if np.sum(np_img, axis=0)[0] > 0 and np.sum(np_img, axis=0)[-1] > 0 and np.sum(np_img, axis=1)[0] > 0 and np.sum(np_img, axis=1)[-1] > 0:
        done_zooming = True
    else:
        done_zooming = False

    prev_left, prev_top, _, _ = global_coords[0][0], global_coords[0][1], global_coords[1][0], global_coords[1][1]
    new_left, new_top, new_right, new_bottom = prev_left + coords[0], prev_top + coords[1], prev_left + coords[2], prev_top + coords[3]

    return [[new_left, new_top], [new_right, new_bottom]], tokens, done_zooming, err


def independent_zoom(
    prompt,
    prompt_no,
    img,
    model: MFMWrapper,
    obj: str,
    global_coords,
    resolution
):
    total_prompt_tokens, total_compl_tokens = 0, 0
    rows, cols = 3, 3
    width, height = img.size
    np_img = np.zeros((rows, cols))

    marks = [0, 1/4, 3/4, 1] if resolution == 1 else [0, 1/8, 7/8, 1]
    x_positions = [int(width * marks[i]) for i in range(len(marks))]
    y_positions = [int(height * marks[i]) for i in range(len(marks))]
    if x_positions[0] == x_positions[1] or y_positions[0] == y_positions[1]:
        return global_coords, (0, 0), True

    locations = ['top left', 'top center', 'top right', 'center left', 'center', 'center right', 'bottom left', 'bottom center', 'bottom right']
    for i in range(rows):
        for j in range(cols):
            img_list = [img, img.crop((x_positions[j], y_positions[i], x_positions[j + 1], y_positions[i + 1]))]

            full_prompt = full_prompt_detect_independent(prompt_no, obj, model.name, locations[cols * i + j]) if not prompt else prompt
            full_prompt = replace_images_in_prompt(full_prompt, img_list)

            resp_dict, tokens, err = model.send_message(full_prompt)
            if resp_dict['1'] == 'yes':
                np_img[i, j] = 1

            total_compl_tokens += tokens[0]
            total_prompt_tokens += tokens[1]

    if np.sum(np_img) == 0:
        np_img = np.ones((rows, cols))
    coords = find_coords(np_img, width, height, resolution)
    if np.sum(np_img, axis=0)[0] > 0 and np.sum(np_img, axis=0)[-1] > 0 and np.sum(np_img, axis=1)[0] > 0 and np.sum(np_img, axis=1)[-1] > 0:
        done_zooming = True
    else:
        done_zooming = False

    prev_left, prev_top, _, _ = global_coords[0][0], global_coords[0][1], global_coords[1][0], global_coords[1][1]
    new_left, new_top, new_right, new_bottom = prev_left + coords[0], prev_top + coords[1], prev_left + coords[2], prev_top + coords[3]

    return [[new_left, new_top], [new_right, new_bottom]], (total_compl_tokens, total_prompt_tokens), done_zooming, err


@MFMWrapper.register_task('detect')
def detect(
    model: MFMWrapper,
    file_name: Union[List[str], str],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_iters: int = 7,
    object_list: Optional[Union[List[List[str]], List[str]]] = None,
    independent_crops: bool = False,
    return_dict: bool = False,
    classification_type: str = 'classify_crop'
):
    """Finds the bounding box of the listed objects in the image.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to detect objects in.
        prompt: The prompt to use for detection
        prompt_no: The prompt number to use (if prompt is None).
        object_list: The list of objects to detect. If None, finds the objects via a classification algorithm. For multiple images, provide a list of lists.
        independent_crops: Whether to process the crops independently.
        return_dict: Whether to return the result as a list of dictionaries.

    Returns:
        (if return_dict is True)
        'coords': mapping each object to its bounding box coordinates. [[x1, y1], [x2, y2]]
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        resp_list: List of images with bounding boxes around the detected objects
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    compl_tokens, prompt_tokens = 0, 0
    file_name = file_name if isinstance(file_name, list) else [file_name]

    if object_list is None:
        object_list, tokens = model.predict(classification_type, file_name, crop=False, labels=COCO_DETECT_LABELS)

    object_list = object_list if isinstance(object_list[0], list) else [object_list]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    resp_dict_list, error_status = [], False
    for img_idx, img in enumerate(imgs):
        objects = object_list[img_idx]
        coords = {"coords": {}}

        for obj_idx, obj in enumerate(objects):
            resolution, done_times = 1, 0
            img_current = deepcopy(img)
            global_coords = [[0, 0], [img.size[0], img.size[1]]]

            for i in range(n_iters):
                if independent_crops:
                    global_coords, tokens, done_zooming, err = independent_zoom(prompt, prompt_no, img_current, model, obj, global_coords, resolution)
                else:
                    global_coords, tokens, done_zooming, err = zoom(prompt, prompt_no, img_current, model, obj, global_coords, resolution)

                if err:
                    error_status = True
                    break
                compl_tokens, prompt_tokens = compl_tokens + tokens[0], prompt_tokens + tokens[1]
                if done_zooming and done_times == 0:
                    resolution, done_times = resolution + 1, done_times + 1
                elif done_zooming and done_times == 1:
                    break
                img_current = img.crop((global_coords[0][0], global_coords[0][1], global_coords[1][0], global_coords[1][1]))
            coords['coords'][obj] = global_coords

        coords['scores'] = [1.0] * len(objects)
        coords['file_name'] = file_name[img_idx]
        resp_dict_list.append(coords)

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        bbox_imgs = model.eval(resp_dict_list, eval='eval_detect', visualise=True)
        return bbox_imgs, (compl_tokens, prompt_tokens)
