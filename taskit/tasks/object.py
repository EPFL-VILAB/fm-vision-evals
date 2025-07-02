from copy import deepcopy
from typing import Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from PIL import Image
from taskit.mfm import (
    TextMFMWrapper,
    OPENAI_MODELS,
    GEMINI_MODELS,
    CLAUDE_MODELS,
    TOGETHER_MODELS,
    QWEN2_MODELS,
)
from taskit.utils.data import (
    replace_images_in_prompt,
    save_images,
    draw_around_superpixel,
)
from taskit.utils.data_constants import COCO_DETECT_LABELS


# --System Prompt----------------------------------------------------------------


def system_prompts_od(
    prompt_no: int, obj: str, independent: bool = False, mark_rectangle: bool = False
) -> str:
    if prompt_no == 1 and independent:
        if mark_rectangle:
            system_prompt = (
                f"""You are an advanced object detection model. You will be provided with an image marked with a red rectangle. Your task is to determine whether any part of the {obj} is present in the rectangle.\n\n"""
                + """You need to focus your analysis only on the rectangle.\n\n"""
                + """Output your reasoning steps and your final conclusion in the following format:\n"""
                + """{\n"""
                + """    "reasoning_steps": [list the steps of your reasoning process],\n"""
                + f"""    "1": "answer" (respond with "yes" if any part of the {obj} is present in the rectangle, or "no" if it isn't)\n"""
                + """}\n\n"""
                + f"""Be thorough in your analysis, but focus solely on the presence or absence of the {obj} in the given rectangle. If you're not sure, or the rectangle is too small, output "yes"."""
            )
        else:
            system_prompt = (
                f"""You are an advanced object detection model. You will be provided with a full image for context, followed by a specific grid cell from that image. Your task is to determine whether any part of the {obj} is present in the given grid cell.\n\n"""
                + """First, you'll see the full image to understand the overall context. Then, you'll be shown a specific grid cell, which is a section of the full image. You need to focus your analysis on this grid cell.\n\n"""
                + """Output your reasoning steps and your final conclusion in the following format:\n"""
                + """{\n"""
                + """    "reasoning_steps": [list the steps of your reasoning process],\n"""
                + f"""    "1": "answer" (respond with "yes" if any part of the {obj} is present in the grid cell, or "no" if it isn't)\n"""
                + """}\n\n"""
                + f"""Be thorough in your analysis, but focus solely on the presence or absence of the {obj} in the given grid cell. If you're not sure, or the grid cell is too small, output "yes"."""
            )
    elif prompt_no == 1:
        system_prompt = (
            f""" You are an advanced object detection model. You will be provided with 9 grid cells (indexed from 1 to 9). For each grid cell, you need to output whether any part of the {obj} is present in the cell or not ("yes" if it is, "no" if it isn't).\n\n"""
            + """Output format: {{'reasoning_steps': [the reasoning steps leading to the final conclusion], "1": "answer" (yes/no), "2": "answer" (yes/no), ..., "9": "answer" (yes/no)}}"""
        )

    elif prompt_no == 2 and independent:
        if mark_rectangle:
            system_prompt = (
                """You are an advanced object detection model. You will be provided with an image marked with a red rectangle. Your task is to determine whether any part of the specified object is present in the rectangle.\n"""
                + """The object you are looking for is:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """You need to focus your analysis on the rectangle.\n\n"""
                + """Follow these steps for the rectangle:\n"""
                + """1. Carefully examine the contents of the rectangle.\n"""
                + """2. Look for any features or parts that could belong to the target object.\n"""
                + """3. Consider partial appearances of the object, not just complete views.\n"""
                + """4. Make a decision based on your analysis.\n\n"""
                + """As you analyze the rectangle, document your reasoning process. This will help explain your decisions and ensure a thorough examination of the rectangle.\n"""
                + """Present your final output in the following JSON format:\n"""
                + """{\n"""
                + """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the rectangle, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """Ensure that your reasoning steps are clear, concise, and directly related to the presence or absence of the target object in the rectangle. Be as objective as possible in your analysis, basing your decisions on visual evidence present in the rectangle.\n"""
                + """If you're not sure, or the rectangle is too small, output "yes". Your goal is to accurately detect the presence of the specified object, providing a well-reasoned analysis for your decision.\n"""
            )
        else:
            system_prompt = (
                """You are an advanced object detection model. You will be provided with a full image for context, followed by a specific grid cell from that image. Your task is to determine whether any part of the specified object is present in the given grid cell.\n"""
                + """The object you are looking for is:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """You need to focus your analysis only on the rectangle.\n\n"""
                + """Follow these steps for the rectangle:\n"""
                + """1. Carefully examine the contents of the rectangle.\n"""
                + """2. Look for any features or parts that could belong to the target object.\n"""
                + """3. Consider partial appearances of the object inside the rectangle, not just complete views.\n"""
                + """4. Make a decision based on your analysis.\n\n"""
                + """As you analyze the rectangle, document your reasoning process. This will help explain your decisions and ensure a thorough examination of the rectangle.\n"""
                + """Present your final output in the following JSON format:\n"""
                + """{\n"""
                + """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the rectangle, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """Ensure that your reasoning steps are clear, concise, and directly related to the presence or absence of the target object in the rectangle. Be as objective as possible in your analysis, basing your decisions on visual evidence present in the rectangle.\n"""
                + """If you're not sure, or the rectangle is too small, output "yes". Your goal is to accurately detect the presence of the specified object, providing a well-reasoned analysis for your decision.\n"""
            )
    elif prompt_no == 2:
        system_prompt = (
            """You are an advanced object detection model tasked with identifying the presence of a specific object in a set of grid cells. Your goal is to analyze each cell and determine whether any part of the target object is present.\n"""
            + """The object you are looking for is:\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """You will be provided with 9 grid cells, indexed from 1 to 9. These cells represent a 3x3 grid of image crops from a larger image. Your task is to analyze each grid cell and determine if any part of the specified object is present in that cell. For each cell, you should output "yes" if the object is present, or "no" if it isn't.\n"""
            + """Follow these steps for each grid cell:\n"""
            + """1. Carefully examine the contents of the cell.\n"""
            + """2. Look for any features or parts that could belong to the target object.\n"""
            + """3. Consider partial appearances of the object, not just complete views.\n"""
            + """4. Make a decision based on your analysis.\n\n"""
            + """As you analyze each cell, document your reasoning process. This will help explain your decisions and ensure thorough examination of each cell.\n"""
            + """Present your final output in the following JSON format:\n"""
            + """{\n"""
            + """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n"""
            + """    "1": "answer" (yes/no),\n"""
            + """    "2": "answer" (yes/no),\n"""
            + """    ..., \n"""
            + """    "9": "answer" (yes/no)\n"""
            + """}\n"""
            + """Ensure that your reasoning steps are clear, concise, and directly related to the presence or absence of the target object in each cell. Be as objective as possible in your analysis, basing your decisions on visual evidence present in the grid cells.\n"""
            + """Remember, your goal is to accurately detect the presence of the specified object in each of the 9 grid cells, providing a well-reasoned analysis for each decision.\n"""
        )

    elif prompt_no == 3 and independent:
        if mark_rectangle:
            system_prompt = (
                """You are an advanced object detection model with exceptional analytical capabilities. Your task is to detect the presence of a specified object within the rectangle marked in red.\n\n"""
                + """**Target Object**:\n"""
                + f"""<object>{obj}</object>\n"""
                + """\n"""
                + """**Grid Information**:\n"""
                + """- You will analyze the rectangle containing a section of a larger image.\n"""
                + """- The rectangle may contain all, part, or none of the target object.\n\n"""
                + """**Your Objectives**:\n"""
                + """1. **Analyze The Rectangle**:\n"""
                + """   - Examine the visual content inside the rectangle carefully.\n"""
                + """   - Look for any features, patterns, or fragments associated with the target object.\n"""
                + """   - Consider partial appearances, occlusions, rotations, scaling, and variations in lighting or perspective.\n\n"""
                + """2. **Determine Presence of the Object**:\n"""
                + """   - Decide whether any part of the target object is present in the rectangle.\n"""
                + """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n"""
                + """3. **Document Your Reasoning**:\n"""
                + """   - Provide clear and concise reasoning for each decision.\n"""
                + """   - Your reasoning should focus on the key visual evidence that supports your conclusion.\n\n"""
                + """**Output Format**:\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [\n"""
                + """        "Your reasoning for the rectangle",\n"""
                + """    ],\n"""
                + """    "1": "yes" or "no",\n"""
                + """}\n"""
                + """\n"""
                + """**Guidelines**:\n"""
                + """- **Be Objective**: Base your analysis solely on the visual content of the rectangle.\n"""
                + """- **Be Concise**: Keep your reasoning for the rectangle to a few sentences, emphasizing the most significant observations.\n"""
                + """- **Ensure Accuracy**: Double-check your conclusions to maintain high accuracy in object detection.\n"""
                + """- **Maintain Clarity**: Use clear and direct language in your reasoning.\n\n"""
                + """**Remember**:\n"""
                + """Your primary goal is to accurately detect the presence of the specified object in the rectangle and provide justifiable reasoning for your decisions. Avoid including any information not pertinent to the task."""
            )
        else:
            system_prompt = (
                """You are an advanced object detection model with exceptional analytical capabilities. Your task is to detect the presence of a specified object within a provided grid cell.\n\n"""
                + """**Target Object**:\n"""
                + f"""<object>{obj}</object>\n"""
                + """\n"""
                + """**Grid Information**:\n"""
                + """- You will analyze a grid cell representing a section of a larger image.\n"""
                + """- The cell may contain all, part, or none of the target object.\n\n"""
                + """**Your Objectives**:\n"""
                + """1. **Analyze The Cell**:\n"""
                + """   - Examine the visual content of the grid cell carefully.\n"""
                + """   - Look for any features, patterns, or fragments associated with the target object.\n"""
                + """   - Consider partial appearances, occlusions, rotations, scaling, and variations in lighting or perspective.\n\n"""
                + """2. **Determine Presence of the Object**:\n"""
                + """   - Decide whether any part of the target object is present in the cell.\n"""
                + """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n"""
                + """3. **Document Your Reasoning**:\n"""
                + """   - Provide clear and concise reasoning for each decision.\n"""
                + """   - Your reasoning should focus on the key visual evidence that supports your conclusion.\n\n"""
                + """**Output Format**:\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [\n"""
                + """        "Your reasoning for the cell",\n"""
                + """    ],\n"""
                + """    "1": "yes" or "no",\n"""
                + """}\n"""
                + """\n"""
                + """**Guidelines**:\n"""
                + """- **Be Objective**: Base your analysis solely on the visual content of the cell.\n"""
                + """- **Be Concise**: Keep your reasoning for the cell to a few sentences, emphasizing the most significant observations.\n"""
                + """- **Ensure Accuracy**: Double-check your conclusions to maintain high accuracy in object detection.\n"""
                + """- **Maintain Clarity**: Use clear and direct language in your reasoning.\n\n"""
                + """**Remember**:\n"""
                + """Your primary goal is to accurately detect the presence of the specified object in the grid cell and provide justifiable reasoning for your decisions. Avoid including any information not pertinent to the task."""
            )
    elif prompt_no == 3:
        system_prompt = (
            """You are an advanced object detection model with exceptional analytical capabilities. Your task is to detect the presence of a specified object within a set of grid cells.\n\n"""
            + """**Target Object**:\n"""
            + f"""<object>{obj}</object>\n"""
            + """\n"""
            + """**Grid Information**:\n"""
            + """- You will analyze 9 grid cells, indexed from 1 to 9, representing sections of a larger image.\n"""
            + """- Each cell may contain all, part, or none of the target object.\n\n"""
            + """**Your Objectives**:\n"""
            + """1. **Analyze Each Cell Individually**:\n"""
            + """   - Examine the visual content of each grid cell carefully.\n"""
            + """   - Look for any features, patterns, or fragments associated with the target object.\n"""
            + """   - Consider partial appearances, occlusions, rotations, scaling, and variations in lighting or perspective.\n\n"""
            + """2. **Determine Presence of the Object**:\n"""
            + """   - Decide whether any part of the target object is present in each cell.\n"""
            + """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n"""
            + """3. **Document Your Reasoning**:\n"""
            + """   - Provide clear and concise reasoning for each decision.\n"""
            + """   - Your reasoning should focus on the key visual evidence that supports your conclusion.\n\n"""
            + """**Output Format**:\n"""
            + """Present your findings in the following JSON format:\n\n"""
            + """{\n"""
            + """    "reasoning_steps": [\n"""
            + """        "Cell 1: [Your reasoning for cell 1]",\n"""
            + """        "Cell 2: [Your reasoning for cell 2]",\n"""
            + """        ...\n"""
            + """        "Cell 9: [Your reasoning for cell 9]"\n"""
            + """    ],\n"""
            + """    "1": "yes" or "no",\n"""
            + """    "2": "yes" or "no",\n"""
            + """    ...\n"""
            + """    "9": "yes" or "no"\n"""
            + """}\n"""
            + """\n"""
            + """**Guidelines**:\n"""
            + """- **Be Objective**: Base your analysis solely on the visual content of each cell.\n"""
            + """- **Be Concise**: Keep your reasoning for each cell to a few sentences, emphasizing the most significant observations.\n"""
            + """- **Ensure Accuracy**: Double-check your conclusions to maintain high accuracy in object detection.\n"""
            + """- **Maintain Clarity**: Use clear and direct language in your reasoning.\n\n"""
            + """**Example**:\n"""
            + """If cell 1 contains part of the object, your reasoning might be:\n"""
            + """\"Cell 1: Detected the distinctive curved edge and color pattern characteristic of the object.\"\n\n"""
            + """**Remember**:\n"""
            + """Your primary goal is to accurately detect the presence of the specified object in each grid cell and provide justifiable reasoning for your decisions. Avoid including any information not pertinent to the task."""
        )

    elif prompt_no == 4 and independent:
        if mark_rectangle:
            system_prompt = (
                """You are an advanced object detection model.\n\n"""
                + """**Task**:\n"""
                + """Using the full image provided, determine whether any part of the specified object is present in the rectangle marked in the image.\n\n"""
                + """**Object to Detect**:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """**Input**:\n"""
                + """- You will be given the full image containing the object.\n"""
                + """- The image will have a red rectangle marking a specific region.\n"""
                + """**Instructions**:\n\n"""
                + """1. **Analyze the Full Image**:\n"""
                + """ - Begin by examining the full image to understand the object's location, size, and features.\n\n"""
                + """2. **Evaluate The Rectangle**:\n"""
                + """ - Determine if any part of the object is present within the rectangle.\n"""
                + """ - Look for distinguishing features such as shape, color, texture, or patterns.\n"""
                + """ - Consider partial appearances and overlapping regions.\n"""
                + """ - Decide whether to label the rectangle as containing the object ("yes") or not ("no").\n\n"""
                + """3. **Document Your Reasoning**:\n"""
                + """ - Provide a brief reasoning for the rectangle, focusing on key observations that led to your decision.\n\n"""
                + """**Output Format**:\n\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the rectangle, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """**Example**:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [\n"""
                + """         "The full image shows the object ...",\n"""
                + """    ],\n"""
                + """    "1": "yes",\n"""
                + """}\n"""
                + """**Guidelines**:\n\n"""
                + """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n"""
                + """- **Clarity**: Keep your reasoning concise and focused on the most significant features.\n"""
                + """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n"""
                + """- **Consistency**: Ensure your reasoning aligns with your final decision for the rectangle.\n\n"""
            )
        else:
            system_prompt = (
                """You are an advanced object detection model.\n\n"""
                + """**Task**:\n"""
                + """Using the full image provided, determine whether any part of the specified object is present in a grid cell.\n\n"""
                + """**Object to Detect**:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """**Input**:\n"""
                + """- You will be given the full image containing the object.\n"""
                + """- The image is divided into a 3x3 grid, creating 9 cells numbered from 1 to 9 (left to right, top to bottom). You will be provided one such cell.\n\n"""
                + """- This cell may contain all, part, or none of the object.\n\n"""
                + """**Instructions**:\n\n"""
                + """1. **Analyze the Full Image**:\n"""
                + """ - Begin by examining the full image to understand the object's location, size, and features.\n\n"""
                + """2. **Evaluate The Grid Cell**:\n"""
                + """ - Determine if any part of the object is present within the cell.\n"""
                + """ - Look for distinguishing features such as shape, color, texture, or patterns.\n"""
                + """ - Consider partial appearances and overlapping regions.\n"""
                + """ - Decide whether to label the cell as containing the object ("yes") or not ("no").\n\n"""
                + """3. **Document Your Reasoning**:\n"""
                + """ - Provide a brief reasoning for the cell, focusing on key observations that led to your decision.\n\n"""
                + """**Output Format**:\n\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """**Example**:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [\n"""
                + """         "The full image shows the object ...",\n"""
                + """    ],\n"""
                + """    "1": "yes",\n"""
                + """}\n"""
                + """**Guidelines**:\n\n"""
                + """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n"""
                + """- **Clarity**: Keep your reasoning concise and focused on the most significant features.\n"""
                + """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n"""
                + """- **Consistency**: Ensure your reasoning aligns with your final decision for the cell.\n\n"""
            )
    elif prompt_no == 4:
        system_prompt = (
            """You are an advanced object detection model.\n\n"""
            + """**Task**:\n"""
            + """Using the full image provided, determine whether any part of the specified object is present in each of 9 grid cells.\n\n"""
            + """**Object to Detect**:\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """**Input**:\n"""
            + """- You will be given the full image containing the object.\n"""
            + """- The image is divided into a 3x3 grid, creating 9 cells numbered from 1 to 9 (left to right, top to bottom). You will be provided with 9 crops, each representing one of these cells.\n\n"""
            + """- Each cell may contain all, part, or none of the object.\n\n"""
            + """**Instructions**:\n\n"""
            + """1. **Analyze the Full Image**:\n"""
            + """ - Begin by examining the full image to understand the object's location, size, and features.\n\n"""
            + """2. **Evaluate Each Grid Cell**:\n"""
            + """ - For each cell (1 to 9):\n"""
            + """ - Determine if any part of the object is present within that cell.\n"""
            + """ - Look for distinguishing features such as shape, color, texture, or patterns.\n"""
            + """ - Consider partial appearances and overlapping regions.\n"""
            + """ - Decide whether to label the cell as containing the object ("yes") or not ("no").\n\n"""
            + """3. **Document Your Reasoning**:\n"""
            + """ - Provide a brief reasoning for each cell, focusing on key observations that led to your decision.\n\n"""
            + """**Output Format**:\n\n"""
            + """Present your findings in the following JSON format:\n\n"""
            + """{\n"""
            + """    "reasoning_steps": [\n"""
            + """        "analysis of full image",\n"""
            + """        "Cell 1: [Your reasoning for cell 1]",\n"""
            + """        "Cell 2: [Your reasoning for cell 2]",\n"""
            + """        "...",\n"""
            + """        "Cell 9: [Your reasoning for cell 9]"\n"""
            + """    ],\n"""
            + """    "1": "yes" or "no",\n"""
            + """    "2": "yes" or "no",\n"""
            + """    "...",\n"""
            + """    "9": "yes" or "no"\n"""
            + """}\n"""
            + """\n\n"""
            + """**Example**:\n\n"""
            + """{\n"""
            + """    "reasoning_steps": [\n"""
            + """         "The full image shows the object ...",\n"""
            + """         "Cell 1: The object's distinctive color is visible in the top-left corner.",\n"""
            + """         "Cell 2: No features of the object are present.",\n"""
            + """         "...",\n"""
            + """         "Cell 9: The edge of the object crosses into this cell."\n"""
            + """    ],\n"""
            + """    "1": "yes",\n"""
            + """    "2": "no",\n"""
            + """    "...",\n"""
            + """    "9": "yes"\n"""
            + """}\n"""
            + """**Guidelines**:\n\n"""
            + """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n"""
            + """- **Clarity**: Keep your reasoning concise and focused on the most significant features.\n"""
            + """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n"""
            + """- **Consistency**: Ensure your reasoning aligns with your final decision for each cell.\n\n"""
        )

    elif prompt_no == 5 and independent:
        if mark_rectangle:
            system_prompt = (
                """You are an advanced object detection model.\n\n"""
                + """**Task**:\n"""
                + """- Analyze the marked rectangle (in red) in the image to determine if any part of the specified object is present.\n"""
                + """- The rectangle is a section of the full image, and may contain all, part, or none of the object.\n\n"""
                + """**Object to Detect**:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """**Inputs**:\n"""
                + """- **Image**: You have the full image to understand the context and specifics of the object, with a marked rectangle.\n"""
                + """**Instructions**:\n\n"""
                + """1. **Examine the Object in the Full Image**:\n"""
                + """   - Understand the object's features: shape, color, texture, patterns, and any distinctive marks.\n\n"""
                + """2. **Analyze The Rectangle Thoroughly**:\n"""
                + """   - Look for any visual evidence of the object, even if it's a very small part or a tiny sliver.\n"""
                + """3. **Decision Criteria**:\n"""
                + """   - **Label as "yes"** if any part of the object is present in the rectangle, regardless of how small.\n"""
                + """   - **Label as "no"** if there is no visual evidence of the object in the rectangle.\n"""
                + """   - Base your decision solely on the visual content of the rectangle.\n\n"""
                + """4. **Document Your Reasoning**:\n"""
                + """   - Provide a brief reasoning for the rectangle.\n"""
                + """   - Mention specific features observed that led to your decision.\n"""
                + """   - Be precise and focus on the visual evidence.\n\n"""
                + """**Output Format**:\n\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the rectangle, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """**Guidelines**:\n\n"""
                + """- **Detect Even Small Parts**: If any part of the object is present, no matter how small, label the rectangle as "yes".\n"""
                + """- **Precision**: Do not assume the object's presence without clear visual confirmation.\n"""
                + """- **Clarity**: Keep your reasoning concise and focused on the key visual details observed in the rectangle.\n\n"""
            )
        else:
            system_prompt = (
                """You are an advanced object detection model.\n\n"""
                + """**Task**:\n"""
                + """- Analyze the provided grid cell from the image to determine if any part of the specified object is present.\n"""
                + """- The cell is a section of the full image, and may contain all, part, or none of the object.\n\n"""
                + """**Object to Detect**:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """**Inputs**:\n"""
                + """- **Full Image**: You have the full image to understand the context and specifics of the object.\n"""
                + """- **Grid Cell**: You have an individual image of the grid cell extracted from the full image.\n\n"""
                + """**Instructions**:\n\n"""
                + """1. **Examine the Object in the Full Image**:\n"""
                + """   - Understand the object's features: shape, color, texture, patterns, and any distinctive marks.\n\n"""
                + """2. **Analyze The Grid Cell Thoroughly**:\n"""
                + """   - Look for any visual evidence of the object, even if it's a very small part or a tiny sliver.\n"""
                + """   - Consider that the object might be partially visible due to the division of the grid.\n\n"""
                + """3. **Decision Criteria**:\n"""
                + """   - **Label as "yes"** if any part of the object is present in the cell, regardless of how small.\n"""
                + """   - **Label as "no"** if there is no visual evidence of the object in the cell.\n"""
                + """   - Base your decision solely on the visual content of the cell image.\n\n"""
                + """4. **Document Your Reasoning**:\n"""
                + """   - Provide a brief reasoning for the cell.\n"""
                + """   - Mention specific features observed that led to your decision.\n"""
                + """   - Be precise and focus on the visual evidence.\n\n"""
                + """**Output Format**:\n\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [the reasoning steps leading to the final conclusion],\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """**Guidelines**:\n\n"""
                + """- **Detect Even Small Parts**: If any part of the object is present, no matter how small, label the cell as "yes".\n"""
                + """- **Precision**: Do not assume the object's presence without clear visual confirmation, even if adjacent cells contain the object.\n"""
                + """- **Clarity**: Keep your reasoning concise and focused on the key visual details observed in the cell.\n\n"""
            )
    elif prompt_no == 5:
        system_prompt = (
            """You are an advanced object detection model.\n\n"""
            + """**Task**:\n"""
            + """- Analyze each of the 9 grid cells (numbered 1 to 9) from the provided image to determine if any part of the specified object is present.\n"""
            + """- The cells are sections of the full image, and may contain all, part, or none of the object.\n\n"""
            + """**Object to Detect**:\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """**Inputs**:\n"""
            + """- **Full Image**: You have the full image to understand the context and specifics of the object.\n"""
            + """- **Grid Cells**: You have individual images of the 9 grid cells extracted from the full image.\n\n"""
            + """**Instructions**:\n\n"""
            + """1. **Examine the Object in the Full Image**:\n"""
            + """   - Understand the object's features: shape, color, texture, patterns, and any distinctive marks.\n\n"""
            + """2. **Analyze Each Grid Cell Thoroughly**:\n"""
            + """   - For each cell (1 to 9), examine the cell image in detail.\n"""
            + """   - Look for any visual evidence of the object, even if it's a very small part or a tiny sliver.\n"""
            + """   - Consider that the object might be partially visible due to the division of the grid.\n\n"""
            + """3. **Decision Criteria**:\n"""
            + """   - **Label as "yes"** if any part of the object is present in the cell, regardless of how small.\n"""
            + """   - **Label as "no"** if there is no visual evidence of the object in the cell.\n"""
            + """   - Base your decision solely on the visual content of the cell image.\n\n"""
            + """4. **Document Your Reasoning**:\n"""
            + """   - Provide a brief reasoning for each cell.\n"""
            + """   - Mention specific features observed that led to your decision.\n"""
            + """   - Be precise and focus on the visual evidence.\n\n"""
            + """**Output Format**:\n\n"""
            + """Present your findings in the following JSON format:\n\n"""
            + """{\n"""
            + """    "reasoning_steps": [\n"""
            + """        "Cell 1: [Your reasoning for cell 1]",\n"""
            + """        "Cell 2: [Your reasoning for cell 2]",\n"""
            + """        "...",\n"""
            + """        "Cell 9: [Your reasoning for cell 9]"\n"""
            + """    ],\n"""
            + """    "1": "yes" or "no",\n"""
            + """    "2": "yes" or "no",\n"""
            + """    "...",\n"""
            + """    "9": "yes" or "no"\n"""
            + """}\n"""
            + """**Guidelines**:\n\n"""
            + """- **Detect Even Small Parts**: If any part of the object is present, no matter how small, label the cell as "yes".\n"""
            + """- **Avoid False Positives**: Ensure that you base your "yes" decisions on actual visual evidence of the object in the cell.\n"""
            + """- **Precision**: Do not assume the object's presence without clear visual confirmation, even if adjacent cells contain the object.\n"""
            + """- **Clarity**: Keep your reasoning concise and focused on the key visual details observed in the cell.\n\n"""
            + """**Example**:\n\n"""
            + """{\n"""
            + """    "reasoning_steps": [\n"""
            + """        "Cell 1: A small portion of the object's distinctive edge is visible in the top-left corner.",\n"""
            + """        "Cell 2: No identifiable features of the object are present in this cell.",\n"""
            + """        "...",\n"""
            + """        "Cell 9: The object's unique color pattern is partially visible along the cell's border."\n"""
            + """    ],\n"""
            + """    "1": "yes",\n"""
            + """    "2": "no",\n"""
            + """    "...",\n"""
            + """    "9": "yes"\n"""
            + """}\n"""
        )

    elif prompt_no == 6 and independent:
        if mark_rectangle:
            system_prompt = (
                """You are an advanced object detection model.\n\n"""
                + """**Task**:\n"""
                + """Using the full image provided, determine whether any part of the specified object is present in the marked red rectangle.\n\n"""
                + """**Object to Detect**:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """**Input**:\n"""
                + """- You will be given the full image containing the object.\n"""
                + """- The image will have a red rectangle marking a specific region.\n\n"""
                + """- The rectangle may contain all, part, or none of the object.\n\n"""
                + """**Instructions**:\n\n"""
                + """1. **Analyze the Full Image**:\n"""
                + """ - Begin by examining the full image to understand the object's location, size, and features.\n\n"""
                + """2. **Evaluate The Rectangle**:\n"""
                + """ - Determine if any part of the object is present within the rectangle.\n"""
                + """ - Look for distinguishing features such as shape, color, texture, or patterns.\n"""
                + """ - Consider partial appearances and overlapping regions.\n"""
                + """ - Decide whether to label the rectangle as containing the object ("yes") or not ("no").\n\n"""
                + """**Output Format**:\n\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the rectangle, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """**Guidelines**:\n\n"""
                + """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n"""
                + """- **Clarity**: Keep your analysis concise and focused on the most significant features.\n"""
                + """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n"""
                + """- **Consistency**: Ensure your analysis aligns with your final decision for the rectangle.\n\n"""
            )
        else:
            system_prompt = (
                """You are an advanced object detection model.\n\n"""
                + """**Task**:\n"""
                + """Using the full image provided, determine whether any part of the specified object is present in a grid cell.\n\n"""
                + """**Object to Detect**:\n"""
                + f"""<object>{obj}</object>\n\n"""
                + """**Input**:\n"""
                + """- You will be given the full image containing the object.\n"""
                + """- The image is divided into a 3x3 grid, creating 9 cells numbered from 1 to 9 (left to right, top to bottom). You will be provided one such cell.\n\n"""
                + """- This cell may contain all, part, or none of the object.\n\n"""
                + """**Instructions**:\n\n"""
                + """1. **Analyze the Full Image**:\n"""
                + """ - Begin by examining the full image to understand the object's location, size, and features.\n\n"""
                + """2. **Evaluate The Grid Cell**:\n"""
                + """ - Determine if any part of the object is present within the cell.\n"""
                + """ - Look for distinguishing features such as shape, color, texture, or patterns.\n"""
                + """ - Consider partial appearances and overlapping regions.\n"""
                + """ - Decide whether to label the cell as containing the object ("yes") or not ("no").\n\n"""
                + """**Output Format**:\n\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n"""
                + """}\n\n"""
                + """**Guidelines**:\n\n"""
                + """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n"""
                + """- **Clarity**: Keep your analysis concise and focused on the most significant features.\n"""
                + """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n"""
                + """- **Consistency**: Ensure your analysis aligns with your final decision for the cell.\n\n"""
            )
    elif prompt_no == 6:
        system_prompt = (
            """You are an advanced object detection model.\n\n"""
            + """**Task**:\n"""
            + """Using the full image provided, determine whether any part of the specified object is present in each of 9 grid cells.\n\n"""
            + """**Object to Detect**:\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """**Input**:\n"""
            + """- You will be given the full image containing the object.\n"""
            + """- The image is divided into a 3x3 grid, creating 9 cells numbered from 1 to 9 (left to right, top to bottom). You will be provided with 9 crops, each representing one of these cells.\n\n"""
            + """- Each cell may contain all, part, or none of the object.\n\n"""
            + """**Instructions**:\n\n"""
            + """1. **Analyze the Full Image**:\n"""
            + """ - Begin by examining the full image to understand the object's location, size, and features.\n\n"""
            + """2. **Evaluate Each Grid Cell**:\n"""
            + """ - For each cell (1 to 9):\n"""
            + """ - Determine if any part of the object is present within that cell.\n"""
            + """ - Look for distinguishing features such as shape, color, texture, or patterns.\n"""
            + """ - Consider partial appearances and overlapping regions.\n"""
            + """ - Decide whether to label the cell as containing the object ("yes") or not ("no").\n\n"""
            + """**Output Format**:\n\n"""
            + """Present your findings in the following JSON format:\n\n"""
            + """{\n"""
            + """    "1": "yes" or "no",\n"""
            + """    "2": "yes" or "no",\n"""
            + """    "...",\n"""
            + """    "9": "yes" or "no"\n"""
            + """}\n"""
            + """```\n\n"""
            + """**Guidelines**:\n\n"""
            + """- **Accuracy**: Base your decisions strictly on visual evidence from the image.\n"""
            + """- **Clarity**: Keep your analysis concise and focused on the most significant features.\n"""
            + """- **Objectivity**: Do not incorporate external knowledge or assumptions beyond the provided image.\n"""
            + """- **Consistency**: Ensure your analysis aligns with your final decision for each cell.\n\n"""
        )

    elif prompt_no == 7 and independent:
        if mark_rectangle:
            system_prompt = (
                """You are an advanced object detection model. Your task is to detect the presence of a specified object within the marked rectangle.\n\n"""
                + """**Target Object**:\n"""
                + f"""<object>{obj}</object>\n"""
                + """\n"""
                + """**Rectangle Information**:\n"""
                + """- You will analyze the rectangle representing a section of a larger image.\n"""
                + """- The rectangle may contain all, part, or none of the target object.\n\n"""
                + """**Your Objectives**:\n"""
                + """1. **Analyze The Rectangle**:\n"""
                + """   - Examine the visual content of the rectangle carefully.\n"""
                + """   - Look for any features, patterns, or fragments associated with the target object.\n\n"""
                + """2. **Determine Presence of the Object**:\n"""
                + """   - Decide whether any part of the target object is present in the rectangle.\n"""
                + """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n"""
                + """**Output Format**:\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the rectangle, or "no" if it isn't)\n"""
                + """}"""
            )
        else:
            system_prompt = (
                """You are an advanced object detection model. Your task is to detect the presence of a specified object within a grid cell.\n\n"""
                + """**Target Object**:\n"""
                + f"""<object>{obj}</object>\n"""
                + """\n"""
                + """**Cell Information**:\n"""
                + """- You will analyze a grid cell representing a section of a larger image.\n"""
                + """- The cell may contain all, part, or none of the target object.\n\n"""
                + """**Your Objectives**:\n"""
                + """1. **Analyze The Cell**:\n"""
                + """   - Examine the visual content of the grid cell carefully.\n"""
                + """   - Look for any features, patterns, or fragments associated with the target object.\n\n"""
                + """2. **Determine Presence of the Object**:\n"""
                + """   - Decide whether any part of the target object is present in the cell.\n"""
                + """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n"""
                + """**Output Format**:\n"""
                + """Present your findings in the following JSON format:\n\n"""
                + """{\n"""
                + """    "1": "answer" (respond with "yes" if any part of the object is present in the grid cell, or "no" if it isn't)\n"""
                + """}"""
            )
    elif prompt_no == 7:
        system_prompt = (
            """You are an advanced object detection model. Your task is to detect the presence of a specified object within a set of grid cells.\n\n"""
            + """**Target Object**:\n"""
            + f"""<object>{obj}</object>\n"""
            + """\n"""
            + """**Grid Information**:\n"""
            + """- You will analyze 9 grid cells, indexed from 1 to 9, representing sections of a larger image.\n"""
            + """- Each cell may contain all, part, or none of the target object.\n\n"""
            + """**Your Objectives**:\n"""
            + """1. **Analyze Each Cell Individually**:\n"""
            + """   - Examine the visual content of each grid cell carefully.\n"""
            + """   - Look for any features, patterns, or fragments associated with the target object.\n\n"""
            + """2. **Determine Presence of the Object**:\n"""
            + """   - Decide whether any part of the target object is present in each cell.\n"""
            + """   - Your answer should be "yes" if any portion of the object is detected, or "no" if the object is absent.\n\n"""
            + """**Output Format**:\n"""
            + """Present your findings in the following JSON format:\n\n"""
            + """{\n"""
            + """    "1": "yes" or "no",\n"""
            + """    "2": "yes" or "no",\n"""
            + """    ...\n"""
            + """    "9": "yes" or "no"\n"""
            + """}"""
        )

    return system_prompt


def system_prompts_od_naive(prompt_no: int, obj: str, normalization: int) -> str:

    if prompt_no == 1:
        system_prompt = (
            f""" Locate the {obj} and represent the location of the region. Regions are represented by [x1, y1, x2, y2] coordinates. x1 x2 """
            + f"""are the left-most and right-most points of the region, normalized into 0 to {normalization}, where 0 is the left and {normalization} is the right. y1 y2 are the """
            + f"""top-most and bottom-most points of the region, normalized into 0 to {normalization}, where 0 is the top and {normalization} is the bottom. """
            + """The output should be in JSON format, structured as follows: {"coordinates": [x1, y1, x2, y2]}"""
        )

    elif prompt_no == 2:
        system_prompt = (
            """You are an AI assistant tasked with detecting objects in images. Your goal is to locate a specific object within an image description and represent its location using normalized coordinates.\n\n"""
            + """Your task is to locate the following object in the image:\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """When representing the location of the object, use the following coordinate system:\n"""
            + """- The coordinates should be in the format [x1, y1, x2, y2]\n"""
            + """- x1 and x2 represent the left-most and right-most points of the region\n"""
            + """- y1 and y2 represent the top-most and bottom-most points of the region\n"""
            + f"""- All coordinates should be normalized to a range of 0 to {normalization}\n"""
            + f"""- For x-coordinates: 0 represents the left edge of the image, and {normalization} represents the right edge\n"""
            + f"""- For y-coordinates: 0 represents the top edge of the image, and {normalization} represents the bottom edge\n\n"""
            + """Your output should be in a JSON, structured as under:\n"""
            + """{\n"""
            + """    "coordinates": [x1, y1, x2, y2]\n"""
            + """}\n"""
        )

    elif prompt_no == 3:
        system_prompt = (
            """Your task is to precisely locate the object described below in the given image and represent its position using a normalized bounding box.\n\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """### Bounding Box Representation:\n"""
            + """The bounding box should be represented as [x1, y1, x2, y2] where:\n"""
            + """- **x1** is the left-most point of the object\n"""
            + """- **x2** is the right-most point of the object\n"""
            + """- **y1** is the top-most point of the object\n"""
            + """- **y2** is the bottom-most point of the object\n"""
            + """All coordinates must be normalized according to the following system:\n"""
            + f"""- Horizontal coordinates (x1, x2) are scaled between 0 and {normalization}, where 0 represents the left edge of the image and {normalization} represents the right edge.\n"""
            + f"""- Vertical coordinates (y1, y2) are scaled between 0 and {normalization}, where 0 represents the top edge of the image and {normalization} represents the bottom edge.\n\n"""
            + """### Steps for Accurate Detection:\n"""
            + """1. **Analyze Object Boundaries**: Start by identifying the full extent of the object, including any irregular shapes, occlusions, or edges. Ensure that the entire object is contained within the bounding box.\n"""
            + f"""2. **Normalize Coordinates**: Carefully scale the coordinates of the bounding box to the 0 to {normalization} range, ensuring consistency and precision.\n"""
            + """3. **Refine for Accuracy**: Double-check to ensure the bounding box is correctly oriented and that the object is fully contained.\n\n"""
            + """### Output Format:\n"""
            + """Once you have determined the bounding box, provide the result in the following JSON format:\n"""
            + """{\n"""
            + """    "reasoning_steps": [\n"""
            + """        "Step 1: ...",\n"""
            + """        "Step 2: ...",\n"""
            + """        "Step 3: ..."\n"""
            + """    ]\n"""
            + """    "coordinates": [x1, y1, x2, y2],\n"""
            + """}\n\n"""
            + """### Additional Considerations:\n"""
            + """- Ensure that the bounding box fully captures the object without leaving any part outside the box.\n"""
            + """- Avoid unnecessary padding around the object. Keep the bounding box as tight as possible while still enclosing the object.\n"""
            + """- If the object is partially occluded, base the bounding box on the visible portion only.\n"""
            + """- The normalized coordinates should accurately represent the object's position within the image's full dimensions."""
        )

    elif prompt_no == 4:
        system_prompt = (
            """You are tasked with identifying and locating the following object within an image:\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """### Your Objective:\n"""
            + """Locate the object in the image and create a bounding box that precisely captures its location, ensuring that the box is as tight as possible while including the entire object. If the object is occluded, focus only on the visible portion.\n\n"""
            + """### Bounding Box Instructions:\n"""
            + """- The bounding box should be represented as a list of four coordinates: [x1, y1, x2, y2].\n"""
            + """- **x1** and **x2** are the left-most and right-most points of the object, respectively.\n"""
            + """- **y1** and **y2** are the top-most and bottom-most points of the object, respectively.\n\n"""
            + f"""All coordinates must be normalized to a range of 0 to {normalization}, where:\n"""
            + f"""- 0 represents the left/top edge of the image, and {normalization} represents the right/bottom edge.\n\n"""
            + """### Methodical Approach for Bounding Box Creation:\n"""
            + """1. **Identify the Full Extent of the Object**: Carefully analyze the object's shape and boundaries. Ensure you capture the entire object within the bounding box, considering occlusions or distortions.\n"""
            + """2. **Proportional Precision**: Ensure the bounding box aligns with the object's proportions. Avoid any unnecessary padding or cropping of important object features.\n"""
            + """3. **Coordinate Normalization**: Accurately translate the object's location into the normalized coordinate system, ensuring both x and y coordinates reflect the object's precise position relative to the full image dimensions.\n"""
            + """### Expected Output Format:\n"""
            + """Return your results in the following JSON format:\n"""
            + """{\n"""
            + """    "reasoning_steps": [\n"""
            + """        "Step 1: ...",\n"""
            + """        "Step 2: ...",\n"""
            + """        "Step 3: ..."\n"""
            + """    ]\n"""
            + """    "coordinates": [x1, y1, x2, y2],\n"""
            + """}\n\n"""
            + """### Additional Considerations:\n"""
            + """- Ensure that no parts of the object are left outside the bounding box.\n"""
            + """- The bounding box should be as tight as possible without cutting off important object features.\n"""
            + """- If any part of the object is occluded, base the bounding box on what is visible and clearly define the visible boundaries.\n"""
            + """- Avoid including any unnecessary background space. Only enclose the object itself within the bounding box.\n"""
        )

    elif prompt_no == 5:
        system_prompt = (
            """You are an AI assistant tasked with detecting objects in images. Your goal is to locate a specific object within an image description and represent its location using normalized coordinates.\n\n"""
            + """Your task is to locate the following object in the image:\n"""
            + f"""<object>{obj}</object>\n\n"""
            + """When representing the location of the object, use the following coordinate system:\n"""
            + """- The coordinates should be in the format [y_min, x_min, y_max, x_max]\n"""
            + """- **y_min** and **y_max** represent the top-most and bottom-most points of the region\n"""
            + """- **x_min** and **x_max** represent the left-most and right-most points of the region\n\n"""
            + """Your output should be in a JSON, structured as follows:\n"""
            + """{\n"""
            + """    "coordinates": [y_min, x_min, y_max, x_max]\n"""
            + """}\n"""
        )

    return system_prompt


# --JSON Schema----------------------------------------------------------------


def json_schema_detect(prompt_no: int, obj: str, model: str):
    if model in OPENAI_MODELS:
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": f"The reasoning steps leading to the final conclusion. For each grid cell, consider the location of it in the image, and use this to figure out if any part of the {obj} is present.",
            }
            json_properties.update(
                {
                    str(k + 1): {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": f"""Whether any part of the {obj} belongs to the grid cell indexed {k+1}. The output must either be "yes" or "no".""",
                    }
                    for k in range(9)
                }
            )
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": ["reasoning_steps"] + [str(k + 1) for k in range(9)],
                "additionalProperties": False,
            }

        elif prompt_no in [6, 7]:
            json_schema, json_properties = {}, {}
            json_properties.update(
                {
                    str(k + 1): {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": f"""Whether any part of the {obj} belongs to the grid cell indexed {k+1}. The output must either be "yes" or "no".""",
                    }
                    for k in range(9)
                }
            )
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": [str(k + 1) for k in range(9)],
                "additionalProperties": False,
            }

    elif model in GEMINI_MODELS:
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "reasoning_steps": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description=f"The reasoning steps leading to the final conclusion. For each grid cell, consider the location of it in the image, and use this to figure out if any part of the {obj} is present.",
                    ),
                    **{
                        str(k + 1): genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            enum=["yes", "no"],
                            description=f"Whether any part of the {obj} belongs to the grid cell. The output must either be 'yes' or 'no'.",
                        )
                        for k in range(1)
                    },
                },
                required=["reasoning_steps"] + [str(k + 1) for k in range(1)],
            )

        elif prompt_no in [6, 7]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    **{
                        str(k + 1): genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            enum=["yes", "no"],
                            description=f"Whether any part of the {obj} belongs to the grid cell. The output must either be 'yes' or 'no'.",
                        )
                        for k in range(1)
                    }
                },
                required=[str(k + 1) for k in range(1)],
            )

    elif model in CLAUDE_MODELS:
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema = (
                "Please provide your response in the following JSON format:\n\n"
                + "{\n"
                + '    "reasoning_steps": [\n'
                + '        "Step 1: Description of the reasoning process",\n'
                + '        "Step 2: Further analysis",\n'
                + '        "..."\n'
                + "    ],\n"
                + "".join(f'    "{i+1}": "yes/no",\n' for i in range(9))[:-2]
                + "\n}\n\n"
                + "Where:\n\n"
                + "reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process "
                + f"leading to the final conclusion. For each grid cell, consider the location of it in the image, and use this to figure out if any part of the {obj} is present.\n\n"
                + "1 through 9: These fields should each contain a string that is either 'yes' or 'no', "
                + f"indicating whether any part of the {obj} belongs to the grid cell indexed by that number. "
                + "The grid cells are numbered from 1 to 9, starting from the top-left corner and moving right and down.\n\n"
                + "Please ensure your response adheres strictly to this JSON format, including only the specified fields "
                + "without any additional properties."
            )
            expected_keys = ["reasoning_steps"] + [str(k + 1) for k in range(9)]

        elif prompt_no in [6, 7]:
            json_schema = (
                "Please provide your response in the following JSON format:\n\n"
                + "{\n"
                + "".join(f'    "{i+1}": "yes/no",\n' for i in range(9))[:-2]
                + "\n}\n\n"
                + "Where:\n\n"
                + "1 through 9: These fields should each contain a string that is either 'yes' or 'no', "
                + f"indicating whether any part of the {obj} belongs to the grid cell indexed by that number. "
                + "The grid cells are numbered from 1 to 9, starting from the top-left corner and moving right and down.\n\n"
                + "Please ensure your response adheres strictly to this JSON format, including only the specified fields "
                + "without any additional properties."
            )
            expected_keys = [str(k + 1) for k in range(9)]

        json_schema = (json_schema, expected_keys)

    elif model in TOGETHER_MODELS:
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema = (
                "Please provide your response in the following JSON format:\n\n"
                + "{\n"
                + '    "reasoning_steps": [\n'
                + '        "Step 1: Description of the reasoning process",\n'
                + '        "Step 2: Further analysis",\n'
                + '        "..."\n'
                + "    ],\n"
                + "".join(f'    "{i+1}": "yes/no",\n' for i in range(1))[:-2]
                + "\n}\n\n"
                + "Where:\n\n"
                + "reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process "
                + f"leading to the final conclusion. For the marked rectangle, consider the location of it in the image, and use this to figure out if any part of the {obj} is present.\n\n"
                + "1: This should contain a string that is either 'yes' or 'no', "
                + f"indicating whether any part of the {obj} belongs to the grid cell. "
                + "Please ensure your response adheres strictly to this JSON format, including only the specified fields "
                + "without any additional properties."
            )
            expected_keys = ["reasoning_steps"] + [str(k + 1) for k in range(1)]

        elif prompt_no in [6, 7]:
            json_schema = (
                "Please provide your response in the following JSON format:\n\n"
                + "{\n"
                + "".join(f'    "{i+1}": "yes/no",\n' for i in range(1))[:-2]
                + "\n}\n\n"
                + "Where:\n\n"
                + "1: This should contain a string that is either 'yes' or 'no', "
                + f"indicating whether any part of the {obj} belongs to the rectangle. "
                + "Please ensure your response adheres strictly to this JSON format, including only the specified fields "
                + "without any additional properties."
            )
            expected_keys = [str(k + 1) for k in range(1)]

        json_schema = (json_schema, expected_keys)

    elif model in QWEN2_MODELS:
        if prompt_no in [1, 2, 3, 4, 5]:
            json_schema = """{"reasoning_steps": ["""
        elif prompt_no in [6, 7]:
            json_schema = """{"1": """
    else:
        raise ValueError(f"Model {model} not supported.")

    return json_schema


def json_schema_naive(prompt_no: int, obj: str, normalization: int, model: str):
    if model in OPENAI_MODELS:
        if prompt_no in [3, 4]:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": "The step-by-step reasoning process leading to the final conclusion.",
            }
            json_properties["coordinates"] = {
                "type": "array",
                "items": {"type": "number"},
                "description": f"Coordinates of the bounding box for the {obj}. [x1, y1, x2, y2] normalized into 0 to {normalization}.",
            }
            json_schema["name"] = "schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": ["reasoning_steps", "coordinates"],
                "additionalProperties": False,
            }
        elif prompt_no == 5:
            json_schema, json_properties = {}, {}
            json_properties["coordinates"] = {
                "type": "array",
                "items": {"type": "number"},
                "description": f"Coordinates of the bounding box for the {obj}. Normalized into 0 to {normalization}.",
            }
            json_schema["name"] = "schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": ["coordinates"],
                "additionalProperties": False,
            }
        else:
            json_schema, json_properties = {}, {}
            json_properties["coordinates"] = {
                "type": "array",
                "items": {"type": "number"},
                "description": f"Coordinates of the bounding box for the {obj}. [x1, y1, x2, y2] normalized into 0 to {normalization}.",
            }
            json_schema["name"] = "schema"
            json_schema["strict"] = True
            json_schema["schema"] = {
                "type": "object",
                "properties": json_properties,
                "required": ["coordinates"],
                "additionalProperties": False,
            }

    elif model in GEMINI_MODELS:
        if prompt_no in [3, 4]:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "reasoning_steps": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="The step-by-step reasoning process leading to the final conclusion.",
                    ),
                    "coordinates": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        description=f"Coordinates of the bounding box for the {obj}. [x1, y1, x2, y2] normalized into 0 to {normalization}.",
                    ),
                },
                required=["reasoning_steps", "coordinates"],
            )
        elif prompt_no == 5:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "coordinates": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.INTEGER),
                        description=f"Coordinates of the bounding box for the {obj}. Normalized into 0 to {normalization}.",
                    )
                },
                required=["coordinates"],
            )
        else:
            json_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "coordinates": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.INTEGER),
                        description=f"Coordinates of the bounding box for the {obj}.",
                    )
                },
                required=["coordinates"],
            )

    elif model in CLAUDE_MODELS:
        if prompt_no in [3, 4]:
            json_schema = (
                """Please provide your response in the following JSON format:\n\n"""
                + """{\n"""
                + """    "reasoning_steps": [\n"""
                + """        "Step 1: Description of the reasoning process",\n"""
                + """        "Step 2: Further analysis",\n"""
                + """        "..."\n"""
                + """    ],\n"""
                + """    "coordinates": [x1, y1, x2, y2]\n"""
                + """}\n\n"""
                + """Where:\n\n"""
                + """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """
                + """coordinates: This field should contain an array of four numbers representing the coordinates """
                + f"""of the bounding box for the {obj}. The format is [x1, y1, x2, y2], where (x1, y1) is the """
                + """top-left corner and (x2, y2) is the bottom-right corner of the bounding box. """
                + f"""These coordinates should be normalized to values between 0 and {normalization}.\n\n"""
                + """Please ensure your response adheres strictly to this JSON format, including only the specified field """
                + """without any additional properties. The coordinates should be numbers, not strings."""
            )
            expected_keys = ["reasoning_steps", "coordinates"]
        elif prompt_no == 5:
            json_schema = (
                """Please provide your response in the following JSON format:\n\n"""
                + """{\n"""
                + """    "coordinates": [y_min, x_min, y_max, x_max]\n"""
                + """}\n\n"""
                + """Where:\n\n"""
                + """coordinates: This field should contain an array of four numbers representing the coordinates """
                + f"""of the bounding box for the {obj}. The format is [y_min, x_min, y_max, x_max], where (y_min, x_min) is the """
                + """top-left corner and (y_max, x_max) is the bottom-right corner of the bounding box. """
                + """Note here that the y coordinates are listed first, followed by the x coordinates. """
                + f"""These coordinates should be normalized to values between 0 and {normalization}.\n\n"""
                + """Please ensure your response adheres strictly to this JSON format, including only the specified field """
                + """without any additional properties. The coordinates should be numbers, not strings."""
            )
            expected_keys = ["coordinates"]
        else:
            json_schema = (
                """Please provide your response in the following JSON format:\n\n"""
                + """{\n"""
                + """    "coordinates": [x1, y1, x2, y2]\n"""
                + """}\n\n"""
                + """Where:\n\n"""
                + """coordinates: This field should contain an array of four numbers representing the coordinates """
                + f"""of the bounding box for the {obj}. The format is [x1, y1, x2, y2], where (x1, y1) is the """
                + """top-left corner and (x2, y2) is the bottom-right corner of the bounding box. """
                + f"""These coordinates should be normalized to values between 0 and {normalization}.\n\n"""
                + """Please ensure your response adheres strictly to this JSON format, including only the specified field """
                + """without any additional properties. The coordinates should be numbers, not strings."""
            )
            expected_keys = ["coordinates"]

        json_schema = (json_schema, expected_keys)
    elif model in QWEN2_MODELS:
        if prompt_no in [3, 4]:
            json_schema = """{"reasoning_steps": ["""
        else:
            json_schema = """{"coordinates": ["""
    else:
        raise ValueError(f"Model {model} not supported.")

    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_detect(prompt_no: int, obj: str, model: str) -> str:
    messages = []
    rows, cols = 3, 3

    system_prompt = system_prompts_od(prompt_no, obj, False)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_0 = """Here is the full image for context."""
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_0},
                {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}},
            ],
        }
    )

    for i in range(rows):
        for j in range(cols):
            user_prompt_local = f"""Here is grid cell index {cols * i + j + 1}. Is any part of the {obj} present in this cell? """
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_local},
                        {
                            "type": "image_url",
                            "image_url": {"url": "<img>", "detail": "high"},
                        },
                    ],
                }
            )

    json_schema = json_schema_detect(prompt_no, obj, model)

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_detect_independent(
    prompt_no: int,
    obj: str,
    model: str,
    location: str,
    no_context: bool,
    mark_rectangle: bool,
) -> str:
    messages = []

    system_prompt = system_prompts_od(prompt_no, obj, True, mark_rectangle)
    messages.append({"role": "system", "content": system_prompt})

    if not no_context:
        context_prompt = f"""Here is the full image for context. We will be focusing on the {location} section."""
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "<img>", "detail": "high"},
                    },
                ],
            }
        )

    user_prompt_local = f"""Here is a grid cell taken from the {location} of the full image. Is any part of the {obj} present in this cell?"""
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_local},
                {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}},
            ],
        }
    )

    json_schema = json_schema_detect(prompt_no, obj, model)

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_naive(prompt_no: int, obj: str, normalization: int, model: str) -> str:
    messages = []

    system_prompt = system_prompts_od_naive(prompt_no, obj, normalization)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_0 = """Here is the full image."""
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_0},
                {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}},
            ],
        }
    )

    user_prompt_local = f"""Locate the bounding box for the {obj} in the image. The bounding box should be as tight as possible while including the entire object."""
    messages.append({"role": "user", "content": user_prompt_local})

    json_schema = json_schema_naive(prompt_no, obj, normalization, model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


def find_coords(
    np_img: np.ndarray, img_width, img_height, resolution: int
) -> List[int]:
    coords = [-1, -1, -1, -1]  # left, top, right, bottom

    if resolution == 1:
        marks = [0, 1 / 4, 3 / 4, 1]
    elif resolution == 2:
        marks = [0, 1 / 8, 7 / 8, 1]
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
    prompt, prompt_no, img, model: TextMFMWrapper, obj: str, global_coords, resolution
):
    rows, cols = 3, 3
    width, height = img.size
    np_img = np.zeros((rows, cols))

    marks = [0, 1 / 4, 3 / 4, 1] if resolution == 1 else [0, 1 / 8, 7 / 8, 1]
    x_positions = [int(width * marks[i]) for i in range(len(marks))]
    y_positions = [int(height * marks[i]) for i in range(len(marks))]
    if x_positions[0] == x_positions[1] or y_positions[0] == y_positions[1]:
        return global_coords, (0, 0), True, False

    img_list = [img]
    for i in range(rows):
        for j in range(cols):
            img_list.append(
                img.crop(
                    (
                        x_positions[j],
                        y_positions[i],
                        x_positions[j + 1],
                        y_positions[i + 1],
                    )
                )
            )

    full_prompt = (
        full_prompt_detect(prompt_no, obj, model.name) if not prompt else prompt
    )
    full_prompt = replace_images_in_prompt(full_prompt, img_list)

    resp_dict, tokens, err = model.send_message(full_prompt)

    if resp_dict:
        for i in range(9):
            if resp_dict[str(i + 1)] == "yes":
                np_img[i // cols, i % cols] = 1

    if np.sum(np_img) == 0:
        np_img = np.ones((rows, cols))
    coords = find_coords(np_img, width, height, resolution)
    if (
        np.sum(np_img, axis=0)[0] > 0
        and np.sum(np_img, axis=0)[-1] > 0
        and np.sum(np_img, axis=1)[0] > 0
        and np.sum(np_img, axis=1)[-1] > 0
    ):
        done_zooming = True
    else:
        done_zooming = False

    prev_left, prev_top, _, _ = (
        global_coords[0][0],
        global_coords[0][1],
        global_coords[1][0],
        global_coords[1][1],
    )
    new_left, new_top, new_right, new_bottom = (
        prev_left + coords[0],
        prev_top + coords[1],
        prev_left + coords[2],
        prev_top + coords[3],
    )

    return [[new_left, new_top], [new_right, new_bottom]], tokens, done_zooming, err


def independent_zoom(
    prompt,
    prompt_no,
    img,
    model: TextMFMWrapper,
    obj: str,
    global_coords,
    resolution,
    no_context,
    mark_rectangle,
):
    total_prompt_tokens, total_compl_tokens = 0, 0
    rows, cols = 3, 3
    width, height = img.size
    np_img = np.zeros((rows, cols))

    marks = [0, 1 / 4, 3 / 4, 1] if resolution == 1 else [0, 1 / 8, 7 / 8, 1]
    x_positions = [int(width * marks[i]) for i in range(len(marks))]
    y_positions = [int(height * marks[i]) for i in range(len(marks))]
    if x_positions[0] == x_positions[1] or y_positions[0] == y_positions[1]:
        return global_coords, (0, 0), True, False

    locations = [
        "top left",
        "top center",
        "top right",
        "center left",
        "center",
        "center right",
        "bottom left",
        "bottom center",
        "bottom right",
    ]
    for i in range(rows):
        for j in range(cols):
            if mark_rectangle:
                segment_array = np.zeros((height, width))
                segment_array[
                    y_positions[i]: y_positions[i + 1],
                    x_positions[j]: x_positions[j + 1],
                ] = 1
                img_list = [
                    draw_around_superpixel(
                        img, segment_array, 1, "rectangle", rectangle_width=1
                    )
                ]
            elif no_context:
                img_list = [
                    img.crop(
                        (
                            x_positions[j],
                            y_positions[i],
                            x_positions[j + 1],
                            y_positions[i + 1],
                        )
                    )
                ]
            else:
                img_list = [
                    img,
                    img.crop(
                        (
                            x_positions[j],
                            y_positions[i],
                            x_positions[j + 1],
                            y_positions[i + 1],
                        )
                    ),
                ]
            full_prompt = (
                full_prompt_detect_independent(
                    prompt_no,
                    obj,
                    model.name,
                    locations[cols * i + j],
                    no_context,
                    mark_rectangle,
                )
                if not prompt
                else prompt
            )
            full_prompt = replace_images_in_prompt(full_prompt, img_list)

            resp_dict, tokens, err = model.send_message(full_prompt)
            if not resp_dict or resp_dict["1"] == "yes":
                np_img[i, j] = 1

            total_compl_tokens += tokens[0]
            total_prompt_tokens += tokens[1]

    if np.sum(np_img) == 0:
        np_img = np.ones((rows, cols))
    coords = find_coords(np_img, width, height, resolution)
    if (
        np.sum(np_img, axis=0)[0] > 0
        and np.sum(np_img, axis=0)[-1] > 0
        and np.sum(np_img, axis=1)[0] > 0
        and np.sum(np_img, axis=1)[-1] > 0
    ):
        done_zooming = True
    else:
        done_zooming = False

    prev_left, prev_top, _, _ = (
        global_coords[0][0],
        global_coords[0][1],
        global_coords[1][0],
        global_coords[1][1],
    )
    new_left, new_top, new_right, new_bottom = (
        prev_left + coords[0],
        prev_top + coords[1],
        prev_left + coords[2],
        prev_top + coords[3],
    )

    return (
        [[new_left, new_top], [new_right, new_bottom]],
        (total_compl_tokens, total_prompt_tokens),
        done_zooming,
        err,
    )


@TextMFMWrapper.register_task("detect")
def detect(
    model: TextMFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_iters: int = 7,
    object_list: Optional[Union[List[List[str]], List[str]]] = None,
    independent: bool = False,
    return_dict: bool = False,
    classification_type: str = "classify_crop",
    no_context: bool = False,
    mark_rectangle: bool = False,
):
    """Finds the bounding box of the listed objects in the image.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to detect objects in. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
        prompt: The prompt to use for detection
        prompt_no: The prompt number to use (if prompt is None).
        object_list: The list of objects to detect. If None, finds the objects via a classification algorithm. For multiple images, provide a list of lists.
        independent: Whether to process the grid cell queries independently.
        return_dict: Whether to return the result as a list of dictionaries.
        classification_type: The type of classification to use.
        no_context: Whether to use provide the full image as context for independent mode (when crops are used).
        mark_rectangle: Whether to mark the grid cell on the image instead of cropping for the "independent" mode.

    Returns:
        (if return_dict is True)
        'coords': mapping each object to its bounding box coordinates. [[x1, y1], [x2, y2]]
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        resp_list: List of images (Image.Image) with bounding boxes around the detected objects
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path="temp_images")

    compl_tokens, prompt_tokens = 0, 0
    file_name = file_name if isinstance(file_name, list) else [file_name]

    if object_list is None:
        object_list, tokens = model.predict(
            classification_type, file_name, crop=False, labels=COCO_DETECT_LABELS
        )
        compl_tokens, prompt_tokens = (
            compl_tokens + tokens[0],
            prompt_tokens + tokens[1],
        )

    object_list = object_list if isinstance(object_list[0], list) else [object_list]
    imgs = [Image.open(fn.strip()).convert("RGB") for fn in file_name]

    resp_dict_list, error_status = [], False
    for img_idx, img in enumerate(imgs):
        objects = object_list[img_idx]
        coords = {"coords": {}}

        for obj_idx, obj in enumerate(objects):
            resolution, done_times = 1, 0
            img_current = deepcopy(img)
            global_coords = [[0, 0], [img.size[0], img.size[1]]]

            for i in range(n_iters):
                if independent:
                    global_coords, tokens, done_zooming, err = independent_zoom(
                        prompt,
                        prompt_no,
                        img_current,
                        model,
                        obj,
                        global_coords,
                        resolution,
                        no_context,
                        mark_rectangle,
                    )
                else:
                    global_coords, tokens, done_zooming, err = zoom(
                        prompt,
                        prompt_no,
                        img_current,
                        model,
                        obj,
                        global_coords,
                        resolution,
                    )

                if err:
                    error_status = True
                    break
                compl_tokens, prompt_tokens = (
                    compl_tokens + tokens[0],
                    prompt_tokens + tokens[1],
                )
                if done_zooming and done_times == 0:
                    resolution, done_times = resolution + 1, done_times + 1
                elif done_zooming and done_times == 1:
                    break
                img_current = img.crop(
                    (
                        global_coords[0][0],
                        global_coords[0][1],
                        global_coords[1][0],
                        global_coords[1][1],
                    )
                )
            coords["coords"][obj] = global_coords

        coords["scores"] = [1.0] * len(objects)
        coords["file_name"] = file_name[img_idx]
        resp_dict_list.append(coords)

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        bbox_imgs = model.eval(
            eval="eval_detect", predictions=resp_dict_list, visualise=True
        )
        return bbox_imgs, (compl_tokens, prompt_tokens)


@TextMFMWrapper.register_task("detect_naive")
def detect_naive(
    model: TextMFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    normalization: float = 1,
    object_list: Optional[Union[List[List[str]], List[str]]] = None,
    return_dict: bool = False,
    classification_type: str = "classify_crop",
):
    """Finds the bounding box of the listed objects in the image by directly regressing the bounding box coordinates.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to detect objects in. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
        prompt: The prompt to use for detection
        prompt_no: The prompt number to use (if prompt is None).
        normalization: The normalization factor for the bounding box coordinates.
        object_list: The list of objects to detect. If None, finds the objects via a classification algorithm. For multiple images, provide a list of lists.
        return_dict: Whether to return the result as a list of dictionaries.
        classification_type: The type of classification to use.

    Returns:
        (if return_dict is True)
        'coords': mapping each object to its bounding box coordinates. [[x1, y1], [x2, y2]]
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        resp_list: List of images (Image.Image) with bounding boxes around the detected objects
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path="temp_images")

    compl_tokens, prompt_tokens = 0, 0
    file_name = file_name if isinstance(file_name, list) else [file_name]

    if object_list is None:
        object_list, tokens = model.predict(
            classification_type, file_name, crop=False, labels=COCO_DETECT_LABELS
        )
        compl_tokens, prompt_tokens = (
            compl_tokens + tokens[0],
            prompt_tokens + tokens[1],
        )

    object_list = object_list if isinstance(object_list[0], list) else [object_list]
    imgs = [Image.open(fn.strip()).convert("RGB") for fn in file_name]

    resp_dict_list, error_status = [], False
    for img_idx, img in enumerate(imgs):
        objects = object_list[img_idx]
        width, height = img.size
        coords = {"coords": {}}
        for obj in objects:
            full_prompt = (
                full_prompt_naive(prompt_no, obj, normalization, model.name)
                if not prompt
                else prompt
            )
            full_prompt = replace_images_in_prompt(full_prompt, [img])

            resp_dict, tokens, err = model.send_message(full_prompt)
            if err:
                error_status = True
                break
            compl_tokens, prompt_tokens = (
                compl_tokens + tokens[0],
                prompt_tokens + tokens[1],
            )

            coords["coords"][obj] = resp_dict["coordinates"]
            if len(coords["coords"][obj]) != 4:
                coords["coords"][obj] = (
                    [0, 0, width, height] if prompt_no != 5 else [0, 0, height, width]
                )

            if prompt_no == 5:
                coords["coords"][obj] = [
                    coords["coords"][obj][1],
                    coords["coords"][obj][0],
                    coords["coords"][obj][3],
                    coords["coords"][obj][2],
                ]

            coords["coords"][obj] = [
                [
                    float(coords["coords"][obj][0] * width / normalization),
                    float(coords["coords"][obj][1] * height / normalization),
                ],
                [
                    float(coords["coords"][obj][2] * width / normalization),
                    float(coords["coords"][obj][3] * height / normalization),
                ],
            ]

        coords["scores"] = [1.0] * len(objects)
        coords["file_name"] = file_name[img_idx]

        resp_dict_list.append(coords)

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        bbox_imgs = model.eval(
            eval="eval_detect", predictions=resp_dict_list, visualise=True
        )
        return bbox_imgs, (compl_tokens, prompt_tokens)
