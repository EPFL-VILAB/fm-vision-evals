import queue
from typing import Dict, List, Optional, Tuple, Union


import google.generativeai as genai
import numpy as np
from skimage import graph
from skimage.util import img_as_float
from skimage.segmentation import slic
from PIL import Image

from taskit.mfm import MFMWrapper
from taskit.utils.data import replace_images_in_prompt, draw_around_superpixel, save_images

# --System Prompt----------------------------------------------------------------


def system_prompts_grouping(prompt_no: int):
    assert prompt_no in [1, 2], "Invalid prompt number."

    if prompt_no == 1:
        system_prompt = """You will be shown a region of an image marked by a red boundary. This region is part of a larger object or background in the image. """ +\
                        """You'll also see several additional regions marked with blue boundaries. """ +\
                        """Your task is to determine which blue-marked regions belong to the same broad object or background category as the red-marked region. Focus on identifying the general, overarching object or background category, rather than specific details within the region. """ +\
                        """Provide your answer in JSON format, where the key is the region index and the value is either "yes" or "no":\n""" +\
                        """"yes" if the region is part of the same object or background category as the initial region.\n""" +\
                        """"no" if it is not.\n""" +\
                        """Also, include a brief explanation of your reasoning. Example: {"reasoning_steps": [the reasoning steps leading to the final conclusion], "1": "yes", "2": "no", "3": "yes", "4": "no"} (if there are 4 blue regions)\n""" +\
                        """Important guidelines:\n""" +\
                        """1. Always identify the broader context: If a region shows a part of an object (e.g., an animal's fur, a car's wheel), you MUST identify it as the whole object (e.g., the animal, the car), not just that specific part.""" +\
                        """2. Multiple instances: If regions belong to different instances of the same type of object (e.g., two different chairs), mark them as "no" even if they're the same type of object."""

    elif prompt_no == 2:
        system_prompt = """You will be shown a region of an image marked by a red boundary. This region is part of a larger object or background in the image. You'll also see several additional regions marked with blue boundaries.\n\n""" +\
                        """Your task is to determine which blue-marked regions belong to the same overall object or background category as the red-marked region. Focus on identifying the complete object or background, rather than specific parts or features of the region.\n\n""" +\
                        """### Important Guidelines:\n""" +\
                        """1. **Holistic Identification**: Always consider the broader context of the red-marked region. For example, if the red-marked region shows part of an animal's nose, you should identify the entire animal as the object, not just the nose. Similarly, if it shows part of a vehicle, such as a wheel, the object is the entire vehicle. Avoid focusing on small details or fragments; think in terms of the whole object or background.\n""" +\
                        """2. **Apostrophe-S Test**: To ensure holistic identification, apply the "apostrophe-s" test in your reasoning. For example, if the region shows "the animal's nose," the full object is the "animal." If the red-marked region is "the vehicle's wheel," the full object is the "vehicle." Use this test to think about whether the part belongs to a larger, more complete entity.\n""" +\
                        """3. **General Object or Background Category**: Focus on the broad category (e.g., "animal," "vehicle," "natural environment"). If the blue region belongs to the same object category as the red-marked region, mark it as "yes." For instance, if the red-marked region shows part of an animal, and a blue region shows another part of the same animal, mark it as "yes." If a blue region shows another animal, mark it as "no," even if it's the same type of animal.\n""" +\
                        """4. **Instance Consistency**: If regions belong to different instances of the same object type (e.g., two separate objects of the same kind), mark them as "no." Each region should be part of the exact same object or background as the red-marked region.\n""" +\
                        """5. **Clear Boundaries**: Ensure that you consider the complete boundaries of the object or background when matching the regions. The entire region must fit within the same object category.\n\n""" +\
                        """### Reasoning Steps and Output Format:\n""" +\
                        """As you determine whether each blue-marked region belongs to the same object or background, document your thought process step by step. Specifically, explain whether the region is part of a larger entity and how the apostrophe-s test influences your decision. Provide your answer in JSON format, where the key is the region index and the value is either "yes" or "no":\n""" +\
                        """- "yes" if the region is part of the same object or background category as the red-marked region.\n""" +\
                        """- "no" if it is not.\n\n""" +\
                        """Also, include a brief explanation of your reasoning. Example: {"reasoning_steps": [the reasoning steps leading to the final conclusion, including how the apostrophe-s test was used], "1": "yes", "2": "no", "3": "yes", "4": "no"} (if there are 4 blue regions).\n"""

    return system_prompt


def system_prompts_grouping_sans_context(prompt_no: int):
    assert prompt_no in [1, 2], "Invalid prompt number."

    if prompt_no == 1:
        system_prompt = """You will be shown a region of an image marked by a red boundary. This region is part of a larger object or background in the image. """ +\
                        """You'll also see a region marked with a blue boundary. """ +\
                        """Your task is to determine whether the blue-marked region belongs to the same broad object or background category as the red-marked region. Focus on identifying the general, overarching object or background category, rather than specific details within the region. """ +\
                        """Provide your answer in JSON format, where the key is "1" and the value is either "yes" or "no":\n""" +\
                        """"yes" if the region is part of the same object or background category as the initial region.\n""" +\
                        """"no" if it is not.\n""" +\
                        """Also, include a brief explanation of your reasoning. Example: {"reasoning_steps": [the reasoning steps leading to the final conclusion], "1": "yes"}\n""" +\
                        """Important guidelines:\n""" +\
                        """1. Always identify the broader context: If a region shows a part of an object (e.g., an animal's fur, a car's wheel), you MUST identify it as the whole object (e.g., the animal, the car), not just that specific part.""" +\
                        """2. Multiple instances: If regions belong to different instances of the same type of object (e.g., two different chairs), mark them as "no" even if they're the same type of object."""

    elif prompt_no == 2:
        system_prompt = """You will be shown a region of an image marked by a red boundary. This region is part of a larger object or background in the image. You'll also see a region marked with a blue boundary.\n\n""" +\
                        """Your task is to determine whether the blue-marked region belongs to the same overall object or background category as the red-marked region. Focus on identifying the complete object or background, rather than specific parts or features of the region.\n\n""" +\
                        """### Important Guidelines:\n""" +\
                        """1. **Holistic Identification**: Always consider the broader context of the red-marked region. For example, if the red-marked region shows part of an animal's nose, you should identify the entire animal as the object, not just the nose. Similarly, if it shows part of a vehicle, such as a wheel, the object is the entire vehicle. Avoid focusing on small details or fragments; think in terms of the whole object or background.\n""" +\
                        """2. **Apostrophe-S Test**: To ensure holistic identification, apply the "apostrophe-s" test in your reasoning. For example, if the region shows "the animal's nose," the full object is the "animal." If the red-marked region is "the vehicle's wheel," the full object is the "vehicle." Use this test to think about whether the part belongs to a larger, more complete entity.\n""" +\
                        """3. **General Object or Background Category**: Focus on the broad category (e.g., "animal," "vehicle," "natural environment"). If the blue region belongs to the same object category as the red-marked region, mark it as "yes." For instance, if the red-marked region shows part of an animal, and a blue region shows another part of the same animal, mark it as "yes." If a blue region shows another animal, mark it as "no," even if it's the same type of animal.\n""" +\
                        """4. **Instance Consistency**: If regions belong to different instances of the same object type (e.g., two separate objects of the same kind), mark them as "no." Each region should be part of the exact same object or background as the red-marked region.\n""" +\
                        """5. **Clear Boundaries**: Ensure that you consider the complete boundaries of the object or background when matching the regions. The entire region must fit within the same object category.\n\n""" +\
                        """### Reasoning Steps and Output Format:\n""" +\
                        """As you determine whether the blue-marked region belongs to the same object or background, document your thought process step by step. Specifically, explain whether the region is part of a larger entity and how the apostrophe-s test influences your decision. Provide your answer in JSON format, where the key is "1" and the value is either "yes" or "no":\n""" +\
                        """- "yes" if the region is part of the same object or background category as the red-marked region.\n""" +\
                        """- "no" if it is not.\n\n""" +\
                        """Also, include a brief explanation of your reasoning. Example: {"reasoning_steps": [the reasoning steps leading to the final conclusion, including how the apostrophe-s test was used], "1": "yes"}\n"""

    return system_prompt
# --JSON Schema----------------------------------------------------------------


def json_schema_grouping(prompt_no: int, model: str, n_batch_segments: int):
    if model == 'gpt-4o-2024-08-06':
        if prompt_no == 1:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {"type": "array", "items": {"type": "string"}, "description": "The reasoning steps leading to the final conclusion. Avoid being specific about parts."}
            json_properties.update({str(k+1): {"type": "string",  "enum": ["yes", "no"], "description": f"Does region {k+1} belong to the same broad object or background category? Does it also belong to the same object or background instance as the region marked in red?"} for k in range(n_batch_segments)})
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["reasoning_steps"] + [str(k+1) for k in range(n_batch_segments)], "additionalProperties": False}

        elif prompt_no == 2:
            json_schema, json_properties = {}, {}
            json_properties["reasoning_steps"] = {"type": "array", "items": {"type": "string"}, "description": "The reasoning steps leading to the final conclusion. Avoid being specific about parts. Use the apostrophe-s test, as mentioned in the system prompt in your first step to identify the object."}
            json_properties.update({str(k+1): {"type": "string",  "enum": ["yes", "no"], "description": f"Does region {k+1} belong to the same broad object or background category? Does it also belong to the same object or background instance as the region marked in red?"} for k in range(n_batch_segments)})
            json_schema["name"] = "reasoning_schema"
            json_schema["strict"] = True
            json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["reasoning_steps"] + [str(k+1) for k in range(n_batch_segments)], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        if prompt_no == 1:
            json_schema = genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "reasoning_steps": genai.protos.Schema(
                            type=genai.protos.Type.ARRAY,
                            items=genai.protos.Schema(type=genai.protos.Type.STRING),
                            description="The reasoning steps leading to the final conclusion. Avoid being specific about parts."
                        ),
                        **{str(k+1): genai.protos.Schema(type=genai.protos.Type.STRING, enum=["yes", "no"], description=f"Does region {k+1} belong to the same broad object or background category? Does it also belong to the same object or background instance as the region marked in red?") for k in range(n_batch_segments)}
                    },
                    required=["reasoning_steps"] + [str(k+1) for k in range(n_batch_segments)],
                )

        elif prompt_no == 2:
            json_schema = genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "reasoning_steps": genai.protos.Schema(
                            type=genai.protos.Type.ARRAY,
                            items=genai.protos.Schema(type=genai.protos.Type.STRING),
                            description="The reasoning steps leading to the final conclusion. Avoid being specific about parts. Use the apostrophe-s test, as mentioned in the system prompt in your first step to identify the object."
                        ),
                        **{str(k+1): genai.protos.Schema(type=genai.protos.Type.STRING, enum=["yes", "no"], description=f"Does region {k+1} belong to the same broad object or background category? Does it also belong to the same object or background instance as the region marked in red?") for k in range(n_batch_segments)}
                    },
                    required=["reasoning_steps"] + [str(k+1) for k in range(n_batch_segments)],
                )

    elif model == 'claude-3-5-sonnet-20240620':
        if prompt_no == 1:
            reasoning_steps_description = """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """ +\
                                          """leading to the final conclusion. Begin by describing the full image context. Then, analyze each region, """ +\
                                          """discussing its characteristics and determining whether it belongs to the same broad object or background category as the region marked in red. """

            segment_descriptions = "\n".join([
                f'"{k+1}": This field should contain a string with either "yes" or "no" representing whether region {k+1} belongs to the same broad object or background category as the region marked in red. '
                "Additionally, provide a brief explanation of your reasoning for each region."
                for k in range(n_batch_segments)
            ])

            # Combine the prompt
            json_schema = (
                "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
                "{\n"
                '  "reasoning_steps": [list of reasoning steps],\n'
                + "\n".join([f'  "{k+1}": "yes/no",' for k in range(n_batch_segments)]) +
                "\n}\n\n"
                "Where:\n"
                f"- {reasoning_steps_description}\n"
                f"- {segment_descriptions}\n"
                "Please ensure that the output follows this format strictly, without additional fields or changes in structure. Don't forget the reasoning steps, they are important!"
            )
            expected_keys = ["reasoning_steps"] + [str(k+1) for k in range(n_batch_segments)]

        elif prompt_no == 2:
            reasoning_steps_description = """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """ +\
                                          """leading to the final conclusion. Begin by describing the full image context. Then, analyze each region, """ +\
                                          """discussing its characteristics and determining whether it belongs to the same broad object or background category as the region marked in red. """ +\
                                          """Use the apostrophe-s test, as mentioned in the system prompt, to identify the object."""

            segment_descriptions = "\n".join([
                f'"{k+1}": This field should contain a string with either "yes" or "no" representing whether region {k+1} belongs to the same broad object or background category as the region marked in red. '
                "Additionally, provide a brief explanation of your reasoning for each region."
                for k in range(n_batch_segments)
            ])

            # Combine the prompt
            json_schema = (
                "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
                "{\n"
                '  "reasoning_steps": [list of reasoning steps],\n'
                + "\n".join([f'  "{k+1}": "yes/no",' for k in range(n_batch_segments)]) +
                "\n}\n\n"
                "Where:\n"
                f"- {reasoning_steps_description}\n"
                f"- {segment_descriptions}\n"
                "Please ensure that the output follows this format strictly, without additional fields or changes in structure. Don't forget the reasoning steps, they are important!"
            )
            expected_keys = ["reasoning_steps"] + [str(k+1) for k in range(n_batch_segments)]
        json_schema = (json_schema, expected_keys)

    elif model == 'llama-3.2-90b':
        if prompt_no == 1:
            reasoning_steps_description = """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """ +\
                                          """leading to the final conclusion. Begin by describing the full image context. Then, analyze the blue region, """ +\
                                          """discussing its characteristics and determining whether it belongs to the same broad object or background category as the region marked in red. """

            segment_descriptions = "\n".join([
                f'"{k+1}": This field should contain a string with either "yes" or "no" representing whether region {k+1} belongs to the same broad object or background category as the region marked in red. '
                "Additionally, provide a brief explanation of your reasoning for each region."
                for k in range(1)
            ])

            # Combine the prompt
            json_schema = (
                "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
                "{\n"
                '  "reasoning_steps": [list of reasoning steps],\n'
                + "\n".join([f'  "{k+1}": "yes/no",' for k in range(1)]) +
                "\n}\n\n"
                "Where:\n"
                f"- {reasoning_steps_description}\n"
                f"- {segment_descriptions}\n"
                "Please ensure that the output follows this format strictly, without additional fields or changes in structure. Don't forget the reasoning steps, they are important!"
            )
            expected_keys = ["reasoning_steps"] + [str(k+1) for k in range(1)]

        elif prompt_no == 2:
            reasoning_steps_description = """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """ +\
                                          """leading to the final conclusion. Begin by describing the full image context. Then, analyze the region, """ +\
                                          """discussing its characteristics and determining whether it belongs to the same broad object or background category as the region marked in red. """ +\
                                          """Use the apostrophe-s test, as mentioned in the system prompt, to identify the object."""

            segment_descriptions = "\n".join([
                f'"{k+1}": This field should contain a string with either "yes" or "no" representing whether region {k+1} belongs to the same broad object or background category as the region marked in red. '
                "Additionally, provide a brief explanation of your reasoning for each region."
                for k in range(1)
            ])

            # Combine the prompt
            json_schema = (
                "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
                "{\n"
                '  "reasoning_steps": [list of reasoning steps],\n'
                + "\n".join([f'  "{k+1}": "yes/no",' for k in range(1)]) +
                "\n}\n\n"
                "Where:\n"
                f"- {reasoning_steps_description}\n"
                f"- {segment_descriptions}\n"
                "Please ensure that the output follows this format strictly, without additional fields or changes in structure. Don't forget the reasoning steps, they are important!"
            )
            expected_keys = ["reasoning_steps"] + [str(k+1) for k in range(1)]
        json_schema = (json_schema, expected_keys)

    return json_schema


# --Full Text Prompt----------------------------------------------------------------


def full_prompt_grouping(prompt_no: int, model: str, n_batch_segments: int, reasoning_so_far: list = []):
    messages = []

    system_prompt = system_prompts_grouping(prompt_no)
    messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": [{"type": "text", "text": "Here was the region at the first iteration."}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Here is the current zoomed in region"}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Here is a little context around the current region"}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Here is the current region in the full image."}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": f"This is the reasoning you've used so far for reference: {reasoning_so_far}."}]})

    for i in range(n_batch_segments):
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here is zoomed in region {i+1}"}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here is a little context around region {i+1}"}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Here is region {i+1} in the full image"}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    messages.append({"role": "user", "content": "Identify the regions that belong to the same object or background category as the region marked in red. Output your answer as a JSON object, where the key is the segment index and the value is either 'yes' or 'no'."})

    json_schema = json_schema_grouping(prompt_no, model, n_batch_segments)

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_grouping_sans_context(prompt_no: int, model: str, reasoning_so_far: list = []):
    messages = []
    system_prompt = system_prompts_grouping_sans_context(prompt_no)
    messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": [{"type": "text", "text": "Here is the image with the red region and the blue region."}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})
    messages.append({"role": "user", "content": "Identify whether the blue region belongs to the same object or background category as the region marked in red. Output your answer as a JSON object, where the key is the segment index and the value is either 'yes' or 'no'."})
    
    json_schema = json_schema_grouping(prompt_no, model, None)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


def segment_image(img: Image.Image, n_segments: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SLIC segmentation and build adjacency matrix."""
    segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
    rag = graph.rag_mean_color(img_as_float(img), segments)
    n_segments_actual = len(np.unique(segments))

    adjacency_matrix = np.zeros((n_segments_actual, n_segments_actual), dtype=int)
    for edge in rag.edges:
        adjacency_matrix[edge[0] - 1, edge[1] - 1] = 1
        adjacency_matrix[edge[1] - 1, edge[0] - 1] = 1

    return segments, adjacency_matrix


def prepare_images_for_prompt(
    img: Image.Image,
    segments: np.ndarray,
    current_mask: np.ndarray,
    batch_segments: List[int],
    segment_idx: int,
    shape: str
) -> List[np.ndarray]:
    """Prepare images with superpixel markers for the prompt."""
    img_list = [
        draw_around_superpixel(img, current_mask, 1, shape, 'red', 0, radius=4),
        draw_around_superpixel(img, current_mask, 1, shape, 'red', 40, radius=4),
        draw_around_superpixel(img, current_mask, 1, shape, 'red', None, radius=4),
        draw_around_superpixel(img, segments, segment_idx, shape, 'red', 0, radius=4)  # Initial image
    ]

    for segment in batch_segments:
        img_list.extend([
            draw_around_superpixel(img, segments, segment, shape, 'blue', 0, radius=4),
            draw_around_superpixel(img, segments, segment, shape, 'blue', 40, radius=4),
            draw_around_superpixel(img, segments, segment, shape, 'blue', None, radius=4)
        ])
    return img_list


def process_point(
    img: Image.Image,
    segments: np.ndarray,
    point: Tuple[int, int],
    adjacency_matrix: np.ndarray,
    n_segments: int,
    max_batch_size: int,
    shape: str,
    model: MFMWrapper,
    prompt: Optional[Dict],
    prompt_no: int,
    reasoning_array_limit: int
) -> np.ndarray:
    """Process a single point to generate a mask."""
    segment_idx = segments[point[1], point[0]]
    initial_mask = segments == segment_idx

    q = queue.Queue()
    for i in range(n_segments):
        if adjacency_matrix[segment_idx - 1, i]:
            q.put(i + 1)

    visited, part_of_object = {segment_idx}, {segment_idx}
    current_mask = initial_mask.copy()

    reasoning_so_far, error_status = [], False
    compl_tokens, prompt_tokens = 0, 0
    while not q.empty():
        batch_segments = []
        q_size = q.qsize()
        for _ in range(min(max_batch_size, q_size)):
            batch = q.get()
            if batch not in visited:
                batch_segments.append(batch)
                visited.add(batch)

        if not batch_segments:
            break

        img_list = prepare_images_for_prompt(img, segments, current_mask, batch_segments, segment_idx, shape)

        if prompt is None:
            full_prompt = full_prompt_grouping(
                prompt_no, model.name, len(batch_segments), reasoning_so_far
            )
        else:
            full_prompt = prompt
        full_prompt = replace_images_in_prompt(full_prompt, img_list)

        resp_dict, tokens, error_status = model.send_message(full_prompt)
        compl_tokens += tokens[0]
        prompt_tokens += tokens[1]
        if error_status:
            break
        if "reasoning_steps" in resp_dict:
            reasoning_so_far.append(resp_dict["reasoning_steps"])
            if len(reasoning_so_far) > reasoning_array_limit:
                reasoning_so_far.pop(0)

        for idx, segment in enumerate(batch_segments):
            response = resp_dict.get(str(idx + 1), "").lower()
            if "yes" in response:
                part_of_object.add(segment)

                for neighbor in np.where(adjacency_matrix[segment - 1] == 1)[0]:
                    if (neighbor + 1 not in visited) and (neighbor + 1 not in q.queue) and (neighbor + 1 not in batch_segments):
                        q.put(neighbor + 1)

        current_mask = np.isin(segments, list(part_of_object))

    return current_mask, (compl_tokens, prompt_tokens), error_status


def process_point_sans_context(
    img: Image.Image,
    segments: np.ndarray,
    point: Tuple[int, int],
    adjacency_matrix: np.ndarray,
    n_segments: int,
    shape: str,
    model: MFMWrapper,
    prompt: Optional[Dict],
    prompt_no: int,
) -> np.ndarray:
    """Process a single point to generate a mask."""
    segment_idx = segments[point[1], point[0]]
    initial_mask = segments == segment_idx

    q = queue.Queue()
    for i in range(n_segments):
        if adjacency_matrix[segment_idx - 1, i]:
            q.put(i + 1)

    visited, part_of_object = {segment_idx}, {segment_idx}
    current_mask = initial_mask.copy()

    reasoning_so_far, error_status = [], False
    compl_tokens, prompt_tokens = 0, 0
    while not q.empty():
        batch_segment = None
        q_size = q.qsize()
        for _ in range(min(1, q_size)):
            batch = q.get()
            if batch not in visited:
                batch_segment = batch
                visited.add(batch)

        if not batch_segment:
            break

        img_marked = draw_around_superpixel(img, current_mask, 1, shape, 'red', None, radius=4)
        img_list = [draw_around_superpixel(img_marked, segments, batch_segment, shape, 'blue', None, radius=4)]
        if prompt is None:
            full_prompt = full_prompt_grouping_sans_context(
                prompt_no, model.name, reasoning_so_far
            )
        else:
            full_prompt = prompt
        full_prompt = replace_images_in_prompt(full_prompt, img_list)

        resp_dict, tokens, error_status = model.send_message(full_prompt)
        compl_tokens += tokens[0]
        prompt_tokens += tokens[1]
        if error_status:
            resp_dict = {"1": "no"}
            # break
        if "reasoning_steps" in resp_dict:
            if len(reasoning_so_far) == 0:
                reasoning_so_far.append(resp_dict["reasoning_steps"])

            response = resp_dict.get(str(1), "").lower()
            if "yes" in response:
                part_of_object.add(batch_segment)

                for neighbor in np.where(adjacency_matrix[batch_segment - 1] == 1)[0]:
                    if (neighbor + 1 not in visited) and (neighbor + 1 not in q.queue) and (neighbor + 1 != batch_segment):
                        q.put(neighbor + 1)

        current_mask = np.isin(segments, list(part_of_object))

    return current_mask, (compl_tokens, prompt_tokens), error_status


@MFMWrapper.register_task('group')
def group(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    point_list: Union[List[List[List[int]]], List[List[int]]],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_segments: int = 400,
    max_batch_size: int = 8,
    reasoning_array_limit: int = 5,
    shape: str = "curve",
    return_dict: bool = False,
    overlay_on_same_image: bool = False,
):
    """
    Groups pixels in images based on semantic entities around given points.

    Args:
        model: The MFM model to use.
        file_name: Path(s) to the image file(s) to process. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
        point_list: Points around which to group pixels.
        prompt: Prompt for the task. Defaults to None.
        prompt_no: Prompt number. Defaults to -1.
        n_segments: Number of segments for SLIC. Defaults to 400.
        max_batch_size: Maximum number of segments in the queue. Defaults to 8.
        reasoning_array_limit: Max reasoning steps to keep in memory. Defaults to 5.
        shape: Shape of the visual marker. Defaults to "curve".
        return_dict: Return result as a list of dicts if True. Defaults to False.
        overlay_on_same_image: Overlay masks on the same image if True (if return_dict is False). Defaults to False.

    Returns:
        (if return_dict is True)
        resp_list: List of dicts, each containing the 'prediction' for each instance (predicted mask in a 2D numpy array)
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        grouped_imgs: List of images (np.ndarray) with masks overlaid
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path='temp_images')

    file_name = file_name if isinstance(file_name, list) else [file_name]
    point_list = point_list if isinstance(point_list[0][0], list) else [point_list]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    compl_tokens, prompt_tokens = 0, 0
    resp_dict_list, error_status = [], False

    for img_idx, img in enumerate(imgs):
        segments, adjacency_matrix = segment_image(img, n_segments)
        points = point_list[img_idx]
        current_masks = []

        for point in points:
            mask, tokens, err = process_point(
                img=img,
                segments=segments,
                point=point,
                adjacency_matrix=adjacency_matrix,
                n_segments=len(np.unique(segments)),
                max_batch_size=max_batch_size,
                shape=shape,
                model=model,
                prompt=prompt,
                prompt_no=prompt_no,
                reasoning_array_limit=reasoning_array_limit
            )
            current_masks.append(mask)
            compl_tokens += tokens[0]
            prompt_tokens += tokens[1]
            error_status = True if err else error_status

        results = {
            i: {'prediction': mask.tolist(), 'point': points[i]}
            for i, mask in enumerate(current_masks)
        }
        results['file_name'] = file_name[img_idx]
        resp_dict_list.append(results)

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        grouped_imgs = model.eval(eval='eval_group', predictions=resp_dict_list, visualise=True, overlay_on_same_image=overlay_on_same_image)
        return grouped_imgs, (compl_tokens, prompt_tokens)


@MFMWrapper.register_task('group_sans_context')
def group_sans_context(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    point_list: Union[List[List[List[int]]], List[List[int]]],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_segments: int = 400,
    shape: str = "rectangle",
    return_dict: bool = False,
    overlay_on_same_image: bool = False,
):
    """
    Groups pixels in images based on semantic entities around given points. Unlike the 'group' function, doesn't use any context around the superpixels.

    Args:
        model: The MFM model to use.
        file_name: Path(s) to the image file(s) to process. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
        point_list: Points around which to group pixels.
        prompt: Prompt for the task. Defaults to None.
        prompt_no: Prompt number. Defaults to -1.
        n_segments: Number of segments for SLIC. Defaults to 400.
        max_batch_size: Maximum number of segments in the queue. Defaults to 8.
        reasoning_array_limit: Max reasoning steps to keep in memory. Defaults to 5.
        shape: Shape of the visual marker. Defaults to "curve".
        return_dict: Return result as a list of dicts if True. Defaults to False.
        overlay_on_same_image: Overlay masks on the same image if True (if return_dict is False). Defaults to False.

    Returns:
        (if return_dict is True)
        resp_list: List of dicts, each containing the 'prediction' for each instance (predicted mask in a 2D numpy array)
        tokens: A tuple containing the completion tokens and the prompt tokens
        error_status: A boolean indicating whether an error occurred

        OR

        (if return_dict is False)
        grouped_imgs: List of images (np.ndarray) with masks overlaid
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path='temp_images')

    file_name = file_name if isinstance(file_name, list) else [file_name]
    point_list = point_list if isinstance(point_list[0][0], list) else [point_list]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    compl_tokens, prompt_tokens = 0, 0
    resp_dict_list, error_status = [], False

    for img_idx, img in enumerate(imgs):
        segments, adjacency_matrix = segment_image(img, n_segments)
        points = point_list[img_idx]
        current_masks = []

        for point in points:
            mask, tokens, err = process_point_sans_context(
                img=img,
                segments=segments,
                point=point,
                adjacency_matrix=adjacency_matrix,
                n_segments=len(np.unique(segments)),
                shape=shape,
                model=model,
                prompt=prompt,
                prompt_no=prompt_no,
            )
            current_masks.append(mask)
            compl_tokens += tokens[0]
            prompt_tokens += tokens[1]
            error_status = True if err else error_status

        results = {
            i: {'prediction': mask.tolist(), 'point': points[i]}
            for i, mask in enumerate(current_masks)
        }
        results['file_name'] = file_name[img_idx]
        resp_dict_list.append(results)

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        grouped_imgs = model.eval(eval='eval_group', predictions=resp_dict_list, visualise=True, overlay_on_same_image=overlay_on_same_image)
        return grouped_imgs, (compl_tokens, prompt_tokens)
