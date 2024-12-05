import threading
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
import numpy as np
from PIL import Image
from skimage.util import img_as_float
import supervision as sv
from skimage.segmentation import slic
from tqdm import tqdm

from taskit.eval import eval_segment
from taskit.mfm import MFMWrapper
from taskit.utils.data import replace_images_in_prompt, draw_around_superpixel, save_images
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


def system_prompts_segment_sans_context(prompt_no: int, seg_classes: list, shape: str, classes_guessed=[], description=None):
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    if prompt_no == 1:
        system_prompt = f"""You are an advanced classifier. You will be provided with an image marked by a {shape}, representing a part of an object or background that belongs to a semantic class. Your task is to determine the semantic class of the underlying object or background to which the {shape} belongs. The available semantic classes are: {seg_classes}.\n\n""" +\
                    """Your response should follow this format:\n""" +\
                    """{{"1": "class of segment"}}.\n\n""" +\
                    f"""It is important to note that you are not identifying the class the {shape} demarcates itself, but rather the class of the object or background that the {shape} is on."""

    elif prompt_no == 2 and (len(classes_guessed) > 0):
        system_prompt = f"""You are an advanced image classifier AI. Your task is to determine the semantic class of objects or backgrounds marked by the {shape} in an image. The {shape} marks a part of an object or background, and you need to identify the class to which the entire object or background belongs.\n\n""" +\
                        f"""<seg_classes>{seg_classes}</seg_classes>\nThese are the available semantic classes from which you must choose when classifying the {shape}.\n\n""" +\
                        f"""<classes_guessed>{classes_guessed}</classes_guessed>\nThese are the classes you have already guessed in previous segments. Use them if they are relevant to the current segment, otherwise, consider other classes from <seg_classes>.\n\n""" +\
                        """To complete this task effectively, follow these steps:\n\n""" +\
                        f"""1. For the {shape}, consider its location, size, and any other visual cues that might help identify the object or background it's part of.\n\n""" +\
                        f"""2. Think about which of the available semantic classes in <seg_classes> best matches the object or background represented by the {shape}.\n\n""" +\
                        """3. Use your knowledge of common objects, their parts, and typical backgrounds to make informed decisions.\n\n""" +\
                        """4. If you're unsure, consider multiple possibilities.\n\n""" +\
                        """Provide your response in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "1": "class of segment",\n""" +\
                        """}\n\n""" +\
                        """Important notes:\n""" +\
                        f"""- You are identifying the class of the entire object or background, not just the part demarcated by the {shape}.\n""" +\
                        f"""- Ensure that your classification for the {shape} is one of the classes provided in <seg_classes>.\n""" +\
                        f"""Remember, your goal is to provide accurate classifications based on the limited information given by the {shape}, using your understanding of common objects and backgrounds."""
    elif prompt_no == 2:
        system_prompt = f"""You are an advanced image classifier AI. Your task is to determine the semantic class of objects or backgrounds marked by the {shape} in an image. The {shape} marks a part of an object or background, and you need to identify the class to which the entire object or background belongs.\n\n""" +\
                        f"""<seg_classes>{seg_classes}</seg_classes>\nThese are the available semantic classes from which you must choose when classifying the {shape}.\n\n""" +\
                        """To complete this task effectively, follow these steps:\n\n""" +\
                        f"""1. For the {shape}, consider its location, size, and any other visual cues that might help identify the object or background it's part of.\n\n""" +\
                        f"""2. Think about which of the available semantic classes in <seg_classes> best matches the object or background represented by the {shape}.\n\n""" +\
                        """3. Use your knowledge of common objects, their parts, and typical backgrounds to make informed decisions.\n\n""" +\
                        """4. If you're unsure, consider multiple possibilities.\n\n""" +\
                        """Provide your response in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "1": "class of segment",\n""" +\
                        """}\n\n""" +\
                        """Important notes:\n""" +\
                        f"""- You are identifying the class of the entire object or background, not just the part demarcated by the {shape}.\n""" +\
                        f"""- Ensure that your classification for the {shape} is one of the classes provided in <seg_classes>.\n""" +\
                        f"""Remember, your goal is to provide accurate classifications based on the limited information given by the {shape}, using your understanding of common objects and backgrounds."""

    elif prompt_no == 3:
        system_prompt = f"""You are an advanced image classifier specializing in identifying semantic classes of objects or backgrounds marked by specific shapes in images. Your task is to analyze a full image with a red {shape}, and classify the semantic class of the object or background that the {shape} demarcates.\n""" +\
                        """1. First, examine the full image.\n""" +\
                        f"""2. After that, identify the semantic class of the object or background that the {shape} is marking.\n""" +\
                        """The available semantic classes are:\n""" +\
                        f"""<semantic_classes>\n{seg_classes}\n</semantic_classes>\n\n""" +\
                        """Guidelines for classification:\n""" +\
                        f"""1. Focus on the object or background that the {shape} is marking, not the {shape} itself.\n""" +\
                        """2. Consider the context provided by the full image.\n""" +\
                        """3. If you're unsure, provide your best guess.\n""" +\
                        """Present your classification in the following format:\n\n""" +\
                        """{\n""" +\
                        """  "1": "class",\n""" +\
                        """}\n\n""" +\
                        """If you cannot confidently classify a shape, provide your best estimate based on the available information."""

    elif prompt_no == 4:
        system_prompt = f"""You are an expert image classifier tasked with identifying the semantic classes of objects or backgrounds marked by specific shapes in images. Your job is to analyze a full image with a red {shape}, and classify the semantic class of the object or background that the {shape} marks.\n\n""" +\
                        """### Instructions:\n\n""" +\
                        """1. **Examine the Full Image**: Start by analyzing the entire image to understand the overall context.\n""" +\
                        f"""2. **Analyzing the {shape}**: Identify the semantic class of the object or background that the {shape} is marking.\n""" +\
                        """### Task:\n\n""" +\
                        f"""Determine the semantic class of the object or background associated with the {shape}. The possible semantic classes are listed below:\n\n""" +\
                        f"""<semantic_classes>\n{seg_classes}\n</semantic_classes>\n\n""" +\
                        """### Guidelines for Classification:\n\n""" +\
                        f"""1. Focus on the object or background marked by the {shape}, not the {shape} itself.\n""" +\
                        """2. Use the context from the full image to make your decision.\n""" +\
                        """3. Leverage the additional surrounding context to refine your classification.\n""" +\
                        """### Response Format:\n\n""" +\
                        """Present your classification in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "1": "class 1",\n""" +\
                        """}\n\n""" +\
                        """Ensure that you provide a classification for the shape. If you cannot confidently classify a shape, provide your best estimate based on the given context."""

    elif prompt_no == 5:
        system_prompt = f"""You are an expert semantic classifier. You will be given an image with a red {shape}, representing a segment of an object or background. Your task is to identify the semantic class of the underlying object or background marked by the {shape}.\n\n""" +\
                        """### Available Semantic Classes:\n""" +\
                        f"""{seg_classes}\n\n""" +\
                        """### Task Instructions:\n\n""" +\
                        f"""1. **Analyze the Image**: Determine the semantic class of the object or background represented by the {shape}, using the list of available semantic classes.\n""" +\
                        """### Response Format:\n\n""" +\
                        """Provide your response in the following structured format:\n\n""" +\
                        """{\n""" +\
                        """  "1": "class 1",\n""" +\
                        """}\n\n""" +\
                        """### Important Notes:\n\n""" +\
                        """- If uncertain about any classification, provide your best judgment.\n\n""" +\
                        f"""- Make sure the predicted classes belong in {seg_classes}\n\n""" +\
                        """Follow these guidelines to accurately classify each shape and ensure your output matches the format specified above."""

    return system_prompt


def system_prompts_describe(prompt_no: int, previous_description=""):
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    if len(previous_description) > 0:
        system_prompt = """You will be provided with the crop of an image, taken from a larger scene. Your task is to describe the contents of the image in as much detail as possible. Do not hallucinate or make up details.\n\n""" +\
                        """Here is the description of the full image from which the crop was taken:\n\n""" +\
                        f"""{previous_description}\n\n""" +\
                        """Only use this for context; do not include any information that is not visible in the crop.\n\n""" +\
                        """Provide your response in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "description": "Your description here."\n""" +\
                        """}\n\n""" +\
                        """Important notes:\n""" +\
                        """- Describe the crop in detail, focusing on the objects and backgrounds present.\n""" +\
                        """- Do not include any information that is not visible in the crop. Specifically, do not use information from the image that is not present in the crop.\n"""
    else:
        system_prompt = """You will be provided with an image. Your job is to describe the image in as much detail as possible. Do not hallucinate or make up details.\n\n""" +\
                        """Provide your response in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "description": "Your description here."\n""" +\
                        """}\n\n""" +\
                        """Important notes:\n""" +\
                        """- Describe the image in detail, focusing on the objects and backgrounds present. Make sure you describe every single object or entity present.\n""" +\
                        """- Focus on objective appearance, instead of subjective interpretation.\n""" +\
                        """- Do not include any information that is not visible in the image.\n"""
    return system_prompt


def system_prompts_naive(prompt_no: int, seg_classes: list, batch_size: int, classes_guessed=[]):
    if prompt_no == 1:
        system_prompt = f"""You are an advanced image classifier AI designed to analyze images and identify the semantic classes of objects or backgrounds marked by numbered points. In this task, you will be provided with an image containing a series of points, each labeled with a number from 0 to {batch_size - 1}. Each point marks a specific location on an object or background, and your goal is to determine the semantic class associated with each point.\n\n""" +\
                        f"""The available semantic classes are:\n<seg_classes>{seg_classes}</seg_classes>\nThese are the categories from which you must choose when classifying each point.\n\n""" +\
                        """To successfully complete this task, please follow these steps:\n\n""" +\
                        """1. Review the provided image carefully, paying attention to the location of each numbered point and its surrounding context.\n\n""" +\
                        """2. For each point, analyze the immediate area around it to determine whether it is marking an object or background, and then identify the most appropriate semantic class from the provided list.\n\n""" + \
                        """3. Utilize your understanding of typical objects, backgrounds, and their visual characteristics to make an informed classification for each point.\n\n""" + \
                        """4. If you encounter ambiguity or uncertainty in classifying a point, consider multiple possible classes and provide your reasoning for selecting the most likely one.\n\n""" + \
                        """5. Document your reasoning process clearly for each point, outlining the steps you took to arrive at your conclusion.\n\n""" + \
                        """Your response should be structured in the following JSON format:\n\n""" + \
                        """{\n""" + \
                        """  "reasoning_steps": [\n""" + \
                        """    "Step 1",\n""" + \
                        """    "Step 2",\n""" + \
                        """    ...\n""" + \
                        """  ],\n""" + \
                        """  "0": "class 0",\n""" + \
                        """  "1": "class 1",\n""" + \
                        f"""  ... "{batch_size - 1}": "class {batch_size - 1}"\n""" + \
                        """}\n\n""" + \
                        """Important reminders:\n""" + \
                        """- Your objective is to identify the semantic class of the object or background that each numbered point represents, not the point itself.\n""" + \
                        """- Ensure that the classification for each point corresponds to one of the semantic classes listed in <seg_classes>.\n""" + \
                        """- Provide detailed reasoning for each classification to justify your choices.\n""" + \
                        """- In cases of uncertainty, describe your thought process and why you selected one class over others.\n\n""" + \
                        f"""Your goal is to deliver accurate classifications for all {batch_size} points based on the visual evidence provided in the image, using your knowledge of common objects, backgrounds, and their distinguishing features."""

    elif prompt_no == 2 and (len(classes_guessed) > 0):
        system_prompt = f"""You are an advanced image classifier AI designed to analyze images and identify the semantic classes of objects or backgrounds marked by numbered points. In this task, you will be provided with an image containing a set of points, each labeled with a number from 0 to {batch_size - 1}. Each point marks a specific location on an object or background, and your goal is to determine the semantic class associated with each point.\n\n""" +\
                        f"""The available semantic classes are:\n<seg_classes>{seg_classes}</seg_classes>\nThese are the categories from which you must choose when classifying each point.\n\n""" +\
                        f"""<classes_guessed>{classes_guessed}</classes_guessed>\nThese are the classes you have already guessed in previous points. Use them if they are relevant to the current point, otherwise, consider other classes from <seg_classes>.\n\n""" +\
                        """To successfully complete this task, please follow these steps:\n\n""" +\
                        """1. Review the provided image carefully, paying attention to the location of each numbered point and its surrounding context.\n\n""" +\
                        """2. For each point, analyze the immediate area around it to determine whether it is marking an object or background, and then identify the most appropriate semantic class from the provided list.\n\n""" + \
                        """3. Utilize your understanding of typical objects, backgrounds, and their visual characteristics to make an informed classification for each point.\n\n""" + \
                        """4. If you encounter ambiguity or uncertainty in classifying a point, consider multiple possible classes and provide your reasoning for selecting the most likely one.\n\n""" + \
                        """5. Document your reasoning process clearly for each point, outlining the steps you took to arrive at your conclusion.\n\n""" + \
                        """Your response should be structured in the following JSON format:\n\n""" + \
                        """{\n""" + \
                        """  "reasoning_steps": [\n""" + \
                        """    "Step 1",\n""" + \
                        """    "Step 2",\n""" + \
                        """    ...\n""" + \
                        """  ],\n""" + \
                        """  "0": "class 0",\n""" + \
                        """  "1": "class 1",\n""" + \
                        f"""  ... "{batch_size - 1}": "class {batch_size - 1}"\n""" + \
                        """}\n\n""" + \
                        """Important reminders:\n""" + \
                        """- Your objective is to identify the semantic class of the object or background that each numbered point represents, not the point itself.\n""" + \
                        """- Ensure that the classification for each point corresponds to one of the semantic classes listed in <seg_classes>.\n""" + \
                        """- Provide detailed reasoning for each classification to justify your choices.\n""" + \
                        """- In cases of uncertainty, describe your thought process and why you selected one class over others.\n\n""" + \
                        f"""Your goal is to deliver accurate classifications for all {batch_size} points based on the visual evidence provided in the image, using your knowledge of common objects, backgrounds, and their distinguishing features."""
    elif prompt_no == 2:
        system_prompt = f"""You are an advanced image classifier AI designed to analyze images and identify the semantic classes of objects or backgrounds marked by numbered points. In this task, you will be provided with an image containing a set of points, each labeled with a number from 0 to {batch_size - 1}. Each point marks a specific location on an object or background, and your goal is to determine the semantic class associated with each point.\n\n""" +\
                        f"""The available semantic classes are:\n<seg_classes>{seg_classes}</seg_classes>\nThese are the categories from which you must choose when classifying each point.\n\n""" +\
                        """To successfully complete this task, please follow these steps:\n\n""" +\
                        """1. Review the provided image carefully, paying attention to the location of each numbered point and its surrounding context.\n\n""" +\
                        """2. For each point, analyze the immediate area around it to determine whether it is marking an object or background, and then identify the most appropriate semantic class from the provided list.\n\n""" + \
                        """3. Utilize your understanding of typical objects, backgrounds, and their visual characteristics to make an informed classification for each point.\n\n""" + \
                        """4. If you encounter ambiguity or uncertainty in classifying a point, consider multiple possible classes and provide your reasoning for selecting the most likely one.\n\n""" + \
                        """5. Document your reasoning process clearly for each point, outlining the steps you took to arrive at your conclusion.\n\n""" + \
                        """Your response should be structured in the following JSON format:\n\n""" + \
                        """{\n""" + \
                        """  "reasoning_steps": [\n""" + \
                        """    "Step 1",\n""" + \
                        """    "Step 2",\n""" + \
                        """    ...\n""" + \
                        """  ],\n""" + \
                        """  "0": "class 0",\n""" + \
                        """  "1": "class 1",\n""" + \
                        f"""  ... "{batch_size - 1}": "class {batch_size - 1}"\n""" + \
                        """}\n\n""" + \
                        """Important reminders:\n""" + \
                        """- Your objective is to identify the semantic class of the object or background that each numbered point represents, not the point itself.\n""" + \
                        """- Ensure that the classification for each point corresponds to one of the semantic classes listed in <seg_classes>.\n""" + \
                        """- Provide detailed reasoning for each classification to justify your choices.\n""" + \
                        """- In cases of uncertainty, describe your thought process and why you selected one class over others.\n\n""" + \
                        f"""Your goal is to deliver accurate classifications for all {batch_size} points based on the visual evidence provided in the image, using your knowledge of common objects, backgrounds, and their distinguishing features."""

    elif prompt_no == 3:
        system_prompt = """You are an advanced image classifier specializing in identifying semantic classes of objects or backgrounds marked by numbered points in images. Your task is to analyze the full image with the points, and classify the semantic class of the object or background that each point demarcates.\n""" +\
                        """The available semantic classes are:\n""" +\
                        f"""<semantic_classes>\n{seg_classes}\n</semantic_classes>\n\n""" +\
                        """Guidelines for classification:\n""" +\
                        """1. Focus on the object or background that the point is marking.\n""" +\
                        """2. Consider the context provided by the full image and the zoomed-in crop.\n""" +\
                        """3. Use the additional context to inform your decision.\n""" +\
                        """4. If you're unsure, explain your reasoning and provide your best guess.\n""" +\
                        """Present your analysis and classification in the following format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "0": "class 0",\n""" +\
                        """ "1": "class 1",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        f"""Ensure that you provide classifications for all {batch_size} points. If you cannot confidently classify a point, explain your reasoning and provide your best estimate based on the available information."""

    elif prompt_no == 4:
        system_prompt = """You are an expert image classifier tasked with identifying the semantic classes of objects or backgrounds marked by numbered points in images. Your job is to analyze the full image with the points, and classify the semantic class of the object or background that each point marks.\n\n""" +\
                        """### Instructions:\n\n""" +\
                        """1. **Examine the Full Image**: Start by analyzing the entire image to understand the overall context.\n""" +\
                        """2. **Analyze Numbered Points**: You will then be provided with several numbered points in the image, each marking a specific location of interest.\n""" +\
                        """3. **Consider Additional Context**: Finally, review any additional context around these numbered points to help inform your classification.\n\n""" +\
                        """### Task:\n\n""" +\
                        """Determine the semantic class of the object or background associated with each numbered point. The possible semantic classes are listed below:\n\n""" +\
                        f"""<semantic_classes>\n{seg_classes}\n</semantic_classes>\n\n""" +\
                        """### Guidelines for Classification:\n\n""" +\
                        """1. Focus on the object or background marked by each numbered point, not the point itself.\n""" +\
                        """2. Use the context from the full image and the numbered points to make your decision.\n""" +\
                        """3. Leverage the additional surrounding context to refine your classification.\n""" +\
                        """4. If uncertain, provide your reasoning and your best guess based on the information available.\n\n""" +\
                        """### Response Format:\n\n""" +\
                        """Present your analysis and classifications in the following JSON format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "0": "class 0",\n""" +\
                        """ "1": "class 1",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        f"""Ensure that you provide a classification for each of the {batch_size} points. If you cannot confidently classify a point, explain your reasoning and provide your best estimate based on the given context."""

    elif prompt_no == 5:
        system_prompt = """You are an expert semantic classifier. You will be given a set of numbered points, each marking a location on an object or background within an image. Your task is to identify the semantic class of the underlying object or background marked by each numbered point.\n\n""" +\
                        """### Available Semantic Classes:\n""" +\
                        f"""{seg_classes}\n\n""" +\
                        """### Task Instructions:\n\n""" +\
                        """1. **Analyze the Numbered Points**: For each numbered point provided, determine the semantic class of the object or background it represents, using the list of available semantic classes.\n""" +\
                        """2. **Focus on the Underlying Content**: Do not classify the point itself; instead, focus on the object or background that the point highlights.\n\n""" +\
                        """### Response Format:\n\n""" +\
                        """Provide your response in the following structured format:\n\n""" +\
                        """{\n""" +\
                        """  "reasoning_steps": [\n""" +\
                        """  "Step 1: ...",\n""" +\
                        """  "Step 2: ..."\n""" +\
                        """ ...],\n""" +\
                        """  "0": "class 0",\n""" +\
                        """ "1": "class 1",\n""" +\
                        f""" ... "{batch_size}": "class {batch_size}"\n""" +\
                        """}\n\n""" +\
                        """### Important Notes:\n\n""" +\
                        """- Make sure your reasoning steps are clear and detailed, leading logically to your final classification.\n""" +\
                        """- If uncertain about any classification, provide your best judgment along with a brief explanation of your reasoning.\n\n""" +\
                        f"""- Make sure the predicted classes belong in {seg_classes}\n\n""" +\
                        """Follow these guidelines to accurately classify each point and ensure your output matches the format specified above."""

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

    elif model == 'llama-3.2-90b':
        reasoning_steps_description = """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """ +\
                                      f"""leading to the final conclusion. Analyze the {shape} provided, """ +\
                                      """discussing its characteristics and inferring which object or background class it belongs to. Use the semantic classes provided in the system prompt for your reasoning."""

        segment_descriptions = "\n".join([
            f'"{k+1}": This field should contain a string representing the semantic class of the object or background in the full image to which the {shape} belongs. '
            "The output must be one of the classes listed in the system prompt."
            for k in range(1)
        ])

        json_schema = (
            "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
            "{\n"
            + "\n".join([f'  "{k+1}": "class name",' for k in range(1)]) +
            "\n}\n\n"
            "Where:\n"
            f"- {segment_descriptions}\n"
            "Please ensure that the output follows this format strictly, without additional fields or changes in structure."
        )

        expected_keys = [str(k+1) for k in range(1)]
        json_schema = (json_schema, expected_keys)
    else:
        json_schema = """{"reasoning_steps": ["""

    return json_schema


def json_schema_describe():
    json_schema = (
        "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
        "{\n"
        '  "description": "Your description here"\n'
        "}\n\n"
        "Where:\n"
        "- description: This field should contain a string describing the image in detail, focusing on the objects and backgrounds present. "
        "Do not include any information that is not visible in the image. Provide a clear description that captures the essence of the image."
        "Please ensure that the output follows this format strictly, without additional fields or changes in structure."
    )

    expected_keys = ["description"]
    json_schema = (json_schema, expected_keys)

    return json_schema


def json_schema_naive(batch_size: int, model: str):
    if model == 'gpt-4o-2024-08-06':
        json_schema, json_properties = {}, {}
        json_properties["reasoning_steps"] = {"type": "array", "items": {"type": "string"}, "description": "The step-by-step reasoning process leading to the final conclusion. Begin by describing the full image. Then, analyze each numbered point in the image, and infer which object or background class in the image each numbered point represents. Use the semantic classes provided in the system prompt for your reasoning."}
        json_properties.update({str(k): {"type": "string", "description": f"The semantic class of the object or background in the full image which the numbered point {k} marks. The output must be one of the classes listed in the system prompt."} for k in range(batch_size)})
        json_schema["name"] = "reasoning_schema"
        json_schema["strict"] = True
        json_schema["schema"] = {"type": "object", "properties": json_properties, "required": ["reasoning_steps"] + [str(k) for k in range(batch_size)], "additionalProperties": False}

    elif model == 'gemini-1.5-pro':
        json_schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                    "reasoning_steps": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="The step-by-step reasoning process leading to the final conclusion. Begin by describing the full image. Then, analyze each numbered point in the image, and infer which object or background class in the image each numbered point represents. Use the semantic classes provided in the system prompt for your reasoning."
                    ),
                    **{str(k): genai.protos.Schema(type=genai.protos.Type.STRING, description=f"The semantic class of the object or background in the full image which the numbered point {k} marks. The output must be one of the classes listed in the system prompt.") for k in range(batch_size)}
                },
            required=["reasoning_steps"] + [str(k) for k in range(batch_size)],
        )

    elif model == 'claude-3-5-sonnet-20240620':
        reasoning_steps_description = """reasoning_steps: This field should contain a list of strings representing the step-by-step reasoning process """ +\
                                    """leading to the final conclusion. Begin by describing the full image context. Then, analyze each numbered point in the image, """ +\
                                    """inferring which object or background class it belongs to. Use the semantic classes provided in the system prompt for your reasoning."""

        segment_descriptions = "\n".join([
            f'"{k}": This field should contain a string representing the semantic class of the object or background in the full image to which the numbered point {k} belongs. '
            "The output must be one of the classes listed in the system prompt."
            for k in range(batch_size)
        ])

        json_schema = (
            "As a reminder, your response should be formatted as a JSON object with the following structure:\n"
            "{\n"
            '  "reasoning_steps": [list of reasoning steps],\n'
            + "\n".join([f'  "{k}": "class name",' for k in range(batch_size)]) +
            "\n}\n\n"
            "Where:\n"
            f"- {reasoning_steps_description}\n"
            f"- {segment_descriptions}\n"
            "Please ensure that the output follows this format strictly, without additional fields or changes in structure."
        )

        expected_keys = ["reasoning_steps"] + [str(k) for k in range(batch_size)]
        json_schema = (json_schema, expected_keys)
    else:
        json_schema = """{"reasoning_steps": ["""

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


def full_prompt_segment_sans_context(prompt_no: int, class_labels: list, shape: str, model: str, start_idx: int, n_unique_segments: int, classes_guessed: list = [], description=None):
    messages = []

    system_prompt = system_prompts_segment_sans_context(prompt_no, class_labels, shape, classes_guessed, description)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_1 = f"""Here is the image with the {shape} for analysis."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_1}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "low"}}]})

    json_schema = json_schema_segment(shape, 1, model)

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_describe(prompt_no: int, previous_description=""):
    messages = []

    system_prompt = system_prompts_describe(prompt_no, previous_description)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_1 = """Here is the image for description."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_1}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    json_schema = json_schema_describe()

    return {"messages": messages, "json_schema": json_schema}


def full_prompt_naive(prompt_no: int, class_labels: list, batch_size: int, model: str, start_idx: int, n_unique_segments: int, classes_guessed: list = []):
    messages = []

    system_prompt = system_prompts_naive(prompt_no, class_labels, batch_size, classes_guessed)
    messages.append({"role": "system", "content": system_prompt})

    user_prompt_1 = """Here is the full image with the numbered points for analysis."""
    messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt_1}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    json_schema = json_schema_naive(batch_size, model)

    return {"messages": messages, "json_schema": json_schema}


# --Full Prompt----------------------------------------------------------------


def get_bbox_coords(segments: np.ndarray):
    """Returns an array of shape (n, 4), containing the bounding box coordinates for the segments in the format [x1, y1, x2, y2]"""
    unique_segments = np.unique(segments)
    n_formed_segments = len(unique_segments)
    bbox_coords = np.zeros((n_formed_segments, 4), dtype=int)  # (n, 4) array to hold bbox coordinates

    for i, seg_val in enumerate(unique_segments):
        # Find all positions where segment equals seg_val
        y_indices, x_indices = np.where(segments == seg_val)

        # Calculate bounding box coordinates
        x1 = np.min(x_indices)
        y1 = np.min(y_indices)
        x2 = np.max(x_indices)
        y2 = np.max(y_indices)

        # Store the bounding box coordinates in the output array
        bbox_coords[i] = np.array([x1, y1, x2, y2])

    return bbox_coords


def get_indiv_segs(segments: np.ndarray):
    """Returns an array of shape (n, H, W), each containing 1 segmentation mask"""
    unique_segments = np.unique(segments)
    H, W = segments.shape

    # Initialize an array to store individual masks
    indiv_segs = []

    for i, seg_val in enumerate(unique_segments):
        # Create a mask for the current segment
        mask = (segments == seg_val)

        # Store the mask in the corresponding slice of the output array
        indiv_segs.append(mask)

    return np.array(indiv_segs)


@MFMWrapper.register_task('segment')
def segment(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
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
        file_name: The path(s) to the image file to segment. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
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
        resp_list: List of segment images (np.ndarray) (display using plt.imshow())
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
        seg_maps = model.eval(eval='eval_segment', predictions=resp_dict_list, n_segments=n_segments, labels=labels, color_map=color_map, visualise=True)
        return seg_maps, (compl_tokens, prompt_tokens)


@MFMWrapper.register_task('segment_sans_context')
def segment_sans_context(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_segments: int = 400,
    shape: str = "point",
    labels: List[str] = COCO_SEMSEG_LABELS,
    color_map: Dict[str, list] = COCO_COLOR_MAP,
    return_dict: bool = False
) -> Union[Tuple[List[Dict], Tuple[int, int], bool], Tuple[List[np.ndarray], Tuple[int, int]]]:
    # Prepare file_name list
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path='temp_images')

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    compl_tokens, prompt_tokens = 0, 0
    error_status = False
    lock = threading.Lock()  # For thread-safe updates of tokens
    results_by_image = []

    def process_segment(img, segments, start_idx, img_idx):
        nonlocal compl_tokens, prompt_tokens
        prompt_imgs = []
        segment = draw_around_superpixel(img, segments, start_idx, shape, crop_width=None)
        prompt_imgs.append(segment)

        full_prompt = full_prompt_segment_sans_context(
            prompt_no, labels, shape, model.name, start_idx, len(np.unique(segments)), []
        ) if not prompt else prompt
        full_prompt = replace_images_in_prompt(full_prompt, prompt_imgs)

        resp_dict, tokens, local_error_status = model.send_message(full_prompt)
        if local_error_status:
            return start_idx, None, True  # Return immediately in case of error

        with lock:  # Safely update token counters
            compl_tokens += tokens[0]
            prompt_tokens += tokens[1]

        resp_dict.pop("reasoning_steps", None)
        return start_idx, resp_dict.get("1", "unknown"), False

    # Process each image separately
    for img_idx, img in enumerate(imgs):
        segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
        n_unique_segments = len(np.unique(segments))
        segment_results = []

        # Create a thread pool for processing segments in parallel
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(process_segment, img, segments, start_idx, img_idx): start_idx
                for start_idx in range(1, n_unique_segments + 1)
            }

            # Collect results as they complete, preserving segment order
            for future in tqdm(as_completed(futures)):
                start_idx = futures[future]
                try:
                    seg_idx, result, seg_error = future.result()
                    segment_results.append((seg_idx, result))
                    error_status = error_status or seg_error  # Set error_status if any segment failed
                except Exception as e:
                    print(f"An error occurred while processing segment {start_idx} of image {img_idx}: {e}")
                    error_status = True

        # Sort segments by their index to ensure order
        segment_results.sort(key=lambda x: x[0])
        all_results = {str(idx): result for idx, result in segment_results if result is not None}
        all_results["file_name"] = file_name[img_idx].strip()
        results_by_image.append(all_results)

    if return_dict:
        return results_by_image, (compl_tokens, prompt_tokens), error_status
    else:
        seg_maps = model.eval(
            eval='eval_segment', predictions=results_by_image, n_segments=n_segments,
            labels=labels, color_map=color_map, visualise=True
        )
        return seg_maps, (compl_tokens, prompt_tokens)


@MFMWrapper.register_task('segment_naive')
def segment_naive(
    model: MFMWrapper,
    file_name: Union[List[str], str, List[Image.Image], Image.Image],
    prompt: Optional[Dict] = None,
    prompt_no: int = -1,
    n_segments: int = 400,
    shape: str = "none",
    batch_size: int = 16,
    shuffle: bool = True,
    labels: List[str] = COCO_SEMSEG_LABELS,
    color_map: Dict[str, list] = COCO_COLOR_MAP,
    return_dict: bool = False
) -> Union[Tuple[List[Dict], Tuple[int, int], bool], Tuple[List[np.ndarray], Tuple[int, int]]]:
    """Segments image(s) using an MFM. Uses no context, operates in batches.

    Args:
        model: The MFM model to use.
        file_name: The path(s) to the image file to segment. Can also be a list of PIL Image objects, in which case images are saved to a temporary directory.
        prompt: The prompt to use for segmentation.
        prompt_no: The prompt number to use (if prompt is None).
        n_segments: The number of segments to split the image into (using SLIC). The actual number of segments will be close but may be different.
        batch_size: The number of segments to classify in each batch.
        shape: The visual marker. One of 'curve', 'rectangle', 'none'
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
        resp_list: List of segment images (np.ndarray) (display using plt.imshow())
        tokens: A tuple containing the completion tokens and the prompt tokens
    """
    assert shape in ["curve", "rectangle", "none"], "Invalid shape type. Choose from 'curve', 'rectangle', or 'none'."

    # Prepare file_name list
    if isinstance(file_name, Image.Image):
        file_name = [file_name]
    if isinstance(file_name, list) and isinstance(file_name[0], Image.Image):
        file_name = save_images(file_name, save_path='temp_images')

    compl_tokens, prompt_tokens = 0, 0
    resp_dict_list, error_status = [], False

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.3)
    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_position=sv.Position.CENTER,
        text_scale=0.5,
        text_color=sv.Color.WHITE,
        color=sv.Color.BLACK,
        text_thickness=1,
        text_padding=2
    )

    file_name = file_name if isinstance(file_name, list) else [file_name]
    imgs = [Image.open(fn.strip()).convert('RGB') for fn in file_name]

    for img_idx, img in enumerate(imgs):
        segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
        n_unique_segments = len(np.unique(segments))

        bboxes = get_bbox_coords(segments)
        indiv_segs = get_indiv_segs(segments)
        detections = sv.Detections(xyxy=bboxes, mask=indiv_segs)

        n_bboxes = len(bboxes)
        idxs = np.arange(n_bboxes)
        mapping = {i: j for i, j in zip(idxs, np.random.permutation(idxs))} if shuffle else {i: i for i in idxs}

        all_results, classes_guessed = {}, []
        for start_idx in tqdm(range(0, len(bboxes), batch_size)):
            indices = np.array([mapping[j] for j in range(start_idx, start_idx + batch_size) if j < len(bboxes)])
            annotated_image = img.copy()
            if shape == 'curve':
                annotated_image = mask_annotator.annotate(
                    scene=annotated_image, detections=detections[indices]
                )
            elif shape == 'rectangle':
                annotated_image = box_annotator.annotate(
                    scene=annotated_image, detections=detections[indices]
                )
            annotated_image = label_annotator.annotate(
                    scene=annotated_image, detections=detections[indices], labels=[str(j) for j in range(len(indices))]
                )

            prompt_imgs = [annotated_image]
            full_prompt = full_prompt_naive(prompt_no, labels, batch_size, model.name, start_idx, n_unique_segments, classes_guessed) if not prompt else prompt
            full_prompt = replace_images_in_prompt(full_prompt, prompt_imgs)

            resp_dict, tokens, error_status = model.send_message(full_prompt)
            compl_tokens += tokens[0]
            prompt_tokens += tokens[1]
            if not error_status:
                resp_dict.pop("reasoning_steps", None)
            else:
                resp_dict = {}

            for k in range(len(indices)):
                all_results[str(indices[k])] = resp_dict.get(str(k), "unknown")

            classes_guessed += [v for k, v in resp_dict.items() if k != "reasoning_steps"]
            classes_guessed = list(set(classes_guessed))

        all_results["file_name"] = file_name[img_idx].strip()
        resp_dict_list.append(all_results)

    if return_dict:
        return resp_dict_list, (compl_tokens, prompt_tokens), error_status
    else:
        seg_maps = model.eval(eval='eval_segment', predictions=resp_dict_list, n_segments=n_segments, labels=labels, color_map=color_map, visualise=True)
        return seg_maps, (compl_tokens, prompt_tokens)
