import google.generativeai as genai

# ==Classification==================================================================


# --System Prompt----------------------------------------------------------------


def system_prompts_cls(prompt_no: int, imagenet_classes: list, batch_size: int) -> str:
    assert prompt_no in [1, 2, 3, 4, 5], "Invalid prompt number."

    n_classes = len(imagenet_classes)
    if prompt_no == 1:
        system_prompt = f"""You will be provided with a set of {n_classes} classes in a list. The user will provide you with {batch_size} images, """ +\
                        """and your job is to correctly identify the label corresponding to the images. Only output the label """ +\
                        f"""corresponding to the image, and nothing else. Output the class name in a JSON, with key "<image number>". For example, {{"1": "image 1 class", "2": "image 2 class", ... , "{batch_size}": "image {batch_size} class"}}. Classes: {imagenet_classes}"""

    elif prompt_no == 2:
        system_prompt = f"""You are an AI assistant tasked with classifying images. You will be provided with {batch_size} images and must assign each image to one of the {n_classes} classes. Follow these instructions carefully:\n""" +\
                        f"""1. Here is the list of {n_classes} classes you will use for classification:\n<image_classes>\n{imagenet_classes}\n</image_classes>\n\n 2. Present your classifications in JSON format, with keys representing the image number (starting from 1 and ending at {batch_size}) and values representing the class assigned to the image. For example: {{"1": "image 1 class", "2": "image 2 class", "3": "image 3 class", ... , "{batch_size}": "image {batch_size} class"}}\n\n""" +\
                        f"""3. If you are unsure about a classification:\na. Choose the most likely class based on the available information.\nb. Do not express uncertainty in your output or suggest alternative classes.\n\n4.After classifying all images, present your final output in a single JSON object. Ensure that you have an entry for each image, numbered from 1 to {batch_size}."""

    elif prompt_no == 3:
        system_prompt = """You are an advanced AI image classification system with expertise in recognizing a wide variety of objects, animals, and scenes. Your task is to classify a batch of images into predefined categories with high accuracy.\n\n""" +\
                        """First, familiarize yourself with the list of image classes you will be using for classification:\n\n""" +\
                        f"""<image_classes>\n{imagenet_classes}\n</image_classes>\n\n""" +\
                        f"""You will be presented with {batch_size} images to classify. Each image must be assigned to one of the classes listed above. Follow these instructions carefully:\n\n""" +\
                        """1. Analyze each image thoroughly, considering all visible elements, objects, and context clues.\n\n""" +\
                        """2. Select the most appropriate class for each image based on your analysis. If an image contains multiple objects or elements, choose the class that best represents the main subject or most prominent feature of the image.\n\n""" +\
                        f"""3. Present your classifications in a JSON format. The keys should represent the image number (starting from 1 and ending at {batch_size}), and the values should represent the assigned class. For example:\n\n""" +\
                        f"""{{"1": "{imagenet_classes[0]}", "2": "{imagenet_classes[2]}", "3": "{imagenet_classes[1]}", ... , "{batch_size}": "{imagenet_classes[9]}"}}\n\n""" +\
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
                        f"""Strive for maximum accuracy in your classifications. If you're unsure about a particular image, choose the class that most closely matches the image content. The classes to choose from are: {imagenet_classes}."""
    elif prompt_no == 5:
        system_prompt = f"""You are a highly advanced image classification system. You have been trained on a vast array of visual data and can accurately identify objects, scenes, and concepts across a wide range of categories. You will be presented with {batch_size} images for classification. Your task is to analyze each image carefully, considering multiple aspects such as shape, color, texture, context, and any distinguishing features. Draw upon your extensive knowledge to determine the most accurate label for each image from the provided set of {n_classes} classes.\n\n""" +\
                        """Output your classifications in JSON format, with each image number as the key and the corresponding class name as the value. Be as precise and specific as possible in your classifications. If you're unsure, choose the most likely class based on the visual information available. Here's the expected output format:\n\n""" +\
                        f"""{{"1": "class_name_1", "2": "class_name_2", ..., "{batch_size}": "class_name_{batch_size}"}}\n\n""" +\
                        f"""Remember, only output the JSON object with your classifications. Do not include any explanations or additional text. Classes: {imagenet_classes}."""

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

    elif model == 'claude-3-5-sonnet-20240620':
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


# --Full Prompt----------------------------------------------------------------

def full_prompt_cls(prompt_no: int, imagenet_classes: list, batch_size: int, model: str):
    messages = []

    system_prompt = system_prompts_cls(prompt_no, imagenet_classes, batch_size)
    messages.append({"role": "system", "content": system_prompt})
    for i in range(batch_size):
        user_prompt = f"Please identify the class of the image provided. The class has to belong to one of the classes specified in the system prompt. Output the answer in a JSON, with key '{i+1}'."
        messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": "<img>", "detail": "high"}}]})

    json_schema = json_schema_cls(batch_size, model)

    return {"messages": messages, "json_schema": json_schema}


# ==Object Detection==================================================================


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


# ==Semantic Segmentation==================================================================


# --System Prompt----------------------------------------------------------------

# change number of classes for other datasets***
def system_prompts_semseg(prompt_no: int, seg_classes: list, batch_size: int, shape: str, classes_guessed=[]):
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


# ==Depth Prediction==================================================================


# --System Prompt----------------------------------------------------------------

##### REMEMBER TO CHANGE SYSTEM PROMPT NUMBERS EVERYWHERE FOR DEPTH
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


# ==Surface Normal Prediction==================================================================


# --System Prompt----------------------------------------------------------------


def system_prompts_normals_superpix(prompt_no: int, shape: str) -> str:
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
