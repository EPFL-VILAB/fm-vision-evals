import base64
from copy import deepcopy
from io import BytesIO

from PIL import Image


# ==General Utils==================================================================


def encode_image(image: Image.Image) -> str:
    """Function to encode the image."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # You can specify the format if needed
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()  # closing the buffer
    return encoded_string


def decode_image(encoded_string: str) -> Image:
    """Function to decode the image."""
    decoded_string = base64.b64decode(encoded_string)
    with BytesIO(decoded_string) as buffer:  # Use context manager
        image = Image.open(buffer)
        image.load()  # Force the image to be fully loaded while the buffer is still open
    return image


def replace_images_in_prompt(full_prompt_output: dict, images: list):
    """
    Replaces all occurrences of <img> in the 'messages' of the given prompt dict with the encoded images.

    Args:
    - full_prompt_output (dict): The dictionary returned from full_prompt_cls.
    - encoded_images (list): List of encoded images to replace <img> in the messages.
    """
    messages = full_prompt_output["messages"]

    # Loop over the user messages (excluding system messages)
    img_index = 0
    for message in messages:
        if message["role"] == "user":
            # Loop through content to find where <img> is located
            for content in message["content"]:
                if "image_url" in content and content["image_url"]["url"] == "<img>":
                    # Replace <img> with the corresponding encoded image
                    content["image_url"]["url"] = f"data:image/png;base64,{encode_image(images[img_index])}"
                    img_index += 1

    return full_prompt_output


# ==Visual Prompts==================================================================


# ==Image Manipulation==================================================================

def crop_img(
    img: Image.Image,
    crop_size: int = 224,
    shortest_side: int = 224
) -> Image:
    """ Center cropping function for ImageNet"""

    img = deepcopy(img)
    width, height = img.size
    # resize so the smallest side is the shortest_side
    if width < height:
        img = img.resize((shortest_side, int(height * shortest_side / width)))
    else:
        img = img.resize((int(width * shortest_side / height), shortest_side))
    width, height = img.size
    left, top = (width - crop_size) // 2, (height - crop_size) // 2
    right, bottom = (width + crop_size) // 2, (height + crop_size) // 2

    return img.crop((left, top, right, bottom))
