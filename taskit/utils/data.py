import base64
import os
from copy import deepcopy
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float


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


def save_images(imgs: List[Image.Image], save_path: str = 'temp_images'):
    """Function to save a list of images to a directory."""
    file_paths = []
    os.makedirs(save_path, exist_ok=True)
    for i, img in enumerate(imgs):
        file_path = os.path.join(save_path, f"temp_{i}.png")
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths


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
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if "image_url" in content and content["image_url"]["url"] == "<img>":
                        # Replace <img> with the corresponding encoded image
                        content["image_url"]["url"] = f"data:image/png;base64,{encode_image(images[img_index])}"
                        img_index += 1
            else:
                if isinstance(message["content"], dict):
                    content = message["content"]
                    if "image_url" in content and content["image_url"]["url"] == "<img>":
                        # Replace <img> with the corresponding encoded image
                        content["image_url"]["url"] = f"data:image/png;base64,{encode_image(images[img_index])}"
                        img_index += 1

    return full_prompt_output


def find_adjacent_segments(segments):
    n_segments = len(np.unique(segments))
    adjacency_matrix = np.zeros((n_segments, n_segments), dtype=bool)

    # Get the shape of the image
    height, width = segments.shape

    # Check each pixel and its 4-connected neighbors (up, down, left, right)
    for y in range(height):
        for x in range(width):
            segment_id = segments[y, x] - 1
            # Check right neighbor
            if x < width - 1 and segments[y, x + 1] - 1 != segment_id:
                adjacency_matrix[segment_id, segments[y, x + 1] - 1] = True
                adjacency_matrix[segments[y, x + 1] - 1, segment_id] = True
            # Check down neighbor
            if y < height - 1 and segments[y + 1, x] - 1 != segment_id:
                adjacency_matrix[segment_id, segments[y + 1, x] - 1] = True
                adjacency_matrix[segments[y + 1, x] - 1, segment_id] = True

    return adjacency_matrix


# ==Visual Prompts==================================================================


def draw_around_superpixel(
    img: Image.Image,
    segments: np.ndarray,
    seg_number: int,
    seg_type: str,
    color: str = "red",
    crop_width: int = None,
    radius: Optional[int] = 16,
    rectangle_width: int = 4,
) -> Image.Image:
    """Function which marks a segment in an image with a bounding box, center point or boundary.
    If seg_type is 'rectangle', a bounding rectangle is drawn around the segment.
    If seg_type is 'point', a rectangle inscribed in the segment is found, and a filled circle of radius 'radius' is drawn at the center.
    If seg_type is 'curve', the boundary of the segment is marked with a color.

    Args:
        img (Image.Image): PIL Image object
        segments (np.ndarray): Segmentation map formed by SLIC
        seg_number (int): Segment number/ID to mark
        seg_type (str): Type of visual marker to use. Can be 'rectangle', 'point' or 'curve'
        color (str): Color of the visual marker. Can be 'red', 'green', 'blue' or 'yellow'
        crop_width (int): Width of the crop around the segment. If None, no cropping is done, and full image with marker is returned
        radius (Optional[int]): Radius of the point marker. Only used if seg_type is 'point'
        rectangle_width (int): Width of the rectangle border. Only used if seg_type is 'rectangle'
    """
    assert seg_type in ['rectangle', 'point', 'curve'], "Invalid segment type"
    assert color in ['red', 'green', 'blue', 'yellow'], "Invalid color"

    # Convert image to float format
    width, height = img.size
    img = img_as_float(deepcopy(img))

    # Create a mask for the segment
    mask = np.zeros_like(segments)
    mask[segments == seg_number] = 1

    if seg_type == 'rectangle':
        # Draw a bounding rectangle around the segment
        rows, cols = np.where(mask)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([min_col, min_row, max_col, max_row], outline=color, width=rectangle_width)

    elif seg_type == 'point':
        rows, cols = np.where(mask)

        # Find the bounding rectangle of the segment
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        inscribed_min_row, inscribed_max_row = min_row, max_row
        inscribed_min_col, inscribed_max_col = min_col, max_col

        # Adjust the bounds to find the inscribed rectangle
        for r in range(min_row, max_row + 1):
            if not np.all(mask[r, min_col:max_col + 1]):
                inscribed_min_row = r
                break

        for r in range(max_row, min_row - 1, -1):
            if not np.all(mask[r, min_col:max_col + 1]):
                inscribed_max_row = r
                break

        for c in range(min_col, max_col + 1):
            if not np.all(mask[min_row:max_row + 1, c]):
                inscribed_min_col = c
                break

        for c in range(max_col, min_col - 1, -1):
            if not np.all(mask[min_row:max_row + 1, c]):
                inscribed_max_col = c
                break

        # Calculate center of the inscribed rectangle
        center_row = (inscribed_min_row + inscribed_max_row) // 2
        center_col = (inscribed_min_col + inscribed_max_col) // 2

        # Draw the center point
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)
        draw.ellipse([(max(0, center_col-radius), max(0, center_row-radius)), (min(width, center_col+radius), min(height, center_row+radius))], fill=color, outline=color)

    elif seg_type == 'curve':
        rgb = (1, 0, 0) if color == "red" else (0, 1, 0) if color == "green" else (0, 0, 1) if color == "blue" else (1, 1, 0)
        img_with_boundaries = mark_boundaries(img, mask, color=rgb)

        img_pil = Image.fromarray((img_with_boundaries * 255).astype(np.uint8))

    if crop_width is None:
        return img_pil
    else:  # Crop 'crop_width' pixels around the segment
        rows, cols = np.where(mask)

        # Find the bounding rectangle of the segment
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        min_row, max_row = max(0, min_row - crop_width), min(height, max_row + crop_width)
        min_col, max_col = max(0, min_col - crop_width), min(width, max_col + crop_width)

        return img_pil.crop((min_col, min_row, max_col, max_row))


# ==Image Manipulation or Creation==================================================================


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


def flood_superpixels(img, all_pred, label_to_id, n_segments, ignore_index=255):
    """ Returns a numpy array with "label" values for each superpixel
    Args:
        img: PIL Image object
        all_pred: List of predictions for each superpixel (from 1 to n_unique_segments)
        label_to_id: Dict mapping from segment labels to segment ids
        n_segments: Number of segments image was segmented into. This is the parameter used in SLIC (and may not be equal to the number of unique segments)
    """

    segments = slic(img_as_float(img), n_segments=n_segments, sigma=5)
    height, width = img.size
    dense_array = np.zeros_like(segments)
    for i, seg in enumerate(np.unique(segments)):
        dense_array[segments == seg] = label_to_id[all_pred[i]] if all_pred[i] in label_to_id else ignore_index

    return dense_array


def colorize(pred_array, label_to_id, color_map):
    """ Colorizes the segmentation map """
    height, width = pred_array.shape
    pred_img = np.zeros((height, width, 3), dtype=np.uint8)
    for label, label_id in label_to_id.items():
        pred_img[pred_array == label_id] = color_map[label]

    return pred_img


# ==Sampling==================================================================


def sample_segments(segments, min_picked=1, min_samples=100, seed=25, shuffle: bool = True) -> List[Tuple]:
    """
    Sample segment pairs
    """
    np.random.seed(seed)
    tuples, n_picked = [], 0
    segment_ids = np.sort(np.unique(segments))
    while len(tuples) < min_samples or n_picked < min_picked:
        if shuffle:
            np.random.shuffle(segment_ids)
        for i in range(0, len(segment_ids), 2):
            if i + 1 < len(segment_ids):
                tuples.append((segment_ids[i], segment_ids[i + 1]))

        if len(segment_ids) % 2:
            tuples.append((segment_ids[-1], segment_ids[0]))

        n_picked += 1
    if len(tuples) > min_samples and n_picked > min_picked:
        return tuples[:min_samples]

    return tuples
