import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from taskit.mfm import GPT4o, GeminiPro, ClaudeSonnet
from taskit.eval import eval_classify, eval_object, eval_depth, eval_segment, eval_normals, eval_grouping
from taskit.tasks import classify, object, depth, segment, normals, grouping
from taskit.utils.data import crop_img


class DemoSampler:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        model_name_2_model = {
            "gpt-4o-2024-08-06": GPT4o,
            "gemini-1.5-pro": GeminiPro,
            "claude-3-5-sonnet-20240620": ClaudeSonnet
        }
        if model_name not in model_name_2_model:
            raise ValueError(f"Model name {model_name} is not supported")
        self.model = model_name_2_model[model_name](api_key)

    def __call__(self, image_path):
        def classify_task():
            return self.model.predict('classify', image_path)

        def object_task():
            return self.model.predict('detect', image_path, object_list=['zebra'])

        def segment_task():
            return self.model.predict('segment', image_path)

        def grouping_task():
            return self.model.predict('group', image_path, point_list=[[300, 250]])

        def depth_task():
            return self.model.predict('depth', image_path, n_threads=20)

        def normals_task():
            return self.model.predict('normals', image_path, n_threads=20)

        # Execute tasks concurrently
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_task = {
                "classify": executor.submit(classify_task),
                "object": executor.submit(object_task),
                "segment": executor.submit(segment_task),
                "grouping": executor.submit(grouping_task),
                "depth": executor.submit(depth_task),
                "normals": executor.submit(normals_task)
            }

            # Gather the results as they complete
            results = {task: future.result() for task, future in future_to_task.items()}
            result_imgs = {task: val[0][0] for task, val in results.items()}
            compl_tokens = sum([val[1][0] for val in results.values()])
            prompt_tokens = sum([val[1][1] for val in results.values()])

        return result_imgs, (compl_tokens, prompt_tokens)

    def visualize(self, result_imgs: Dict[str, Any]) -> Image.Image:
        # Create text image with border
        text_img = Image.new('RGB', (224, 224), 'white')
        font = ImageFont.truetype(os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans-Bold.ttf'), 36)
        draw = ImageDraw.Draw(text_img)
        text_bbox = font.getbbox(result_imgs['classify'])

        # Add padding for the box
        padding = 20
        box_x1 = (224 - text_bbox[2]) // 2 - padding
        box_y1 = (224 - text_bbox[3]) // 2 - padding
        box_x2 = (224 + text_bbox[2]) // 2 + padding
        box_y2 = (224 + text_bbox[3]) // 2 + padding

        # Draw box and text
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline='black', width=2)
        draw.text(
            ((224 - text_bbox[2]) // 2, (224 - text_bbox[3]) // 2),
            result_imgs['classify'], font=font, fill='black'
        )

        # Create figure with larger title font size and bold titles
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 28
        plt.rcParams['axes.titleweight'] = 'bold'

        titles = ['Classification', 'Object Detection', 'Segmentation', 'Grouping', 'Depth', 'Normals']
        images = [text_img, result_imgs['object']] + [Image.fromarray(result_imgs[k]) for k in ['segment', 'grouping']] + [Image.fromarray((result_imgs[k] * 255).astype(np.uint8)) for k in ['depth', 'normals']]

        for ax, img, title in zip(axes.flat, images, titles):
            ax.imshow(crop_img(img) if title != 'Classify' else img)
            ax.set_title(title, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            [ax.spines[spine].set_visible(False) for spine in ['top', 'right', 'bottom', 'left']]

        plt.tight_layout()
        fig.canvas.draw()

        # Convert to PIL Image and return
        result = Image.fromarray(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        ))
        plt.close(fig)
        return result
