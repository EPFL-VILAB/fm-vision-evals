import json
from abc import ABCMeta
from copy import deepcopy
from typing import Dict, List, Optional, Union

import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from together import Together

from taskit.utils.data import decode_image
import taskit.mfm_configs as configs

import re
import requests

OPENAI_MODELS = [
    "gpt-4o-2024-08-06",
    "o1-2024-12-17",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "gpt-image-1",
]
GEMINI_MODELS = ["gemini-1.5-pro-001", "gemini-2.0-flash-001"]
CLAUDE_MODELS = ["claude-3-5-sonnet-20240620"]
TOGETHER_MODELS = ["llama-3.2-90b"]
QWEN2_MODELS = ["qwen2-vl-72b-instruct"]


def format_for_responses(messages: list[dict]) -> list[dict]:
    """
    Convert a list of messages with generic 'text' and 'image_url' parts
    into the Responses API 'input' format, using 'input_text' and 'input_image'.
    """
    formatted = []
    for msg in messages:
        content_items = []
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if isinstance(part, dict) and part["type"] == "text":
                    content_items.append({"type": "input_text", "text": part["text"]})
                elif isinstance(part, dict) and part["type"] == "image_url":
                    # Note: Responses API only supports base64 or inline URLs for images
                    url = part["image_url"]["url"]
                    content_items.append({"type": "input_image", "image_url": url})
        else:
            content_items = [{"type": "input_text", "text": msg["content"]}]
        formatted.append({"role": msg["role"], "content": content_items})
    return formatted


# ==Class Definitions==================================================================


class TaskRegistryABCMeta(ABCMeta, type):

    def __new__(cls, name, bases, attrs):
        # Create the new class
        new_class = super().__new__(cls, name, bases, attrs)
        # Initialize a class-specific TASK_REGISTRY for the new class
        for base in bases:
            if hasattr(base, "TASK_REGISTRY"):
                new_class.TASK_REGISTRY = deepcopy(base.TASK_REGISTRY)
                return new_class

        # Initialize a class-specific EVAL_REGISTRY for the new class
        for base in bases:
            if hasattr(base, "EVAL_REGISTRY"):
                new_class.EVAL_REGISTRY = deepcopy(base.EVAL_REGISTRY)
                return new_class

        new_class.TASK_REGISTRY = {}
        new_class.EVAL_REGISTRY = {}

        return new_class


class MFMWrapper(metaclass=TaskRegistryABCMeta):
    default_evals = {
        "classify": "eval_classify",
        "detect": "eval_detect",
        "segment": "eval_segment",
        "group": "eval_group",
        "depth": "eval_depth",
        "normals": "eval_normals",
    }

    @classmethod
    def register_task(cls, task_name):
        def decorator(func):
            if cls in [MFMWrapper, TextMFMWrapper, ImageMFMWrapper]:
                # Register the function in MFMWrapper's registry
                cls.TASK_REGISTRY[task_name] = func
                # Also register it in all subclasses
                for subclass in cls.__subclasses__():
                    subclass.TASK_REGISTRY[task_name] = func
            else:
                # If it's a subclass, only register in that subclass's registry
                cls.TASK_REGISTRY[task_name] = func
            return func

        return decorator

    @classmethod
    def register_eval(cls, eval_name):
        def decorator(func):
            if cls in [MFMWrapper, TextMFMWrapper, ImageMFMWrapper]:
                # Register the function in MFMWrapper's registry
                cls.EVAL_REGISTRY[eval_name] = func
                # Also register it in all subclasses
                for subclass in cls.__subclasses__():
                    subclass.EVAL_REGISTRY[eval_name] = func
            else:
                # If it's a subclass, only register in that subclass's registry
                cls.EVAL_REGISTRY[eval_name] = func
            return func

        return decorator

    def predict(self, task, file_name: Union[List, str], **kwargs):
        if task in self.TASK_REGISTRY:
            task_func = self.TASK_REGISTRY[task]
        else:
            raise ValueError(
                f"Task {task} not supported, please choose from {self.TASK_REGISTRY.keys()}"
            )
        return task_func(model=self, file_name=file_name, **kwargs)

    def eval(self, eval: Optional[str], predictions: Union[List, str], **kwargs):
        if eval is None:
            eval = self.default_evals[kwargs["task"]]
        if eval in self.EVAL_REGISTRY:
            eval_func = self.EVAL_REGISTRY[eval]
        else:
            raise ValueError(
                f"Evaluation {eval} not supported, please choose from {self.EVAL_REGISTRY.keys()}"
            )
        kwargs.pop("task", None)
        return eval_func(predictions, **kwargs)

    def send_message(self, message: Dict):
        raise NotImplementedError


class TextMFMWrapper(MFMWrapper):
    """Base class for text-generating models"""

    pass


class ImageMFMWrapper(MFMWrapper):
    """Base class for image-generating models"""

    pass


# --GPT----------------------------------------------------------------


class GPT(TextMFMWrapper):

    def __init__(
        self,
        api_key,
        default_settings=configs.GPT4O_DEFAULTS,
        name="gpt-4o-2024-08-06",
        **kwargs,
    ):
        self.client = OpenAI(api_key=api_key)
        self.name = name
        self.seed = 42
        self.default_settings = default_settings
        self.reasoning_effort = kwargs.get("reasoning_effort", "low")

        if self.name[0] == "o":
            print(f"Using reasoning effort: {self.reasoning_effort}")

    def send_message(self, message: Dict):
        messages, json_schema = message["messages"], message["json_schema"]
        compl_tokens, prompt_tokens = 0, 0
        for attempt in range(3):
            try:
                gpt_kwargs = (
                    {"max_tokens": 4000, "temperature": 0}
                    if self.name == "gpt-4o-2024-08-06"
                    else {
                        "max_completion_tokens": 25000,
                        "reasoning_effort": self.reasoning_effort,
                    }
                )
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    seed=self.seed,
                    response_format=(
                        {"type": "json_schema", "json_schema": json_schema}
                        if (json_schema is not None)
                        else {"type": "text"}
                    ),
                    **gpt_kwargs,
                )
                compl_tokens, prompt_tokens = (
                    compl_tokens + response.usage.completion_tokens,
                    prompt_tokens + response.usage.prompt_tokens,
                )
                resp_dict = json.loads(response.choices[0].message.content)
                resp_dict = {} if resp_dict is None else resp_dict
                return resp_dict, (compl_tokens, prompt_tokens), False
            except Exception as e:
                print(f"Error in sending message: {e}")
                if attempt == 2:
                    return {}, (0, 0), True

    def predict(self, task, file_name, **kwargs):
        if (
            task in self.default_settings
            and "task_specific_args" in self.default_settings[task]
        ):
            default_settings = self.default_settings[task]["task_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().predict(task, file_name, **kwargs)

    def eval(self, eval, predictions, **kwargs):
        eval_key = eval.removeprefix("eval_")
        if (
            eval_key in self.default_settings
            and "eval_specific_args" in self.default_settings[eval_key]
        ):
            default_settings = self.default_settings[eval_key]["eval_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().eval(eval, predictions, **kwargs)


class GPTImage(ImageMFMWrapper):

    def __init__(
        self,
        api_key,
        name="gpt-image-1",
        default_settings=configs.GPT_IMAGE_DEFAULTS,
        **kwargs,
    ):
        self.client = OpenAI(api_key=api_key)
        self.default_settings = default_settings
        self.name = name

    def send_message(self, prompt: str, image: List, **kwargs):
        compl_tokens, prompt_tokens = 0, 0
        for attempt in range(3):
            try:
                response = self.client.images.edit(
                    model=self.name,
                    image=image,
                    prompt=prompt,
                    size="1024x1024",
                    quality="high",
                )
                compl_tokens, prompt_tokens = (
                    compl_tokens + response.usage.input_tokens,
                    prompt_tokens + response.usage.output_tokens,
                )
                resp_str = response.data[0].b64_json
                return resp_str, (compl_tokens, prompt_tokens), False
            except Exception as e:
                print(f"Error in sending message: {e}")
                if attempt == 2:
                    return None, (compl_tokens, prompt_tokens), True

    def predict(self, task, file_name, **kwargs):
        if (
            task in self.default_settings
            and "task_specific_args" in self.default_settings[task]
        ):
            default_settings = self.default_settings[task]["task_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().predict(task, file_name, **kwargs)


# --Gemini----------------------------------------------------------------


class GeminiPro(TextMFMWrapper):

    def __init__(
        self,
        api_key,
        name="gemini-1.5-pro-001",
        default_settings=configs.GEMINI_DEFAULTS,
    ):
        genai.configure(api_key=api_key)
        self.client = genai
        self.name = name
        self.default_settings = default_settings

    def parse_message(self, all_messages: Dict):
        system_prompt = all_messages["messages"][0]["content"]
        new_messages = []
        for message in all_messages["messages"][1:]:
            if message["role"] == "user":
                if isinstance(message["content"], str):
                    new_messages.append(message["content"])
                else:  # message is list
                    for m in message["content"]:
                        if m["type"] == "text":
                            new_messages.append(m["text"])
                        elif m["type"] == "image_url":
                            img = decode_image(
                                m["image_url"]["url"].replace(
                                    "data:image/png;base64,", ""
                                )
                            )
                            new_messages.append(img)

        json_schema = all_messages["json_schema"]
        return system_prompt, new_messages, json_schema

    def send_message(self, message: Dict):
        system_prompt, messages, json_schema = self.parse_message(message)
        compl_tokens, prompt_tokens = 0, 0
        for attempt in range(5):
            try:
                model = self.client.GenerativeModel(
                    model_name=self.name, system_instruction=system_prompt
                )
                response = model.generate_content(
                    messages,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                    generation_config={
                        "temperature": 0,
                        "max_output_tokens": 4000,
                        "response_mime_type": "application/json",
                        "response_schema": json_schema,
                    },
                    stream=False,
                )

                resp_dict = json.loads(response.text)
                prompt_tokens, compl_tokens = (
                    prompt_tokens + response.usage_metadata.prompt_token_count,
                    compl_tokens + response.usage_metadata.candidates_token_count,
                )
                resp_dict = {} if resp_dict is None else resp_dict
                return resp_dict, (compl_tokens, prompt_tokens), False

            except Exception as e:
                print(f"Error in sending message: {e}")
                if attempt == 2:
                    return {}, (0, 0), True

    def predict(self, task, file_name, **kwargs):
        if (
            task in self.default_settings
            and "task_specific_args" in self.default_settings[task]
        ):
            default_settings = self.default_settings[task]["task_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().predict(task, file_name, **kwargs)

    def eval(self, eval, predictions, **kwargs):
        eval_key = eval.removeprefix("eval_")
        if (
            eval_key in self.default_settings
            and "eval_specific_args" in self.default_settings[eval_key]
        ):
            default_settings = self.default_settings[eval_key]["eval_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().eval(eval, predictions, **kwargs)


# --Claude----------------------------------------------------------------


class ClaudeSonnet(TextMFMWrapper):

    def __init__(
        self,
        api_key,
        default_settings=configs.CLAUDE_DEFAULTS,
        name="claude-3-5-sonnet-20240620",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.name = name
        self.default_settings = default_settings

    def parse_message(self, all_messages: Dict):
        system_prompt = all_messages["messages"][0]["content"]
        new_messages, content = [], []
        for message in all_messages["messages"][1:]:
            if message["role"] == "user":
                if isinstance(message["content"], str):
                    content.append({"type": "text", "text": message["content"]})
                else:  # message is list
                    for m in message["content"]:
                        if m["type"] == "text":
                            content.append({"type": "text", "text": m["text"]})
                        elif m["type"] == "image_url":
                            img_data = m["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            )
                            content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": img_data,
                                    },
                                }
                            )

        json_schema, expected_keys = all_messages["json_schema"]

        content.append({"type": "text", "text": json_schema})
        new_messages.append({"role": "user", "content": content})
        new_messages.append(
            {"role": "assistant", "content": "{"}
        )  # pre-filling for correct format

        return system_prompt, new_messages, expected_keys

    def send_message(self, message: Dict):
        system_prompt, messages, expected_keys = self.parse_message(message)

        compl_tokens, prompt_tokens, error_status = 0, 0, True
        for _ in range(5):
            try:
                response = None
                response = self.client.messages.create(
                    model=self.name,
                    max_tokens=4000,
                    temperature=0,
                    system=system_prompt,
                    messages=messages,
                )

                str_message = response.content[0].text
                resp_dict = json.loads("{" + str_message)
                compl_tokens, prompt_tokens = (
                    compl_tokens + response.usage.output_tokens,
                    prompt_tokens + response.usage.input_tokens,
                )
                if all([key in resp_dict for key in expected_keys]) and len(
                    resp_dict
                ) == len(expected_keys):
                    error_status = False
                    break

            except Exception as e:
                print(f"Error in sending message: {e}")
                resp_dict = {}
                if (response is not None) and response.usage:
                    compl_tokens, prompt_tokens = (
                        compl_tokens + response.usage.output_tokens,
                        prompt_tokens + response.usage.input_tokens,
                    )

        return resp_dict, (compl_tokens, prompt_tokens), error_status

    def predict(self, task, file_name, **kwargs):
        if (
            task in self.default_settings
            and "task_specific_args" in self.default_settings[task]
        ):
            default_settings = self.default_settings[task]["task_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().predict(task, file_name, **kwargs)

    def eval(self, eval, predictions, **kwargs):
        eval_key = eval.removeprefix("eval_")
        if (
            eval_key in self.default_settings
            and "eval_specific_args" in self.default_settings[eval_key]
        ):
            default_settings = self.default_settings[eval_key]["eval_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().eval(eval, predictions, **kwargs)


# --Llama----------------------------------------------------------------


class Llama_Together(TextMFMWrapper):

    def __init__(
        self, api_key, default_settings=configs.LLAMA_DEFAULTS, name="llama-3.2-90b"
    ):
        self.client = Together(api_key=api_key)
        self.name = name
        self.seed = 42
        self.default_settings = default_settings

    def _image_token_cost(self, img_str: str):
        img = decode_image(img_str)
        return (
            min(2, max(img.height // 560, 1)) * min(2, max(img.width // 560, 1)) * 1601
        )

    def parse_message(self, all_messages: Dict):
        messages = all_messages["messages"]
        json_schema, expected_keys = all_messages["json_schema"]

        messages.append({"role": "user", "content": json_schema})
        messages.append(
            {"role": "user", "content": "Output only the json, and nothing else."}
        )

        # Find the token cost for images
        img_token_costs = 0
        for message in all_messages["messages"][1:]:
            if message["role"] == "user":
                if isinstance(message["content"], list):
                    for m in message["content"]:
                        if m["type"] == "image_url":
                            img_data = m["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            )
                            img_token_costs += self._image_token_cost(img_data)

        return messages, expected_keys, img_token_costs

    def send_message(self, message: Dict):
        messages, expected_keys, img_token_costs = self.parse_message(message)

        compl_tokens, prompt_tokens, error_status = 0, 0, True
        for times in range(5):
            try:
                response = None
                response = self.client.chat.completions.create(
                    model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    messages=messages,
                    max_tokens=4000,
                    seed=self.seed,
                    temperature=0,
                )
                json_start = response.choices[0].message.content.find("{")
                json_end = response.choices[0].message.content.rfind("}") + 1
                json_content = response.choices[0].message.content[json_start:json_end]
                resp_dict = json.loads(json_content)
                compl_tokens, prompt_tokens = (
                    compl_tokens + response.usage.completion_tokens,
                    prompt_tokens + response.usage.prompt_tokens + img_token_costs,
                )
                if all([key in resp_dict for key in expected_keys]) and len(
                    resp_dict
                ) == len(expected_keys):
                    error_status = False
                    break

            except Exception as e:
                print(f"Error in sending message: {e}")
                resp_dict = {}
                if (response is not None) and response.usage:
                    compl_tokens, prompt_tokens = (
                        compl_tokens + response.usage.completion_tokens,
                        prompt_tokens + response.usage.prompt_tokens,
                    )
                if not isinstance(e, json.JSONDecodeError) and times == 2:
                    break
                if response is not None and response.usage.completion_tokens >= 4000:
                    break
                elif (
                    response is not None
                    and response.choices[0].message.content
                    == """I'm not going to engage in this conversation topic."""
                ):
                    break

        return resp_dict, (compl_tokens, prompt_tokens), error_status

    def predict(self, task, file_name, **kwargs):
        if (
            task in self.default_settings
            and "task_specific_args" in self.default_settings[task]
        ):
            default_settings = self.default_settings[task]["task_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().predict(task, file_name, **kwargs)

    def eval(self, eval, predictions, **kwargs):
        eval_key = eval.removeprefix("eval_")
        if (
            eval_key in self.default_settings
            and "eval_specific_args" in self.default_settings[eval_key]
        ):
            default_settings = self.default_settings[eval_key]["eval_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().eval(eval, predictions, **kwargs)


# --Qwen2-VL-------------------------------------------------------------


class Qwen2VL(TextMFMWrapper):

    def __init__(
        self,
        address=None,
        default_settings=configs.QWEN2_DEFAULTS,
        output_format="json",
        name="qwen2-vl-72b-instruct",
    ):
        self.name = name
        self.default_settings = default_settings
        self.addr = address or default_settings.get("address", None)
        self.output_format = output_format

    def send_request(
        self, messages: Dict, max_tokens: int, output_format: str = "json"
    ):
        try:
            response = requests.post(
                self.addr,
                json={
                    "messages": messages,
                    "max_new_tokens": max_tokens,
                    "output_format": output_format,
                },
            )
            response = response.json()
            return response
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def restructure_message(self, messages: Dict):
        for m in messages:
            if m["role"] == "user":
                if isinstance(m["content"], list):
                    for content in m["content"]:
                        if content["type"] == "image_url":
                            if "url" in content["image_url"]:
                                content["image_url"] = content["image_url"]["url"]

    def send_message(self, message: Dict):
        messages, json_schema = message["messages"], message["json_schema"]
        messages = self.restructure_message(messages)

        if json_schema:
            if isinstance(json_schema, str):
                messages.append({"role": "assistant", "content": json_schema})
            else:
                raise ValueError("json_schema should be a string")

        try:
            response = self.send_request(
                messages=messages,
                max_tokens=4000,
                output_format=self.output_format,
            )

            response = response["output"]

            if self.output_format == "text":
                response = [json.loads(re.sub(r"[\n\t]+", "", o)) for o in response]

            if isinstance(response, list) and len(response) == 1:
                response = response[0]

            resp_dict = response

            return resp_dict, (0, 0), False

        except Exception as e:
            print(f"Error in sending message: {e}")
            return {}, (0, 0), True

    def predict(self, task, file_name, **kwargs):
        if (
            task in self.default_settings
            and "task_specific_args" in self.default_settings[task]
        ):
            default_settings = self.default_settings[task]["task_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().predict(task, file_name, **kwargs)

    def eval(self, eval, predictions, **kwargs):
        eval_key = eval.removeprefix("eval_")
        if (
            eval_key in self.default_settings
            and "eval_specific_args" in self.default_settings[eval_key]
        ):
            default_settings = self.default_settings[eval_key]["eval_specific_args"]
            for k, v in default_settings.items():
                kwargs[k] = kwargs.get(k, v)
        return super().eval(eval, predictions, **kwargs)


# ==Functions==================================================================


def get_mfm_wrapper(
    model_name: str, api_key: str = None, address: str = None, **kwargs
) -> MFMWrapper:
    if model_name in OPENAI_MODELS:
        if model_name.lower() == "gpt-image-1":
            return GPTImage(
                api_key=api_key,
                name=model_name,
                default_settings=configs.GPT_IMAGE_DEFAULTS,
            )
        if model_name.lower()[0] == "o":
            default_settings = (
                configs.O1_DEFAULTS
            )  # common settings were used for all o-series models
        else:
            default_settings = configs.GPT4O_DEFAULTS
        return GPT(
            api_key=api_key,
            name=model_name,
            default_settings=default_settings,
            **kwargs,
        )
    elif model_name in GEMINI_MODELS:
        if model_name.lower() == "gemini-2.0-flash-001":
            default_settings = configs.GEMINI_2_DEFAULTS
        else:
            default_settings = configs.GEMINI_DEFAULTS
        return GeminiPro(
            api_key=api_key, name=model_name, default_settings=default_settings
        )
    elif model_name in CLAUDE_MODELS:
        return ClaudeSonnet(api_key=api_key, name=model_name)
    elif model_name in TOGETHER_MODELS:
        return Llama_Together(api_key=api_key, name=model_name)
    elif model_name in QWEN2_MODELS:
        return Qwen2VL(address=address, name=model_name)
    else:
        raise ValueError(f"Unsupported model name '{model_name}'")
