from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Dict

from PIL import Image

from openai import OpenAI
from google.generativeai import genai
import anthropic


class TaskRegistryABCMeta(ABCMeta, type):
    
    def __new__(cls, name, bases, attrs):
        # Create the new class
        new_class = super().__new__(cls, name, bases, attrs)
        # Initialize a class-specific TASK_REGISTERY for the new class
        for base in bases:
            if hasattr(base, 'TASK_REGISTERY'):
                new_class.TASK_REGISTERY = deepcopy(base.TASK_REGISTERY)
                return new_class
        
        new_class.TASK_REGISTERY = {}
        
        return new_class


class MFMWrapper(metaclass=TaskRegistryABCMeta):
    
    @classmethod
    def register_task(cls, task_name):
        def decorator(func):
            if cls is MFMWrapper:
                # Register the function in MFMWrapper's registry
                cls.TASK_REGISTERY[task_name] = func
                # Also register it in all subclasses
                for subclass in cls.__subclasses__():
                    subclass.TASK_REGISTERY[task_name] = func
            else:
                # If it's a subclass, only register in that subclass's registry
                cls.TASK_REGISTERY[task_name] = func
            return func
        return decorator

    def predict(self, task, image: Image, **kwargs):
        if task in self.TASK_REGISTERY:
            task_func = self.TASK_REGISTERY[task]
        else:
            raise ValueError(f"Task {task} not supported, please choose from {self.TASK_REGISTERY.keys()}")
        return task_func(model=self, image=image, **kwargs)
                                 
    def send_message(self, message: Dict):
        raise NotImplementedError


class GPT4o(MFMWrapper):
    
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def send_message(self, message: Dict):
        raise NotImplementedError
    
    def predict(self, task, image: Image, **kwargs):
        return super().predict(task, image, **kwargs)
    

class ClaudeSonnet(MFMWrapper):
    
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def send_message(self, message: Dict):
        raise NotImplementedError
    
    def predict(self, task, image: Image, **kwargs):
        return super().predict(task, image, **kwargs)
    

class GeminiPro(MFMWrapper):
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.client = genai
        
    def send_message(self, message: Dict):
        raise NotImplementedError
    
    def predict(self, task, image: Image, **kwargs):
        return super().predict(task, image, **kwargs)