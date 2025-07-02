import re
import time
import argparse
import json

import ray
from ray import serve

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Initialize argument parser
parser = argparse.ArgumentParser(description="Serve a model using Ray and GPUs")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-72B-Instruct", help="Name of the model to load")
parser.add_argument("--address", type=str, default=None, help="Address for Ray to use")
parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
parser.add_argument("--num_cpus", type=int, default=120, help="Number of CPUs to use")
parser.add_argument("--num_replicas", type=int, default=8, help="Number of replicas to use")
parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model")
args = parser.parse_args()

# Initialize Ray and set GPU resource allocation
ray.init(address=args.address)

# Ensure Ray Serve is started
serve.start()

# Define your PyTorch model class
class QwenWrapper:
    def __init__(self, model_name, cache_dir=None):
        self.model_name = model_name
        self.model, self.processor = self.load_modules(cache_dir)

    def load_modules(self, cache_dir):
        common_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2"
        }

        if cache_dir is not None:
            common_kwargs["cache_dir"] = cache_dir

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **common_kwargs
        )
        min_pixels = 64*28*28
        max_pixels = 128*28*28
        processor = AutoProcessor.from_pretrained(
            self.model_name, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        return model, processor

    def preprocess_message(self, messages):
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs
    
    def attach_assistant_message(self, outputs, messages):
        assitant_message = next(iter([m["content"] for m in messages if m["role"] == "assistant"]), None)
        
        if assitant_message is not None:
            outputs = [assitant_message + output for output in outputs]
        
        return outputs

    def forward(self, messages, max_new_tokens=256):
        print("Preprocessing messages")
        inputs = self.preprocess_message(messages).to(self.model.device)
        print("Generating outputs")
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        print("Postprocessing outputs")
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = self.attach_assistant_message(output_text, messages)
        return output_text

assert args.num_gpus * 2 % args.num_replicas == 0, "Double the number of GPUs must be divisible by number of replicas"
gpu_per_replica = args.num_gpus / args.num_replicas
cpu_per_replica = args.num_cpus / args.num_replicas

# Define a deployment class that will handle requests
@serve.deployment(num_replicas=args.num_replicas, ray_actor_options={"num_gpus": gpu_per_replica, "num_cpus": cpu_per_replica})
class QwenDeployment:
    def __init__(self, model_name, cache_dir=None):
        # Instantiate the model here
        self.model = QwenWrapper(model_name, cache_dir=cache_dir)

    async def __call__(self, request):
        # Process incoming requests
        data = await request.json()
        if "messages" in data:
            max_new_tokens = data.get("max_new_tokens", 256)
            outputs = self.model.forward(data["messages"], max_new_tokens)
            if data["output_format"] == "json":
                return {"output": [json.loads(
                                        re.sub(r"[\n\t]+", "", o)
                                    ) for o in outputs]}
            else:
                return {"output": outputs}
        elif "status" in data:
            return {"status": 1}


# Deploy the model across all available GPUs
serve.run(QwenDeployment.bind(args.model_name, args.cache_dir), route_prefix="/")

print(f"Model {args.model_name} deployed on {args.num_gpus} GPUs")
print("Ray Serve is running and serving the model. Waiting for requests...")
try:
    while True:
        time.sleep(3600)  # Sleep for 1 hour to avoid constant CPU use

except KeyboardInterrupt:
    print("Shutting down...")
    serve.shutdown()
except Exception as e:
    print(f"An error occurred: {e}")
    serve.shutdown()
