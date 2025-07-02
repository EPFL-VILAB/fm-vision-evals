# Documentation

## Table of contents
- [Overview](#overview)
- [MFMs](#mfms)
- [Tasks](#tasks)
- [Evaluation](#evaluation)

### Overview

The `taskit` package provides the necessarily tools and implementations for evaluating multimodal foundation models on different vision tasks and benchmarks. It is designed to be user-friendly and easy to use. The package currently supports 6 tasks, and is structured as follows:

```bash
taskit
├── mfm.py
├── tasks
│   ├── classify.py
│   ├── depth.py
│   ├── grouping.py
│   ├── normals.py
│   ├── object.py
│   └── segment.py
├── eval
│   ├── eval_classify.py
│   ├── eval_depth.py
│   ├── eval_grouping.py
│   ├── eval_normals.py
│   ├── eval_object.py
│   ├── eval_segment.py
│   └── eval_utils.py
└── utils
    ├── data_constants.py
    └── data.py
```

- **`mfm.py`**: Contains the class definition for the Multimodal Foundation Models (MFMs) used in the prompt chains.
- **`tasks/`**: Contains the task-specific prompts, and prompt chaining logic.
- **`eval/`**: Contains the evaluation scripts for each task. Also used to visualize the results for the various tasks
- **`utils/`**: Contains various utility functions and constants used across the package.

### MFMs

The `MFMWrapper` class manages the core functionality, and takes care of registering tasks, evaluation, and prediction. The MFMs subclass the `MFMWrapper` class. The package currently supports the following MFMs:

| Model | API Name | Link |
|-------|----------|------|
| GPT-4o | `gpt-4o-2024-08-06` | [Link](https://openai.com/index/hello-gpt-4o/) |
| o1 | `o1-2024-12-17` | [Link](https://openai.com/o1/) |
| o3 | `o3-2025-04-16` | [Link](https://openai.com/index/introducing-o3-and-o4-mini/) |
| o4-mini | `o4-mini-2025-04-16` | [Link](https://openai.com/index/introducing-o3-and-o4-mini/) |
| Gemini 2.0 Flash | `gemini-2.0-flash-001` | [Link](https://deepmind.google/technologies/gemini/flash/) |
| Gemini 1.5 Pro | `gemini-1.5-pro-001` | [Link](https://deepmind.google/technologies/gemini/pro/) |
| Claude 3.5 Sonnet | `claude-3-5-sonnet-20240620` | [Link](https://www.anthropic.com/news/claude-3-5-sonnet) |
| Qwen2-VL | `qwen2-vl-72b-instruct` | [Link](https://github.com/QwenLM/Qwen2-VL) |
| Llama 3.2 90B | `llama-3.2-90b` | [Link](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |

The individual MFM classes implement the following methods:
- `predict`: Passes the inputs and arguments to the corresponding task-specific prompt chain.
- `parse_message`: Converts the prompt chain output (a list of dictionaries with text and image components) from the default OpenAI format into the model-specific format required by each MFM class.
- `send_message`: Sends the parsed message to the respective MFM API and retrieves the response, using additional handling logic such as retries and prefilled prompts as needed.

In addition to this, each MFM class includes a `default_settings` attribute, which is a dictionary (found in `utils/data_constants.py`) containing default configurations for various tasks. These will be used if the user does not provide custom parameters.

To add a new MFM, subclass `MFMWrapper`, implement the required methods, and register the new class in `mfm.py`. This extensible structure makes it straightforward to integrate additional MFMs as they become available.

### Tasks
The prompt chains for the various tasks are defined in the `tasks/` directory. Each prompt chain is assembled using a system prompt, the other auxiliary user-defined prompts, and the sequence of images in the chain. An additional element of the prompt chain is the `json schema`, which is supported by some MFMs like GPT-4o. All the prompt chains are registered in the MFMs' task registry.

The prompt chain functions take the following arguments:
- `model`: The MFM model to use.
- `file_name`: This can either take in a single file path, a list of file paths, a PIL image, or a list of PIL images.
- `prompt` (optional): The user can pass in a custom text-prompt, to override the text portion of the prompt chain. 
- `prompt_no` (optional): The user can pass in a prompt number, to choose a specific prompt from the defined system prompts.
- Other task-specific arguments (optional): 

For example, this is the signature for the `segment` task:
```python
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
```

Many tasks have a `return_dict` argument, which is useful for scripts. If set to true, the output is returned as a dictionary, which can be used for evaluation/processing. If set to false, then the output is visualized and returned as an image. If a dictionary is returned, it will contain task-specific predictions, and the file name (this will be the temporary file name if the input was an image).

### Evaluation
The evaluation scripts for each task are located in the `eval/` directory. Each function takes `predictions` which is either a list of dictionaries (returned by the task functions with `return_dict=True`) or the path to a JSON file containing the list of dictionaries, under key `data`.

Many functions also take a `visualise` argument, which determines whether the metrics are returned or whether the predictions are visualized. 

For evaluating the metrics, each function expects a task-specific JSON file containing ground truth, defined in the `scripts/metadata` directory. The evaluation scripts calculate the metrics and return them as a dictionary.

- For classification, the JSON file maps each file name to the class label index.
- For object detection, the format is the same as COCO annotations.
- For segmentation, the file maps each file name to the ground truth segmentation mask (file path).
- For grouping, the file maps each file name to a dictionary similar to the dictionary returned by the grouping task function. This dictionary maps the instance indices to a dictionary containing the instance mask (under key 'gt').
- For depth, the file maps each file name to the ground truth depth map (file path).
- For normals, the file maps each file name to the ground truth surface normals (file path).

#### Scripts
We also provide the script used for inference and evaluation (`scripts/infer.py`). We include some example images in the `assets/` directory and the relevant ground truth JSON files in the `scripts/metadata/` so that users can test the inference scripts. The script can be run for various tasks as follows:

1. For **classification**:
```bash
python scripts/infer.py --model gpt-4o-2024-08-06 --api_key <your-api-key> --task classify --eval_type eval_classify --batch_size 1 --n_threads 1 --data_files scripts/data/files/classify.txt --ground_truth scripts/metadata/classify.json --log_name test_classify
```

2. For **object detection**:
```bash
python scripts/infer.py --model gpt-4o-2024-08-06 --api_key <your-api-key> --task detect --eval_type eval_detect --n_threads 2 --data_files scripts/data/files/object.txt --ground_truth scripts/metadata/object.json --log_name test_detect
```

3. For **segmentation**:
```bash
python scripts/infer.py --model gpt-4o-2024-08-06 --api_key <your-api-key> --task segment --eval_type eval_segment --n_threads 2 --data_files scripts/data/files/segment.txt --ground_truth scripts/metadata/segment.json --log_name test_segment
```

4. For **grouping**:
```bash
python scripts/infer.py --model gpt-4o-2024-08-06 --api_key <your-api-key> --task group --eval_type eval_group --n_threads 2 --data_files scripts/data/files/grouping.txt --ground_truth scripts/metadata/grouping.json --log_name test_group
```

5. For **depth**:
```bash
python scripts/infer.py --model gpt-4o-2024-08-06 --api_key <your-api-key> --task depth --eval_type eval_depth --n_threads 2 --data_files scripts/data/files/depth.txt --ground_truth scripts/metadata/depth.json --log_name test_depth
```

6. For **normals**:
```bash
python scripts/infer.py --model gpt-4o-2024-08-06 --api_key <your-api-key> --task normals --eval_type eval_normals --n_threads 2 --data_files scripts/data/files/normals.txt --ground_truth scripts/metadata/normals.json --log_name test_normals
```

The respective log files will be saved in the `scripts/data/logs/outputs` directory. The log files will contain the tokens used, model predictions, evaluation metrics. To visualize the predictions, a procedure similar to the one demonstrated in the notebook can be followed, as the `eval` method can take the path to the log file as an argument. A backup of the log files is also saved in the `scripts/data/logs/backups` directory.

#### Image Generation
Recent MFMs, such as the updated GPT-4o, support image generation. The `taskit` package provides a preliminary implementation for generating dense predictions for `grouping`, `depth`, and `normals` tasks. The `gpt-image-1` model is used for this purpose. The various tasks can be run using the `scripts/infer.py` script, as below:

1. For **grouping**:
```bash
python scripts/infer.py --model gpt-image-1 --api_key <your-api-key> --task dense_group --eval_type eval_dense_group --n_threads 2 --data_files scripts/data/files/grouping.txt --ground_truth scripts/metadata/grouping.json --log_name test_dense_group
```

2. For **depth**:
```bash
python scripts/infer.py --model gpt-image-1 --api_key <your-api-key> --task dense_depth --eval_type eval_dense_depth --n_threads 2 --data_files scripts/data/files/depth.txt --ground_truth scripts/metadata/depth.json --log_name test_dense_depth
```

3. For **normals**:
```bash
python scripts/infer.py --model gpt-image-1 --api_key <your-api-key> --task dense_normals --eval_type eval_dense_normals --n_threads 2 --data_files scripts/data/files/normals.txt --ground_truth scripts/metadata/normals.json --log_name test_dense_normals
```

The generated images will be saved in the `4o-imagegen/<task>-preds` directory.

#### Benchmark
We include the files of various datasets used in the benchmark in the `scripts/data/files/dataset-files` directory. The ground truth is present in `scripts/metadata/benchmark-gt`. Note that we do not provide the actual images or the ground truth images, and they will have to be downloaded separately. 