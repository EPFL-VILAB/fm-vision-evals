# Running Qwen Backend for Vision Tasks

Qwen2-VL is an open-source model, which means that to evaluate it alongside other closed-source models like GPT-4o, you will need to host the Qwen2-VL model on your infrastructure or locally on a backend. This guide walks you through the process of setting up and running the Qwen2-VL backend for vision tasks using the provided framework. Before running the scripts and using the `Qwen2VL` model in the `mfm.py`, ensure the backend is started.

## Steps to Run Qwen Backend

### Prerequisites
Before running the Qwen backend, ensure you have followed the installation instructions in the [README](README.md) file.

---

### Start the Qwen Backend
Use the following command to run the Qwen backend. Replace the placeholders with your desired values:

```bash
python scripts/run_qwen_backend.py \
    --num_gpus <NUM_GPUS> \
    --num_cpus <NUM_CPUS> \
    --num_replicas <NUM_REPLICAS> \
    --cache_dir <CACHE_DIR>
```

> **Note:**  For optimal performance, set `NUM_REPLICAS` to half the value of `NUM_GPUS`

Once the backend is running, it will automatically host the Qwen2-VL service on `http://localhost:8000/`. The `Qwen2VL` model by default calls this address for API requests. If you need to use a different address, you can configure the model as the following:
```python
model = Qwen2VL(address=NEW_ADDRESS)
```
Replace `NEW_ADDRESS` with the desired address where the backend is hosted.

---
This script uses the [Ray Serve library](https://docs.ray.io/en/latest/serve/index.html) to run and host the Qwen2-VL model. To host other offline models, you can follow a similar process using the Ray Serve library.