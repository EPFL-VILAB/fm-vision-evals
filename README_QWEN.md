# Running Qwen Backend for Vision Tasks

This guide explains how to set up and run the Qwen2-VL backend for vision tasks using the provided framework. Before you can run the sampler or use the `Qwen2VL` model in the `mfm.py` script, you must first start the backend.

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