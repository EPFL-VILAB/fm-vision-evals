GPT4O_DEFAULTS = {
    "classify": {
        "task_specific_args": {"prompt_no": 5},
    },
    "detect": {
        "task_specific_args": {"prompt_no": 6, "n_iters": 6},
    },
    "detect_naive": {
        "task_specific_args": {"prompt_no": 5},
    },
    "segment": {
        "task_specific_args": {"prompt_no": 2, "shape": "rectangle"},
    },
    "segment_sans_context": {
        "task_specific_args": {
            "prompt_no": 1,
            "shape": "rectangle",
            "n_segments": 400,
        },
    },
    "segment_naive": {
        "task_specific_args": {"prompt_no": 5},
    },
    "group": {
        "task_specific_args": {"prompt_no": 2, "shape": "curve"},
    },
    "depth": {
        "task_specific_args": {"prompt_no": 4, "shape": "curve"},
        "eval_specific_args": {"smoothness_weight": 10},
    },
    "normals": {
        "task_specific_args": {"prompt_no": 4, "shape": "rectangle"},
        "eval_specific_args": {"smoothness_weight": 10},
    },
}

GPT_IMAGE_DEFAULTS = {
    "dense_depth": {},
}

O1_DEFAULTS = {
    "classify": {
        "task_specific_args": {"prompt_no": 5},
    },
    "detect": {
        "task_specific_args": {"prompt_no": 6, "n_iters": 6},
    },
    "detect_naive": {
        "task_specific_args": {"prompt_no": 5},
    },
    "segment": {
        "task_specific_args": {"prompt_no": 6, "shape": "rectangle"},
    },
    "group": {
        "task_specific_args": {"prompt_no": 2, "shape": "curve"},
    },
    "depth": {
        "task_specific_args": {"prompt_no": 4, "shape": "curve"},
        "eval_specific_args": {"smoothness_weight": 10},
    },
    "normals": {
        "task_specific_args": {"prompt_no": 2, "shape": "rectangle"},
        "eval_specific_args": {"smoothness_weight": 10},
    },
}


GEMINI_DEFAULTS = {
    "classify": {
        "task_specific_args": {"prompt_no": 1},
    },
    "detect": {
        "task_specific_args": {"prompt_no": 6, "n_iters": 8, "independent": True},
    },
    "detect_naive": {
        "task_specific_args": {"prompt_no": 5, "normalization": 1000},
    },
    "segment": {
        "task_specific_args": {"prompt_no": 1, "shape": "point"},
    },
    "segment_naive": {
        "task_specific_args": {"prompt_no": 1},
    },
    "segment_sans_context": {
        "task_specific_args": {
            "prompt_no": 1,
            "shape": "rectangle",
            "n_segments": 400,
        },
    },
    "group": {
        "task_specific_args": {"prompt_no": 2, "shape": "curve"},
    },
    "depth": {
        "task_specific_args": {"prompt_no": 5, "shape": "point"},
        "eval_specific_args": {"smoothness_weight": 20},
    },
    "normals": {
        "task_specific_args": {"prompt_no": 1, "shape": "curve"},
        "eval_specific_args": {"smoothness_weight": 5},
    },
}

GEMINI_2_DEFAULTS = {
    "classify": {
        "task_specific_args": {"prompt_no": 1},
    },
    "detect": {
        "task_specific_args": {"prompt_no": 6, "n_iters": 8, "independent": True},
    },
    "detect_naive": {
        "task_specific_args": {"prompt_no": 5, "normalization": 1},
    },
    "segment": {
        "task_specific_args": {"prompt_no": 1, "shape": "point"},
    },
    "segment_naive": {
        "task_specific_args": {"prompt_no": 1},
    },
    "segment_sans_context": {
        "task_specific_args": {
            "prompt_no": 1,
            "shape": "rectangle",
            "n_segments": 400,
        },
    },
    "group": {
        "task_specific_args": {"prompt_no": 2, "shape": "curve"},
    },
    "depth": {
        "task_specific_args": {"prompt_no": 5, "shape": "point"},
    },
    "normals": {
        "task_specific_args": {"prompt_no": 1, "shape": "curve"},
    },
}


CLAUDE_DEFAULTS = {
    "classify": {
        "task_specific_args": {"prompt_no": 5},
    },
    "detect": {
        "task_specific_args": {
            "prompt_no": 3,
            "n_iters": 7,
            "classification_type": "classify_mult",
        },
    },
    "detect_naive": {
        "task_specific_args": {
            "prompt_no": 1,
            "classification_type": "classify_mult",
        },
    },
    "segment": {
        "task_specific_args": {"prompt_no": 2, "shape": "point"},
    },
    "segment_naive": {
        "task_specific_args": {
            "prompt_no": 1,
        },
    },
    "segment_sans_context": {
        "task_specific_args": {
            "prompt_no": 1,
            "shape": "rectangle",
            "n_segments": 400,
        },
    },
    "group": {
        "task_specific_args": {"prompt_no": 2, "shape": "curve"},
    },
    "depth": {
        "task_specific_args": {"prompt_no": 2, "shape": "rectangle", "n_threads": 10},
        "eval_specific_args": {"smoothness_weight": 10},
    },
    "normals": {
        "task_specific_args": {"prompt_no": 3, "shape": "rectangle"},
        "eval_specific_args": {"smoothness_weight": 5},
    },
}

LLAMA_DEFAULTS = {
    "classify": {
        "task_specific_args": {"prompt_no": 3},
    },
    "detect": {
        "task_specific_args": {
            "prompt_no": 7,
            "n_iters": 8,
            "classification_type": "classify_mult",
            "independent": True,
            "no_context": True,
            "mark_rectangle": True,
        },
    },
    "segment_sans_context": {
        "task_specific_args": {
            "prompt_no": 1,
            "shape": "rectangle",
            "n_segments": 400,
        },
    },
    "group_sans_context": {
        "task_specific_args": {"prompt_no": 1, "shape": "curve"},
    },
    "depth": {
        "task_specific_args": {"prompt_no": 3, "shape": "point"},
        "eval_specific_args": {"smoothness_weight": 20},
    },
    "normals": {
        "task_specific_args": {
            "prompt_no": 4,
            "shape": "rectangle",
            "n_threads": 10,
        },
        "eval_specific_args": {"smoothness_weight": 40},
    },
}

QWEN2_DEFAULTS = {
    "address": "http://localhost:8000/",
    "classify": {
        "task_specific_args": {
            "prompt_no": 3,
        },
    },
    "detect": {
        "task_specific_args": {"prompt_no": 6, "n_iters": 8},
    },
    "detect_naive": {
        "task_specific_args": {
            "prompt_no": 5,
        },
    },
    "segment": {
        "task_specific_args": {"prompt_no": 2, "shape": "rectangle", "n_segments": 128},
        "eval_specific_args": {"n_segments": 128},
    },
    "group": {
        "task_specific_args": {"prompt_no": 2, "shape": "curve"},
    },
    "depth": {"task_specific_args": {"prompt_no": 4, "shape": "curve"}},
    "normals": {"task_specific_args": {"prompt_no": 2, "shape": "rectangle"}},
}
