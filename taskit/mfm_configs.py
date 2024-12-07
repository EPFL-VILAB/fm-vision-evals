O4_DEFAULTS = {
    'classify': {
        'prompt_no': 5,
    },
    'detect': {
        'prompt_no': 6,
        'n_iters': 6
    },
    'detect_naive': {
        'prompt_no': 5,
    },
    'segment': {
        'prompt_no': 2,
        'shape': 'rectangle'
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'segment_naive': {
        'prompt_no': 5,
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 4,
        'shape': 'curve'
    },
    'normals': {
        'prompt_no': 4,
        'shape': 'rectangle'
    }
}


GEMINI_DEFAULTS = {
    'classify': {
        'prompt_no': 1
    },
    'detect': {
        'prompt_no': 6,
        'n_iters': 8,
        'independent': True
    },
    'detect_naive': {
        'prompt_no': 5,
        'normalization': 1000
    },
    'segment': {
        'prompt_no': 1,
        'shape': 'point'
    },
    'segment_naive': {
        'prompt_no': 1,
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 5,
        'shape': 'point'
    },
    'normals': {
        'prompt_no': 1,
        'shape': 'curve'
    }
}


CLAUDE_DEFAULTS = {
    'classify': {
        'prompt_no': 5
    },
    'detect': {
        'prompt_no': 3,
        'n_iters': 7,
        'classification_type': 'classify_mult'
    },
    'detect_naive': {
        'prompt_no': 1,
        'classification_type': 'classify_mult',
    },
    'segment': {
        'prompt_no': 2,
        'shape': 'point'
    },
    'segment_naive': {
        'prompt_no': 1,
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 2,
        'shape': 'rectangle',
        'n_threads': 10
    },
    'normals': {
        'prompt_no': 3,
        'shape': 'rectangle'
    }
}


LLAMA_DEFAULTS = {
    'classify': {
        'prompt_no': 3,
    },
    'detect': {
        'prompt_no': 7,
        'n_iters': 8,
        'classification_type': 'classify_mult',
        'independent': True,
        'no_context': True,
        'mark_rectangle': True
    },
    'segment_sans_context': {
        'prompt_no': 1,
        'shape': 'rectangle',
        'n_segments': 400,
    },
    'group_sans_context': {
        'prompt_no': 1,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 3,
        'shape': 'point',
    },
    'normals': {
        'prompt_no': 4,
        'shape': 'rectangle',
        'n_threads': 10,
    }
}

QWEN2_DEFAULTS = {
    "address": "http://localhost:8000/",
    'classify': {
        'prompt_no': 3,
    },
    'detect': {
        'prompt_no': 6,
        'n_iters': 6
    },
    'detect_naive': {
        'prompt_no': 5,
    },
    'segment': {
        'prompt_no': 1,
        'shape': 'rectangle'
    },
    'group': {
        'prompt_no': 2,
        'shape': 'curve'
    },
    'depth': {
        'prompt_no': 4,
        'shape': 'curve'
    },
    'normals': {
        'prompt_no': 4,
        'shape': 'rectangle'
    }
}