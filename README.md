# Emotion Analysis Pipeline

An image/video emotion analysis pipeline using the LLaVA-NeXT-Video model.

## Key Features

- Emotion analysis for images and videos
- Two analysis modes supported:
  1. Free-form emotion description (Version 1)
  2. Predefined emotion label classification (Version 2)
- Efficient video frame processing
- Detailed debug information
- Random seed setting for reproducibility

## Installation Requirements

```bash
pip install torch transformers pillow numpy
```

## Usage

```python
from src.pipeline.emotion_pipeline import EmotionPipeline

# Basic usage
pipeline = EmotionPipeline(model_path="path/to/model")

# With random seed for reproducibility
pipeline = EmotionPipeline(
    model_path="path/to/model",
    seed=42  # Set desired seed value
)

# Process image
result = pipeline.process({"image": image_array})

# Process video
result = pipeline.process({"video": video_frames})
```

## Random Seed Setting

Setting a random seed fixes the following elements:
- Model generation results
- Video frame sampling
- CUDA operations

Seeds set:
```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: Setting a fixed seed may slightly reduce performance.

## Video Frame Processing Strategy

Video frame sampling uses the following strategies:

1. Temporal Information Preservation:
   - Always includes start and end frames
   - Maintains temporal order for emotion change context

2. Motion-based Key Frame Selection:
   - Calculates differences between consecutive frames
   - Prioritizes frames with significant changes
   - Captures moments with significant facial expressions or gestures

3. Adaptive Sampling:
   - Uses 8 frames total
   - Pads with last frame if insufficient
   - Selects based on importance for longer videos

## Configuration Options

### Prompt Versions

1. Version 1 (Free Description):
   - Image: "Describe the emotional content of this scene. Focus on the visible emotions and their expressions."
   - Video: "What is the main emotion shown in this video clip?"

2. Version 2 (Label Classification):
   - Predefined emotion labels: HAPPY, SAD, ANGRY, SURPRISED, CONFUSED, NEUTRAL, FEARFUL, DISGUSTED

### Generation Settings

```python
generation_config = {
    "max_new_tokens": 128,
    "do_sample": True,
    "num_beams": 3,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.3,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 3,
    "min_length": 10
}
```

## Output Format

```python
{
    "id": "unique_id",
    "emotion": "Emotion category or OBSERVATION",
    "explanation": "Explanation of emotions",
    "is_emotion": True/False,
    "success": True/False,
    "debug_info": {
        "input_type": "image/video",
        "frame_count": number_of_frames,
        "token_count": number_of_tokens
    }
}
```

## Important Notes

1. Video Processing Memory Usage:
   - Optimized for 8 frames
   - Automatically selects key frames for longer videos

2. GPU Memory:
   - Uses 4-bit quantization
   - Uses float16 precision

3. Error Handling:
   - Graceful handling of all exceptions
   - Detailed error messages provided 