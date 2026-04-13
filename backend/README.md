# SpeakPrototype Backend

A complete pipeline for Italian speech recognition and command mapping. The backend combines automatic speech recognition (ASR) with a hybrid machine learning approach to transcribe audio and map it to actionable commands.

## Architecture Overview

The system is built in two stages:

```
Audio Input (16kHz) 
    ↓
[ASR Module] - Whisper + LoRA fine-tuning
    ↓
Transcript (Italian)
    ↓
[Mapper Module] - Hybrid (Rule-based + ML)
    ↓
Command + Confidence
```

## Components

### 1. **ASRModuleHF** (`asr.py`)

Handles automatic speech recognition using OpenAI's Whisper model with LoRA fine-tuning.

**Key Features:**
- Uses `openai/whisper-medium` as base model
- LoRA adapter for Italian language optimization
- Generates predictions with optimized beam search parameters
- Text normalization including punctuation removal and character deduplication

**Input:**
- Audio array (numpy ndarray, float32, 16kHz)

**Output:**
```python
{
    "transcript": str,           # Cleaned, normalized text
    "raw_transcript": str,       # Raw model output
    "tokens": list[str],         # Tokenized transcript
    "confidence": None,
    "language": "it"
}
```

**Decoding Parameters:**
- `max_new_tokens`: 50
- `num_beams`: 5
- `repetition_penalty`: 1.2
- `no_repeat_ngram_size`: 3

### 2. **HybridMapper** (`mapper.py`)

Maps transcribed text to predefined commands using a hybrid approach.

**Modes:**

1. **Rule-based Mode** (for short text ≤3 tokens):
   - Uses Levenshtein edit distance similarity
   - Default similarity threshold: 0.6
   - Includes numeric shortcuts mapping (e.g., "5" → "cinque")

2. **ML Model Mode** (for longer text):
   - Uses pre-trained sklearn classifier
   - Computes probability distribution over command classes
   - Applies confidence and margin thresholds to validate predictions

**Input:**
- Text string (will be normalized)

**Output:**
```python
{
    "status": "success|reject",     # Accept or reject prediction
    "prediction": str,              # The mapped command (or None)
    "confidence": float,            # Confidence score [0, 1]
    "mode": "rule|model"            # Which strategy was used
}
```

**Configurable Thresholds:**
- `short_text_threshold`: 3 tokens (switches between rule/model mode)
- `similarity_threshold`: 0.6 (rule-based acceptance)
- `confidence_threshold`: 0.6 (ML model acceptance)
- `margin_threshold`: 0.15 (margin between top predictions)

### 3. **SpeakPipeline** (`pipeline.py`)

Main orchestrator combining ASR and Mapper.

**Initialization:**
```python
from backend.pipeline import SpeakPipeline

pipeline = SpeakPipeline(
    device="cuda",                              # or "cpu"
    whisper_model_name="openai/whisper-medium",
    lora_path="models/checkpoint-1488",
    mapper_model_path="models/artifacts/mapper_model.joblib",
    label_encoder_path="models/artifacts/label_encoder.joblib"
)
```

**Main Method:**
```python
result = pipeline.predict(audio_array)
```

**Output:**
```python
{
    "transcript": str,           # Cleaned transcript
    "raw_transcript": str,       # Raw ASR output
    "tokens": list[str],         # Tokenized words
    "command": str,              # Mapped command
    "status": str,               # success|reject
    "confidence": float,         # Command mapping confidence
    "mode": str                  # rule|model
}
```

### 4. **MVPDatasetLoader** (`mvp_dataset_loader.py`)

Handles dataset loading and train/test splitting.

**Usage:**
```python
from backend.mvp_dataset_loader import MVPDatasetLoader

loader = MVPDatasetLoader(
    dataset_path="dataset/easycall_mvp_dataset",
    split_ratio=0.7,
    seed=42
)

data = loader.load()
# Returns: {"dataset", "train", "test", "commands"}
```



## Model Artifacts

Place the following pre-trained models in the `models/` directory:

```
models/
├── checkpoint-1488/           # LoRA fine-tuned Whisper adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ... (tokenizer files)
└── artifacts/
    ├── mapper_model.joblib    # Sklearn classifier for command mapping
    └── label_encoder.joblib   # Label encoder for commands
```

## Usage Example

```python
import numpy as np
from backend.pipeline import SpeakPipeline

# Initialize pipeline
pipeline = SpeakPipeline(device="cuda")

# Generate or load 16kHz Italian audio as numpy float32 array
audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio

# Get prediction
result = pipeline.predict(audio)

print(f"Transcript: {result['transcript']}")
print(f"Command: {result['command']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Status: {result['status']}")
```

## Performance Notes

- **Device**: Optimized for GPU (CUDA)
- **Latency**: Depends on audio length and device
- **Memory**: Requires ~4-7GB GPU memory for Whisper + adapters
- **Language**: Trained for Italian (Italian language code specified as "it")

## Error Handling

```python
try:
    result = pipeline.predict(audio_array)
except ValueError as e:
    print(f"Invalid input: {e}")
    # audio_array must be numpy.ndarray of float32
except RuntimeError as e:
    print(f"Model error: {e}")
    # GPU out of memory or model loading issues
```


