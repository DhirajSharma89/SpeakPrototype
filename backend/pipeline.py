import numpy as np
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

from backend.asr import ASRModuleHF
from backend.mapper import HybridMapper


class SpeakPipeline:
    def __init__(
        self,
        device="cuda",
        whisper_model_name="openai/whisper-medium",
        lora_path="models/checkpoint-1488",
        mapper_model_path="models/artifacts/mapper_model.joblib",
        label_encoder_path="models/artifacts/label_encoder.joblib"
    ):
        self.device = device

        # Load ASR
        self.processor = WhisperProcessor.from_pretrained(
            whisper_model_name,
            language="italian",
            task="transcribe"
        )

        base_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_model_name
        ).to(device)

        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()

        self.asr = ASRModuleHF(model, self.processor, device=device)

        # Load Mapper
        self.mapper = HybridMapper(
            model_path=mapper_model_path,
            label_encoder_path=label_encoder_path
        )

    # Main Inference
    def predict(self, audio_array):
        """
        Input:
            audio_array (np.ndarray)

        Output:
            dict
        """

        if not isinstance(audio_array, np.ndarray):
            raise ValueError("audio_array must be a numpy array")

        audio_array = audio_array.astype(np.float32)

        # ASR
        asr_output = self.asr.transcribe(audio_array)

        # Mapper
        mapping_output = self.mapper.predict(asr_output["transcript"])

        return {
            "transcript": asr_output["transcript"],
            "raw_transcript": asr_output["raw_transcript"],
            "tokens": asr_output["tokens"],

            "command": mapping_output.get("prediction"),
            "status": mapping_output["status"],
            "confidence": mapping_output["confidence"],
            "mode": mapping_output.get("mode")
        }