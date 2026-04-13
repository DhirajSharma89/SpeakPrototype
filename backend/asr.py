import re
import torch


class ASRModuleHF:
    def __init__(self, model, processor, device="cuda"):
        self.model = model.to(device)
        self.processor = processor
        self.device = device

    def _normalize(self, text):
        text = text.lower().strip()

        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # collapse repeated words
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

        # collapse repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1', text)

        return text

    def _tokenize(self, text):
        return text.split()

    # Main Inference
    def transcribe(self, audio_array):
        """
        Input:
            audio_array (np.ndarray): float32 audio at 16kHz

        Output:
            dict:
                {
                    "transcript": str,
                    "raw_transcript": str,
                    "tokens": list[str],
                    "confidence": None,
                    "language": "it"
                }
        """

        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs["input_features"].to(self.device)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,

                max_new_tokens=50,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,

                language="it",
                task="transcribe"
            )

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        clean_text = self._normalize(transcription)

        return {
            "transcript": clean_text,
            "raw_transcript": transcription,
            "tokens": self._tokenize(clean_text),
            "confidence": None,
            "language": "it"
        }