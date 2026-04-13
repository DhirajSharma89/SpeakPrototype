import re
import numpy as np
import joblib

# Utility Functions
def edit_similarity(a, b):
    dp = np.zeros((len(a) + 1, len(b) + 1))

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    dist = dp[len(a)][len(b)]
    return 1 - dist / max(len(a), len(b), 1)


def normalize_text(text):
    text = text.lower().strip()

    # collapse repeated characters
    text = re.sub(r"(.)\1+", r"\1", text)

    # map numeric shortcuts
    num_map = {
        "5": "cinque",
        "4": "quattro",
        "9": "nove"
    }

    if text in num_map:
        return num_map[text]

    return text


# Mapper Class
class HybridMapper:
    def __init__(
        self,
        model_path="models/artifacts/mapper_model.joblib",
        label_encoder_path="models/artifacts/label_encoder.joblib",
        short_text_threshold=3,
        similarity_threshold=0.6,
        confidence_threshold=0.6,
        margin_threshold=0.15
    ):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

        self.commands = list(self.label_encoder.classes_)

        self.short_text_threshold = short_text_threshold
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def predict(self, text):
        text = normalize_text(text)

        if len(text) <= self.short_text_threshold:
            best_cmd = None
            best_score = 0

            for cmd in self.commands:
                score = edit_similarity(text, cmd)
                if score > best_score:
                    best_score = score
                    best_cmd = cmd

            if best_score > self.similarity_threshold:
                return {
                    "status": "success",
                    "prediction": best_cmd,
                    "confidence": float(best_score),
                    "mode": "rule"
                }
            else:
                return {
                    "status": "reject",
                    "prediction": None,
                    "confidence": float(best_score),
                    "mode": "rule"
                }

        prob = self.model.predict_proba([text])[0]

        sorted_idx = np.argsort(prob)
        best = sorted_idx[-1]
        second = sorted_idx[-2]

        best_score = prob[best]
        margin = best_score - prob[second]

        pred = self.label_encoder.inverse_transform([best])[0]

        if best_score < self.confidence_threshold:
            return {
                "status": "reject",
                "prediction": None,
                "confidence": float(best_score),
                "mode": "ml"
            }

        if margin < self.margin_threshold:
            return {
                "status": "reject",
                "prediction": None,
                "confidence": float(best_score),
                "mode": "ml"
            }

        return {
            "status": "success",
            "prediction": pred,
            "confidence": float(best_score),
            "mode": "ml"
        }