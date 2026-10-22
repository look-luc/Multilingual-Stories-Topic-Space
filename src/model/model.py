import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data.get_data import get_all_data, get_some_data


class LDA:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("NicolaiSivesind/LDA-Classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "NicolaiSivesind/LDA-Classifier",
            device_map=self.device
        )

class XLM_RoBerta:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            "FacebookAI/xlm-roberta-base"
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            "FacebookAI/xlm-roberta-base",
            device_map="auto",
            attn_implementation="sdpa"
        )
