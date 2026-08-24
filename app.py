from pathlib import Path

import gradio as gr
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "fine_tuned_model"

MAX_LENGTH = 128

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)

model = model.to(device)
model.eval()


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1,
    )[0]

    negative_probability = float(
        probabilities[0].cpu()
    )

    positive_probability = float(
        probabilities[1].cpu()
    )

    return {
        "Negative": negative_probability,
        "Positive": positive_probability,
    }


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        label="Text",
        placeholder="Enter a movie review...",
    ),
    outputs=gr.Label(
        label="Sentiment",
        num_top_classes=2,
    ),
    title="Sentiment Analysis with DistilBERT",
    description=(
        "Enter an English text and the fine-tuned model "
        "will classify its sentiment as Negative or Positive."
    ),
)


if __name__ == "__main__":
    demo.launch()