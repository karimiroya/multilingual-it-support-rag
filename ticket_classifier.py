# ============================================================
# MULTILINGUAL IT TICKET CLASSIFIER
# Fine-tuned XLM-RoBERTa
# ============================================================

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "final_xlmr_ticket_model"


def load_classifier():
    """
    Load the fine-tuned XLM-R classifier and tokenizer.
    """

    print("Loading XLM-R ticket classifier...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    )

    model.to("cpu")
    model.eval()

    print("XLM-R classifier loaded successfully!", flush=True)

    return tokenizer, model


def classify_ticket(text, tokenizer, model):
    """
    Predict the IT support category for an employee question.

    Returns:
        category: predicted IT category
        confidence: model's probability for that category
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predicted_id = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[0, predicted_id].item()

    category = model.config.id2label[predicted_id]

    return {
        "category": category,
        "confidence": round(confidence, 4)
    }


# ============================================================
# TEST CLASSIFIER
# ============================================================

if __name__ == "__main__":

    tokenizer, model = load_classifier()

    examples = [
        "My VPN connection fails with error 800.",
        "I cannot resolve the company website DNS address.",
        "La connessione Wi-Fi aziendale non funziona."
    ]

    for question in examples:

        result = classify_ticket(
            question,
            tokenizer,
            model
        )

        print("\nQuestion:", question)
        print("Predicted category:", result["category"])
        print("Confidence:", result["confidence"])