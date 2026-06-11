from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bilstm_model import BiLSTMClassifier
from common import (
    default_stopwords,
    encode_tokens,
    require_packages,
    select_device,
    sigmoid_scores,
    tokenize_chars,
    transform_ml_texts,
)


def label_name(label: int) -> str:
    return "正向(1)" if int(label) == 1 else "负向(0)"


def read_texts(args) -> list[str]:
    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.text_file:
        path = Path(args.text_file)
        texts.extend([line.strip() for line in path.read_text(encoding="utf-8").splitlines()])
    if not texts:
        texts = [
            "这个商品质量很好，物流也很快。",
            "太差了，以后不会再买。",
            "",
            "好",
            "包装破损，客服处理很慢，但是商品本身还可以。",
        ]
    return texts


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return sigmoid_scores(model.decision_function(X))
    return model.predict(X)


def predict_ml(texts: list[str], model_path: Path):
    require_packages({"joblib": "joblib", "sklearn": "scikit-learn", "scipy": "scipy", "jieba": "jieba"})
    import joblib

    bundle = joblib.load(model_path)
    X = transform_ml_texts(texts, bundle)
    model = bundle["model"]
    preds = model.predict(X)
    scores = positive_scores(model, X)
    return [{"text": text, "label": int(pred), "positive_score": float(score)} for text, pred, score in zip(texts, preds, scores)]


def tokenize_for_checkpoint(text: str, tokenizer_name: str) -> list[str]:
    if tokenizer_name == "char":
        return tokenize_chars(text)
    if tokenizer_name == "jieba":
        from common import clean_text_for_tfidf

        return clean_text_for_tfidf(text, default_stopwords()).split()
    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")


def predict_bilstm(texts: list[str], model_path: Path, device_arg: str):
    require_packages({"torch": "torch", "numpy": "numpy"})
    import torch

    device = select_device(device_arg)
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint["vocab"]
    max_len = int(checkpoint["max_len"])
    tokenizer_name = checkpoint.get("tokenizer", "char")
    model = BiLSTMClassifier(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoded = [encode_tokens(tokenize_for_checkpoint(text, tokenizer_name), vocab, max_len) for text in texts]
    input_ids = torch.tensor(encoded, dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return [{"text": text, "label": int(pred), "positive_score": float(score)} for text, pred, score in zip(texts, preds, probs)]


def predict_transformer(texts: list[str], model_path: Path, device_arg: str, max_len: int):
    require_packages({"torch": "torch", "transformers": "transformers", "numpy": "numpy"})
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = select_device(device_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    encodings = tokenizer(texts, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
    encodings = {key: value.to(device) for key, value in encodings.items()}
    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return [{"text": text, "label": int(pred), "positive_score": float(score)} for text, pred, score in zip(texts, preds, probs)]


def default_model_path(model_type: str, output_dir: str) -> Path:
    base = Path(output_dir)
    if model_type == "ml":
        return base / "ml_baseline" / "model.joblib"
    if model_type == "bilstm":
        return base / "bilstm" / "best_model.pt"
    if model_type == "transformer":
        return base / "transformer" / "best_model"
    raise ValueError(model_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment with a trained ML, BiLSTM, or Transformer model.")
    parser.add_argument("--model_type", choices=["ml", "bilstm", "transformer"], default="ml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--text", action="append", help="Comment text. Can be passed multiple times.")
    parser.add_argument("--text_file", default=None, help="UTF-8 text file with one comment per line.")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    texts = read_texts(args)
    model_path = Path(args.model_path) if args.model_path else default_model_path(args.model_type, args.output_dir)
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    if args.model_type == "ml":
        rows = predict_ml(texts, model_path)
    elif args.model_type == "bilstm":
        rows = predict_bilstm(texts, model_path, args.device)
    else:
        rows = predict_transformer(texts, model_path, args.device, args.max_len)

    for row in rows:
        row["prediction"] = label_name(row["label"])
        print(f"[{row['prediction']}] p(positive)={row['positive_score']:.4f} | {row['text']}")

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label", "prediction", "positive_score"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

