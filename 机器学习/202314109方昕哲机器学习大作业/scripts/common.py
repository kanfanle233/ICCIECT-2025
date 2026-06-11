from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable


LABEL_NAMES = {0: "negative", 1: "positive"}


def require_packages(packages: dict[str, str]) -> None:
    """Fail early with a clear install hint when optional ML packages are missing."""
    missing = [pip_name for import_name, pip_name in packages.items() if importlib.util.find_spec(import_name) is None]
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise SystemExit(
            f"Missing dependencies: {names}\n"
            "Install them with: python -m pip install -r requirements.txt"
        )


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def select_device(device_arg: str = "auto"):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(data_dir: str | Path, split: str):
    import pandas as pd

    path = Path(data_dir) / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Data split not found: {path}")

    df = pd.read_csv(path, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE)
    if "label" not in df.columns or "text_a" not in df.columns:
        df = pd.read_csv(
            path,
            sep="\t",
            names=["label", "text_a"],
            encoding="utf-8",
            quoting=csv.QUOTE_NONE,
            skiprows=1,
        )
    df = df[["label", "text_a"]].copy()
    df["split"] = split
    return df


def load_all_splits(data_dir: str | Path) -> dict[str, Any]:
    return {split: load_split(data_dir, split) for split in ("train", "dev", "test")}


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def cleaned_char_count(text: Any) -> int:
    text = normalize_text(text)
    return len(re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", text))


def prepare_model_frame(df):
    import pandas as pd

    out = df.copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out["text_a"] = out["text_a"].map(normalize_text)
    out["clean_len"] = out["text_a"].map(cleaned_char_count)
    out = out[out["label"].isin([0, 1]) & (out["text_a"] != "") & (out["clean_len"] > 0)].copy()
    out["label"] = out["label"].astype(int)
    out = out.reset_index(drop=True)
    return out


def default_stopwords() -> set[str]:
    return {
        "的",
        "了",
        "是",
        "我",
        "也",
        "就",
        "都",
        "而",
        "及",
        "与",
        "着",
        "或",
        "一个",
        "没有",
        "我们",
        "你们",
        "他们",
        "这个",
        "那个",
        "还是",
        "就是",
    }


def clean_text_for_tfidf(text: Any, stopwords: set[str] | None = None) -> str:
    require_packages({"jieba": "jieba"})
    import jieba

    stopwords = stopwords or default_stopwords()
    text = normalize_text(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\u4e00-\u9fffa-z0-9]", " ", text)
    tokens = [word for word in jieba.lcut(text) if word not in stopwords and len(word.strip()) > 1]
    return " ".join(tokens)


def transform_ml_texts(texts: list[str], bundle: dict[str, Any]):
    require_packages({"scipy": "scipy", "jieba": "jieba"})
    from scipy.sparse import hstack

    stopwords = set(bundle.get("stopwords") or default_stopwords())
    cleaned = [clean_text_for_tfidf(text, stopwords) for text in texts]
    feature_name = bundle["feature_name"]
    word_vectorizer = bundle.get("word_vectorizer")
    char_vectorizer = bundle.get("char_vectorizer")

    if feature_name == "tfidf_word":
        return word_vectorizer.transform(cleaned)
    if feature_name == "char_ngram":
        return char_vectorizer.transform(cleaned)
    if feature_name == "combined":
        return hstack([word_vectorizer.transform(cleaned), char_vectorizer.transform(cleaned)])
    if feature_name == "chi2":
        return bundle["chi2_selector"].transform(word_vectorizer.transform(cleaned))
    raise ValueError(f"Unsupported feature set: {feature_name}")


def tokenize_chars(text: Any) -> list[str]:
    text = normalize_text(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\u4e00-\u9fffa-z0-9]", " ", text)
    return [ch for ch in text if not ch.isspace()]


def build_vocab(token_lists: Iterable[list[str]], min_freq: int = 2, max_vocab_size: int | None = None) -> dict[str, int]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    kept = [(tok, freq) for tok, freq in counter.most_common() if freq >= min_freq]
    if max_vocab_size:
        kept = kept[: max(0, max_vocab_size - len(vocab))]
    for token, _ in kept:
        vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


def sigmoid_scores(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def compute_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "NPV": npv,
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["ROC-AUC"] = float("nan")
        try:
            metrics["Log Loss"] = log_loss(y_true, y_score, labels=[0, 1])
        except Exception:
            metrics["Log Loss"] = float("nan")
    return {k: float(v) for k, v in metrics.items()}


def write_classification_report(y_true, y_pred, path: str | Path) -> None:
    from sklearn.metrics import classification_report

    path = Path(path)
    ensure_dir(path.parent)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["negative(0)", "positive(1)"],
        digits=4,
        zero_division=0,
    )
    path.write_text(report, encoding="utf-8")


def save_confusion_matrix_png(y_true, y_pred, path: str | Path, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    path = Path(path)
    ensure_dir(path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["negative(0)", "positive(1)"],
        yticklabels=["negative(0)", "positive(1)"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
