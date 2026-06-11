from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import (
    build_vocab,
    compute_metrics,
    default_stopwords,
    encode_tokens,
    ensure_dir,
    load_all_splits,
    prepare_model_frame,
    require_packages,
    save_confusion_matrix_png,
    save_json,
    select_device,
    set_seed,
    tokenize_chars,
    write_classification_report,
)
from bilstm_model import BiLSTMClassifier


class SentimentDataset:
    def __init__(self, texts, labels, vocab, max_len, tokenizer_name: str):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, outer):
                self.outer = outer

            def __len__(self):
                return len(self.outer.labels)

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(self.outer.encoded[idx], dtype=torch.long),
                    "label": torch.tensor(int(self.outer.labels[idx]), dtype=torch.long),
                }

        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer_name = tokenizer_name
        self.encoded = [encode_tokens(tokenize_for_lstm(text, tokenizer_name), vocab, max_len) for text in self.texts]
        self.dataset = _Dataset(self)


def tokenize_for_lstm(text: str, tokenizer_name: str) -> list[str]:
    if tokenizer_name == "char":
        return tokenize_chars(text)
    if tokenizer_name == "jieba":
        from common import clean_text_for_tfidf

        return clean_text_for_tfidf(text, default_stopwords()).split()
    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")


def subset(df, n: int | None, seed: int):
    if not n or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def evaluate(model, loader, device, criterion=None):
    import numpy as np
    import torch

    model.eval()
    total_loss = 0.0
    total_rows = 0
    labels_all = []
    preds_all = []
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            total_rows += labels.size(0)
            labels_all.extend(labels.cpu().numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
    metrics = compute_metrics(np.array(labels_all), np.array(preds_all), np.array(probs_all))
    if criterion is not None and total_rows:
        metrics["Loss"] = total_loss / total_rows
    return metrics, labels_all, preds_all, probs_all


def save_training_curve(log_rows, path: Path) -> None:
    import matplotlib.pyplot as plt

    if not log_rows:
        return
    epochs = [row["epoch"] for row in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in log_rows], label="train")
    axes[0].plot(epochs, [row["dev_loss"] for row in log_rows], label="dev")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["dev_f1"] for row in log_rows], marker="o")
    axes[1].set_title("Dev F1")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch BiLSTM sentiment classifier.")
    parser.add_argument("--data_dir", default="shopping_comments")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=60000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tokenizer", choices=["char", "jieba"], default="char")
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    deps = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "matplotlib": "matplotlib",
    }
    if args.tokenizer == "jieba":
        deps["jieba"] = "jieba"
    require_packages(deps)

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / "bilstm")

    frames = {split: prepare_model_frame(df) for split, df in load_all_splits(args.data_dir).items()}
    if args.fast_dev_run:
        frames["train"] = subset(frames["train"], 1000, args.seed)
        frames["dev"] = subset(frames["dev"], 400, args.seed)
        frames["test"] = subset(frames["test"], 400, args.seed)
        args.epochs = min(args.epochs, 1)

    train_tokens = [tokenize_for_lstm(text, args.tokenizer) for text in frames["train"]["text_a"].tolist()]
    vocab = build_vocab(train_tokens, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    print(f"Device: {device}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Data sizes: train={len(frames['train'])}, dev={len(frames['dev'])}, test={len(frames['test'])}")

    train_dataset = SentimentDataset(frames["train"]["text_a"], frames["train"]["label"], vocab, args.max_len, args.tokenizer).dataset
    dev_dataset = SentimentDataset(frames["dev"]["text_a"], frames["dev"]["label"], vocab, args.max_len, args.tokenizer).dataset
    test_dataset = SentimentDataset(frames["test"]["text_a"], frames["test"]["label"], vocab, args.max_len, args.tokenizer).dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dev_f1 = -1.0
    best_epoch = 0
    wait = 0
    log_rows = []
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_rows += labels.size(0)

        train_loss = total_loss / max(1, total_rows)
        dev_metrics, _, _, _ = evaluate(model, dev_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics.get("Loss", 0.0),
            "dev_f1": dev_metrics["F1-Score"],
            "dev_accuracy": dev_metrics["Accuracy"],
            "dev_mcc": dev_metrics["MCC"],
        }
        log_rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} dev_loss={row['dev_loss']:.4f} "
            f"dev_f1={row['dev_f1']:.4f}"
        )

        if dev_metrics["F1-Score"] > best_dev_f1:
            best_dev_f1 = dev_metrics["F1-Score"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "model_args": {
                        "vocab_size": len(vocab),
                        "embed_dim": args.embed_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    },
                    "max_len": args.max_len,
                    "tokenizer": args.tokenizer,
                    "best_dev_metrics": dev_metrics,
                },
                best_path,
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    with (output_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "dev_loss", "dev_f1", "dev_accuracy", "dev_mcc"])
        writer.writeheader()
        writer.writerows(log_rows)
    save_training_curve(log_rows, output_dir / "training_curve.png")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, y_true, y_pred, _ = evaluate(model, test_loader, device, criterion)
    save_json(
        {
            "best_epoch": best_epoch,
            "best_dev_metrics": checkpoint["best_dev_metrics"],
            "test_metrics": test_metrics,
            "data_sizes": {split: int(len(df)) for split, df in frames.items()},
            "vocab_size": len(vocab),
            "device": str(device),
        },
        output_dir / "metrics.json",
    )
    write_classification_report(y_true, y_pred, output_dir / "classification_report.txt")
    save_confusion_matrix_png(y_true, y_pred, output_dir / "confusion_matrix.png", title="BiLSTM Confusion Matrix")

    print("\nFinal test metrics:")
    for name, value in test_metrics.items():
        print(f"{name:16s}: {value:.4f}")
    print(f"\nSaved: {best_path}")
    print(f"Saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

