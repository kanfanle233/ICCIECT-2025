from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import (
    compute_metrics,
    ensure_dir,
    load_all_splits,
    prepare_model_frame,
    require_packages,
    save_confusion_matrix_png,
    save_json,
    select_device,
    set_seed,
    write_classification_report,
)


class TransformerDataset:
    def __init__(self, texts, labels, tokenizer, max_len: int):
        import torch
        from torch.utils.data import Dataset

        encodings = tokenizer(
            list(texts),
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )

        class _Dataset(Dataset):
            def __init__(self, outer):
                self.outer = outer

            def __len__(self):
                return len(self.outer.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(value[idx], dtype=torch.long) for key, value in self.outer.encodings.items()}
                item["labels"] = torch.tensor(int(self.outer.labels[idx]), dtype=torch.long)
                return item

        self.encodings = encodings
        self.labels = list(labels)
        self.dataset = _Dataset(self)


def subset(df, n: int | None, seed: int):
    if not n or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def evaluate(model, loader, device):
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
            labels = batch["labels"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs, labels=labels)
            total_loss += float(outputs.loss.item()) * labels.size(0)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)
            total_rows += labels.size(0)
            labels_all.extend(labels.cpu().numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
    metrics = compute_metrics(np.array(labels_all), np.array(preds_all), np.array(probs_all))
    metrics["Loss"] = total_loss / max(1, total_rows)
    return metrics, labels_all, preds_all, probs_all


def load_hf_objects(args, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            local_files_only=args.local_files_only,
        ).to(device)
        return tokenizer, model
    except Exception as exc:
        raise SystemExit(
            "Could not load the Hugging Face model.\n"
            f"Model: {args.model_name}\n"
            f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'official')}\n"
            "Try one of these:\n"
            "  1. python scripts/03_finetune_transformer.py --model_name bert-base-chinese\n"
            "  2. pass a local cached model path with --model_name /path/to/model\n"
            "  3. install/download dependencies with network access first\n"
            f"Original error: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face Chinese Transformer sentiment classifier.")
    parser.add_argument("--data_dir", default="shopping_comments")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model_name", default="hfl/chinese-macbert-base")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--hf_endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    require_packages(
        {
            "pandas": "pandas",
            "numpy": "numpy",
            "sklearn": "scikit-learn",
            "torch": "torch",
            "transformers": "transformers",
            "matplotlib": "matplotlib",
        }
    )
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / "transformer")
    best_model_dir = ensure_dir(output_dir / "best_model")

    frames = {split: prepare_model_frame(df) for split, df in load_all_splits(args.data_dir).items()}
    if args.fast_dev_run:
        frames["train"] = subset(frames["train"], 128, args.seed)
        frames["dev"] = subset(frames["dev"], 64, args.seed)
        frames["test"] = subset(frames["test"], 64, args.seed)
        args.epochs = min(args.epochs, 1)

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Data sizes: train={len(frames['train'])}, dev={len(frames['dev'])}, test={len(frames['test'])}")

    tokenizer, model = load_hf_objects(args, device)
    train_dataset = TransformerDataset(frames["train"]["text_a"], frames["train"]["label"], tokenizer, args.max_len).dataset
    dev_dataset = TransformerDataset(frames["dev"]["text_a"], frames["dev"]["label"], tokenizer, args.max_len).dataset
    test_dataset = TransformerDataset(frames["test"]["text_a"], frames["test"]["label"], tokenizer, args.max_len).dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_dev_f1 = -1.0
    best_epoch = 0
    wait = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_rows += labels.size(0)

        train_loss = total_loss / max(1, total_rows)
        dev_metrics, _, _, _ = evaluate(model, dev_loader, device)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_metrics["Loss"],
                "dev_f1": dev_metrics["F1-Score"],
                "dev_accuracy": dev_metrics["Accuracy"],
                "dev_mcc": dev_metrics["MCC"],
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} dev_loss={dev_metrics['Loss']:.4f} "
            f"dev_f1={dev_metrics['F1-Score']:.4f}"
        )

        if dev_metrics["F1-Score"] > best_dev_f1:
            best_dev_f1 = dev_metrics["F1-Score"]
            best_epoch = epoch
            wait = 0
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            save_json(dev_metrics, output_dir / "best_dev_metrics.json")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    import csv

    with (output_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "dev_loss", "dev_f1", "dev_accuracy", "dev_mcc"])
        writer.writeheader()
        writer.writerows(log_rows)

    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    test_metrics, y_true, y_pred, _ = evaluate(model, test_loader, device)
    save_json(
        {
            "model_name": args.model_name,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "data_sizes": {split: int(len(df)) for split, df in frames.items()},
            "device": str(device),
        },
        output_dir / "metrics.json",
    )
    write_classification_report(y_true, y_pred, output_dir / "classification_report.txt")
    save_confusion_matrix_png(y_true, y_pred, output_dir / "confusion_matrix.png", title="Transformer Confusion Matrix")

    print("\nFinal test metrics:")
    for name, value in test_metrics.items():
        print(f"{name:16s}: {value:.4f}")
    print(f"\nSaved model: {best_model_dir}")
    print(f"Saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

