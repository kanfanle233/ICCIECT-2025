from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import (
    clean_text_for_tfidf,
    compute_metrics,
    default_stopwords,
    ensure_dir,
    load_all_splits,
    prepare_model_frame,
    require_packages,
    save_confusion_matrix_png,
    save_json,
    set_seed,
    sigmoid_scores,
    transform_ml_texts,
    write_classification_report,
)


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return sigmoid_scores(model.decision_function(X))
    return model.predict(X)


def make_feature_sets(clean_train, clean_dev, clean_test):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2

    word_vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=3000,
        sublinear_tf=True,
        min_df=2,
    )

    X_train_word = word_vectorizer.fit_transform(clean_train)
    X_dev_word = word_vectorizer.transform(clean_dev)
    X_test_word = word_vectorizer.transform(clean_test)

    X_train_char = char_vectorizer.fit_transform(clean_train)
    X_dev_char = char_vectorizer.transform(clean_dev)
    X_test_char = char_vectorizer.transform(clean_test)

    chi2_selector = SelectKBest(chi2, k=min(5000, X_train_word.shape[1]))
    X_train_chi2 = chi2_selector.fit_transform(X_train_word, y_train_global)
    X_dev_chi2 = chi2_selector.transform(X_dev_word)
    X_test_chi2 = chi2_selector.transform(X_test_word)

    return {
        "tfidf_word": {
            "matrices": (X_train_word, X_dev_word, X_test_word),
            "bundle": {"word_vectorizer": word_vectorizer, "char_vectorizer": char_vectorizer},
        },
        "char_ngram": {
            "matrices": (X_train_char, X_dev_char, X_test_char),
            "bundle": {"word_vectorizer": word_vectorizer, "char_vectorizer": char_vectorizer},
        },
        "combined": {
            "matrices": (
                hstack([X_train_word, X_train_char]),
                hstack([X_dev_word, X_dev_char]),
                hstack([X_test_word, X_test_char]),
            ),
            "bundle": {"word_vectorizer": word_vectorizer, "char_vectorizer": char_vectorizer},
        },
        "chi2": {
            "matrices": (X_train_chi2, X_dev_chi2, X_test_chi2),
            "bundle": {
                "word_vectorizer": word_vectorizer,
                "char_vectorizer": char_vectorizer,
                "chi2_selector": chi2_selector,
            },
        },
    }


def make_models(include_tree_models: bool):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=5.0, solver="liblinear", random_state=42),
        "LinearSVC": LinearSVC(C=1.0, random_state=42),
        "RidgeClassifier": RidgeClassifier(),
        "MultinomialNB": MultinomialNB(alpha=0.5),
    }
    if include_tree_models:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=250,
            min_samples_split=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    return models


def maybe_fast_subset(df, n: int | None, seed: int):
    if not n or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train traditional ML baselines for Chinese sentiment classification.")
    parser.add_argument("--data_dir", default="shopping_comments")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="Accepted for interface consistency; sklearn runs on CPU.")
    parser.add_argument("--include_tree_models", action="store_true", help="Also train RandomForest; slower on this dataset.")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a small smoke test subset.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()

    require_packages(
        {
            "pandas": "pandas",
            "numpy": "numpy",
            "sklearn": "scikit-learn",
            "scipy": "scipy",
            "jieba": "jieba",
            "joblib": "joblib",
            "matplotlib": "matplotlib",
        }
    )
    import joblib
    import pandas as pd

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_dir / "ml_baseline")

    frames = {split: prepare_model_frame(df) for split, df in load_all_splits(args.data_dir).items()}
    if args.fast_dev_run:
        frames["train"] = maybe_fast_subset(frames["train"], 2500, args.seed)
        frames["dev"] = maybe_fast_subset(frames["dev"], 1000, args.seed)
        frames["test"] = maybe_fast_subset(frames["test"], 1000, args.seed)
    elif args.max_train_samples:
        frames["train"] = maybe_fast_subset(frames["train"], args.max_train_samples, args.seed)

    stopwords = default_stopwords()
    print("Cleaning text with jieba...")
    clean_train = [clean_text_for_tfidf(text, stopwords) for text in frames["train"]["text_a"].tolist()]
    clean_dev = [clean_text_for_tfidf(text, stopwords) for text in frames["dev"]["text_a"].tolist()]
    clean_test = [clean_text_for_tfidf(text, stopwords) for text in frames["test"]["text_a"].tolist()]

    global y_train_global
    y_train_global = frames["train"]["label"].to_numpy()
    y_dev = frames["dev"]["label"].to_numpy()
    y_test = frames["test"]["label"].to_numpy()

    print("Building TF-IDF feature sets...")
    feature_sets = make_feature_sets(clean_train, clean_dev, clean_test)
    models = make_models(args.include_tree_models)

    rows = []
    trained = {}
    for feature_name, feature_data in feature_sets.items():
        X_train, X_dev, X_test = feature_data["matrices"]
        for model_name, model in models.items():
            print(f"Training {model_name} on {feature_name}...")
            start = time.time()
            model.fit(X_train, y_train_global)
            elapsed = time.time() - start
            dev_pred = model.predict(X_dev)
            dev_score = positive_scores(model, X_dev)
            dev_metrics = compute_metrics(y_dev, dev_pred, dev_score)
            rows.append(
                {
                    "Feature": feature_name,
                    "Model": model_name,
                    "Split": "dev",
                    "Time(s)": round(elapsed, 3),
                    **dev_metrics,
                }
            )
            trained[(feature_name, model_name)] = {
                "model": model,
                "feature_name": feature_name,
                "feature_bundle": feature_data["bundle"],
                "X_test": X_test,
            }

    results = pd.DataFrame(rows).sort_values(["F1-Score", "MCC"], ascending=False)
    results.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    results.to_csv(output_dir / "ml_baseline_metrics.csv", index=False, encoding="utf-8-sig")

    best = results.iloc[0]
    key = (best["Feature"], best["Model"])
    best_item = trained[key]
    best_model = best_item["model"]
    test_pred = best_model.predict(best_item["X_test"])
    test_score = positive_scores(best_model, best_item["X_test"])
    test_metrics = compute_metrics(y_test, test_pred, test_score)

    final = {
        "best_by_dev": {
            "feature": best["Feature"],
            "model": best["Model"],
            "dev_metrics": {k: float(best[k]) for k in results.columns if k not in {"Feature", "Model", "Split"}},
        },
        "test_metrics": test_metrics,
        "data_sizes": {split: int(len(df)) for split, df in frames.items()},
    }
    save_json(final, run_dir / "metrics.json")
    write_classification_report(y_test, test_pred, run_dir / "classification_report.txt")
    save_confusion_matrix_png(y_test, test_pred, run_dir / "confusion_matrix.png", title="ML Baseline Confusion Matrix")

    model_bundle = {
        "model_type": "ml_baseline",
        "model": best_model,
        "feature_name": best_item["feature_name"],
        "stopwords": sorted(stopwords),
        **best_item["feature_bundle"],
    }
    # Smoke-test the saved bundle transform before writing it.
    _ = transform_ml_texts(["这个商品质量很好", "太差了不会再买"], model_bundle)
    joblib.dump(model_bundle, run_dir / "model.joblib")

    print("\nBest dev model:")
    print(results.head(10).to_string(index=False))
    print("\nFinal test metrics:")
    for name, value in test_metrics.items():
        print(f"{name:16s}: {value:.4f}")
    print(f"\nSaved: {run_dir / 'model.joblib'}")
    print(f"Saved: {output_dir / 'ml_baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
