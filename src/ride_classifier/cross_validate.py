"""Stratified k-fold cross-validation BY RIDE: how noisy is the single frozen test split?

    python -m ride_classifier.cross_validate                  # CPU
    python -m ride_classifier.cross_validate --device cuda    # GPU

Every ride is tested exactly once (all 6 Reckless rides instead of 1). In each fold, part of the remaining rides is
held out for early stopping, as in the pipeline. Same params.yaml as the served model.
Measurement only: no MLflow run, no promotion; the frozen split, the gate and the served model are untouched.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ride_classifier.config import CLASSES, RIDE_ID, ROOT, TARGET, load_params
from ride_classifier.data import load_raw
from ride_classifier.model import confusion, majority_baseline, ride_scores
from ride_classifier.train import fit

OUT = ROOT / "reports" / "cv.json"
HEADLINE = ["macro_f1", "balanced_accuracy", "accuracy"]


def ride_folds(labels: pd.DataFrame, k: int, val_share: float, seed: int) -> list[dict[str, list[str]]]:
    """k folds of ride ids: test = one fold, val = a stratified share of the rest (early stopping), train = the rest."""
    rides = labels.drop_duplicates(RIDE_ID).reset_index(drop=True)
    folds = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for rest_idx, test_idx in skf.split(rides[RIDE_ID], rides[TARGET]):
        rest = rides.iloc[rest_idx]
        train_ids, val_ids = train_test_split(
            rest[RIDE_ID], test_size=val_share, stratify=rest[TARGET], random_state=seed
        )
        folds.append({"train": list(train_ids), "val": list(val_ids), "test": list(rides.iloc[test_idx][RIDE_ID])})
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    params = load_params()
    p_data, p_feat, p_train = params["data"], params["features"], params["train"]
    df = load_raw(ROOT / p_data["raw_rides_path"], ROOT / p_data["raw_labels_path"], p_data["max_missing_share"])
    # val keeps the pipeline's share of ALL rides (15 %), taken from the k-1 training folds
    val_share = p_data["val_share"] / (1 - 1 / args.folds)
    folds = ride_folds(df[[RIDE_ID, TARGET]], args.folds, val_share, p_train["random_state"])

    per_fold, preds = [], []
    for i, ids in enumerate(folds, start=1):
        part = {name: df[df[RIDE_ID].isin(ids[name])].reset_index(drop=True) for name in ids}
        t0 = time.perf_counter()
        model, best = fit(part["train"], part["val"], p_feat, p_train, device=args.device)
        pred = model.predict(None, part["test"])
        train_labels = part["train"].drop_duplicates(RIDE_ID)[TARGET]
        scores = {
            "fold": i,
            "test_rides": len(ids["test"]),
            "test_reckless": int((part["test"].drop_duplicates(RIDE_ID)[TARGET] == "Reckless").sum()),
            "best_epoch": best["best_epoch"],
            "val_macro_f1": best["val_macro_f1"],
            **ride_scores(part["test"], pred, "test"),
            **majority_baseline(part["test"], train_labels, "test"),
            "train_s": time.perf_counter() - t0,
        }
        per_fold.append(scores)
        preds.append(pred)
        print(
            f"fold {i}: macro F1 {scores['test_macro_f1']:.3f} (baseline {scores['test_baseline_macro_f1']:.3f}), "
            f"balanced acc {scores['test_balanced_accuracy']:.3f}, best epoch {best['best_epoch']}, "
            f"{scores['train_s']:.0f} s"
        )

    table = pd.DataFrame(per_fold)
    all_pred = pd.concat(preds, ignore_index=True)
    gate = params["gate"]["min_macro_f1"]
    result = {
        "folds": args.folds,
        "device": args.device,
        "rides": int(df[RIDE_ID].nunique()),
        "mean": {f"test_{m}": float(table[f"test_{m}"].mean()) for m in HEADLINE},
        "std": {f"test_{m}": float(table[f"test_{m}"].std(ddof=1)) for m in HEADLINE},
        "baseline_mean_macro_f1": float(table["test_baseline_macro_f1"].mean()),
        "folds_passing_gate": int(
            ((table["test_macro_f1"] >= gate) & (table["test_macro_f1"] > table["test_baseline_macro_f1"])).sum()
        ),
        "gate_min_macro_f1": gate,
        # every ride predicted once, by the model that did not see it: per-class recall over all 500 rides
        "pooled": ride_scores(df, all_pred, "cv"),
        "pooled_confusion": confusion(df, all_pred),
        "per_fold": table.to_dict(orient="records"),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    m, s = result["mean"], result["std"]
    print(
        f"\n{args.folds}-fold CV: macro F1 {m['test_macro_f1']:.3f} ± {s['test_macro_f1']:.3f} "
        f"(baseline {result['baseline_mean_macro_f1']:.3f}), balanced acc {m['test_balanced_accuracy']:.3f} "
        f"± {s['test_balanced_accuracy']:.3f}; gate passed on {result['folds_passing_gate']}/{args.folds} folds"
    )
    recalls = {c: round(result["pooled"][f"cv_recall_{c.lower().replace(' ', '_')}"], 3) for c in CLASSES}
    print(f"pooled recall per class: {recalls}\n-> {OUT.relative_to(ROOT)}")
    np.set_printoptions(linewidth=120)
    print(np.array(result["pooled_confusion"]["rows_true_cols_pred"]))


if __name__ == "__main__":
    main()
