from ride_classifier.config import CLASSES, RIDE_ID, TARGET
from ride_classifier.cross_validate import ride_folds
from ride_classifier.evaluate import gate_passed


def test_every_ride_is_tested_once_and_no_ride_leaks_between_parts(small_data):
    labels = small_data[[RIDE_ID, TARGET]].drop_duplicates()
    folds = ride_folds(labels, k=3, val_share=0.25, seed=0)

    tested = [ride for fold in folds for ride in fold["test"]]
    assert sorted(tested) == sorted(labels[RIDE_ID])
    for fold in folds:
        train, val, test = set(fold["train"]), set(fold["val"]), set(fold["test"])
        assert not (train & val or train & test or val & test)
        assert train | val | test == set(labels[RIDE_ID])
        assert labels[labels[RIDE_ID].isin(test)][TARGET].nunique() == len(CLASSES)


def _cv(fold_scores: list[float], baseline: float = 0.18) -> dict:
    folds = [{"test_macro_f1": s, "test_baseline_macro_f1": baseline} for s in fold_scores]
    return {"mean": {"test_macro_f1": sum(fold_scores) / len(fold_scores)}, "per_fold": folds}


def test_gate_uses_the_cross_validated_mean_and_every_fold():
    gate = {"min_cv_macro_f1": 0.25}
    version = {"test_macro_f1": 0.33, "test_baseline_macro_f1": 0.18}
    assert gate_passed(version, _cv([0.34, 0.27, 0.24, 0.32, 0.44]), gate)
    assert not gate_passed(version, _cv([0.22, 0.24, 0.23, 0.25, 0.24]), gate)  # mean below the floor
    assert not gate_passed(version, _cv([0.40, 0.40, 0.17, 0.40, 0.40]), gate)  # one fold at baseline level
    assert not gate_passed({"test_macro_f1": 0.17, "test_baseline_macro_f1": 0.18}, _cv([0.34] * 5), gate)
