from ride_classifier.config import CLASSES, RIDE_ID, TARGET
from ride_classifier.cross_validate import ride_folds


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
