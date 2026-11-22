from ride_classifier.config import CHANNELS, CLASSES, load_params


def test_params_file_has_every_section():
    params = load_params()
    assert {"data", "features", "train", "gate"} <= params.keys()


def test_split_shares_leave_room_for_training():
    p = load_params()["data"]
    assert 0 < p["test_share"] + p["val_share"] < 0.5


def test_schema_constants():
    assert len(CHANNELS) == 6
    assert len(CLASSES) == 5 and "Reckless" in CLASSES
