"""The environment record pins the reference: same files -> same hashes, different files -> different hashes."""

import json

from ride_classifier.environment import collect, sha256, to_markdown


def test_reference_is_pinned_by_content_hash(model_dir, small_data, tmp_path):
    test_path = tmp_path / "test.parquet"
    small_data.to_parquet(test_path, index=False)

    env = collect(model_dir, test_path)
    ref = env["reference"]
    assert ref["model_version"] == "test"  # from model_meta.json of the fixture
    assert ref["model_parameters"] > 0 and ref["weights_bytes"] > 0
    assert ref["test_rides"] == small_data["ride_id"].nunique()
    assert ref["weights_sha256"] == sha256(model_dir / "artifacts" / "weights.pt")[:16]

    other = tmp_path / "other.parquet"
    small_data.head(100).to_parquet(other, index=False)
    assert collect(model_dir, other)["reference"]["test_sha256"] != ref["test_sha256"]


def test_environment_is_serialisable_and_readable(model_dir, small_data, tmp_path):
    test_path = tmp_path / "test.parquet"
    small_data.to_parquet(test_path, index=False)
    env = collect(model_dir, test_path)

    json.dumps(env)  # environment.json is read by the benchmark scripts
    assert env["packages"]["torch"] and env["hardware"]["logical_cpus"]
    text = to_markdown(env)
    assert "## Reference" in text and env["reference"]["test_sha256"] in text
