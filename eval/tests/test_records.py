from eval.common import record_path, write_json
from eval.evaluate import saved_result


def test_numbered_paths_compare_full_identity_and_survive_restart(tmp_path):
    first = record_path(tmp_path, {"model": "a", "options": [1, 2]})
    assert first.name == "00000001.json"
    assert record_path(tmp_path, {"options": [1, 2], "model": "a"}) == first
    other = record_path(tmp_path, {"model": "a", "options": [2, 1]})
    assert other != first
    # IDs stay safe and short regardless of the original dataset ID.
    long_id = "../../very/long/样本" * 100
    path = record_path(tmp_path, long_id)
    assert path.parent == tmp_path and len(path.name) == 13
    assert record_path(tmp_path, long_id) == path


def test_changed_prediction_invalidates_saved_score(tmp_path):
    identity = ["settings", {"text": "A"}]
    path, stored = saved_result(tmp_path, "sample", identity)
    assert stored is None
    write_json(path, {"id": "sample", "identity": identity, "status": "ok", "score": 1})
    assert saved_result(tmp_path, "sample", identity)[1]["score"] == 1
    assert saved_result(tmp_path, "sample", ["settings", {"text": "B"}])[1] is None
