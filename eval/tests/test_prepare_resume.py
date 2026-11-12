import json
import io
from PIL import Image

from eval.common import write_json
from eval.data import ImageStore
from eval.download import prepare_one
from eval.registry import source_for
from dataclasses import asdict


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_legacy_completed_dataset_skips_all_image_work(tmp_path, monkeypatch):
    prepared = tmp_path / "hrbench_8k" / "prepared"
    manifest = {"count": 800, "settings": {
        "schema_version": 1, "legacy_extra": "ignored",
        "source": asdict(source_for("hrbench_8k")),
        "emma_variant": None, "scienceqa_full": None}, "images": {"legacy.png": "ignored"}}
    write_json(prepared / "manifest.json", manifest)
    monkeypatch.setattr("eval.download.normalize", lambda *a: (_ for _ in ()).throw(AssertionError("must skip")))
    # Neither raw files nor image/annotation scans are required to skip a completed dataset.
    assert prepare_one("hrbench_8k", tmp_path)["count"] == 800


def test_image_checkpoint_skips_decode_and_keeps_legacy_files(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    (prepared / "images").mkdir(parents=True)
    legacy = prepared / "images" / "00000001.png"
    legacy.write_bytes(b"legacy")
    store = ImageStore(tmp_path, prepared)
    first = store.save(image_bytes())
    # Simulate a process restart; the original source need not be read again.
    restarted = ImageStore(tmp_path, prepared)
    monkeypatch.setattr("eval.data.base64.b64decode", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("decode")))
    assert restarted.save("data:image/png;base64,unused") == first
    assert legacy.read_bytes() == b"legacy"
    partial = prepared / "images/source_00000002.png.part"
    partial.write_bytes(b"unfinished")
    second = restarted.save(image_bytes())
    with Image.open(prepared / second) as image:
        assert image.format == "PNG"
    assert not partial.exists()


def test_recover_completed_annotations_before_manifest(tmp_path, monkeypatch):
    prepared = tmp_path / "mmstar/prepared"
    prepared.mkdir(parents=True)
    rows = [{"images": {"image_1": "legacy.png"}}]
    (prepared / "samples.jsonl").write_text(json.dumps(rows[0]) + "\n")
    monkeypatch.setattr("eval.download.normalize", lambda *a: (_ for _ in ()).throw(AssertionError("must reuse")))
    result = prepare_one("mmstar", tmp_path)
    assert result["count"] == 1 and result["images"] == ["legacy.png"]


def test_partial_img_checkpoint_converts_without_reading_raw(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    (prepared / "images").mkdir(parents=True)
    old = prepared / "images/source_00000001.img"
    old.write_bytes(image_bytes())
    store = ImageStore(tmp_path, prepared)
    monkeypatch.setattr("eval.data.base64.b64decode", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("raw decode")))
    result = store.save("data:image/jpeg;base64,unused")
    assert result == "images/source_00000001.png"
    with Image.open(prepared / result) as image:
        assert image.format == "PNG"
    assert old.is_file()


def test_completed_img_dataset_migration_and_restart(tmp_path, monkeypatch):
    prepared = tmp_path / "mmstar/prepared"
    (prepared / "images").mkdir(parents=True)
    old = "images/source_00000001.img"
    new = "images/source_00000001.png"
    (prepared / old).write_bytes(image_bytes())
    write_json(prepared / "manifest.json", {"count": 1, "images": [old]})
    (prepared / "samples.jsonl").write_text(json.dumps({"images": {"image_1": old}}) + "\n")
    result = prepare_one("mmstar", tmp_path)
    assert result["images"] == [new]
    assert json.loads((prepared / "samples.jsonl").read_text())["images"]["image_1"] == new
    # Simulate a crash after annotations were replaced but before manifest update.
    write_json(prepared / "manifest.json", {"count": 1, "images": [old]})
    monkeypatch.setattr(Image, "open", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not reopen")))
    assert prepare_one("mmstar", tmp_path)["images"] == [new]
    assert prepare_one("mmstar", tmp_path)["images"] == [new]
