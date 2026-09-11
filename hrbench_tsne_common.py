"""Portable CPU-only feature archive and joint t-SNE utilities.

Identical copies live beside the three model entry points so each repository
can be uploaded independently. Keep the copies synchronized when editing.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FEATURE_FILES = ("vocabulary_embeddings.npy", "image_features.npy", "latent_features.npy")
KIND_NAMES = np.asarray(["vocabulary_embedding", "image_feature", "latent"])
BLOCK_SIZE = 8192


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)


def sample_indices(available, limit, seed):
    if available < 0 or limit <= 0:
        raise ValueError("Available count must be nonnegative and sample limit positive.")
    return np.sort(np.random.default_rng(seed).choice(
        available, size=min(available, limit), replace=False
    )).astype(np.int64)


def prepare_capture_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Capture output is not empty: {path}. Use a new OUTPUT_DIR.")
    path.mkdir(parents=True, exist_ok=True)
    # Exclusive claim also prevents two capture processes from using one folder.
    write_json(path / "capture_started.json", {"created_at": datetime.now(timezone.utc).isoformat()})
    return path


def validate_settings(mode, vocabulary_limit, image_limit, options, run_name):
    if mode not in {"capture_only", "capture_and_tsne", "tsne_only"}:
        raise ValueError("RUN_MODE must be capture_only, capture_and_tsne, or tsne_only.")
    if vocabulary_limit <= 0 or image_limit <= 0:
        raise ValueError("Vocabulary and image sample limits must be positive.")
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name or any(c in run_name for c in "/\\:"):
        raise ValueError("TSNE_RUN_NAME must be one directory name.")
    if options["pca_components"] < 3:
        raise ValueError("PCA_COMPONENTS must be at least 3 for 3D PCA initialization.")
    if not np.isfinite(options["perplexity"]) or options["perplexity"] <= 0:
        raise ValueError("TSNE_PERPLEXITY must be finite and positive.")
    if options["max_iter"] < 250:
        raise ValueError("TSNE_MAX_ITER must be at least 250.")


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_vector(metadata, name, length):
    if name not in metadata:
        raise ValueError(f"Missing metadata: {name}")
    value = metadata[name]
    if value.shape != (length,) or value.dtype.kind not in "iu":
        raise ValueError(f"Invalid integer metadata shape/dtype: {name}")
    return value


def validate_features(arrays, metadata, statistics):
    if len(arrays) != 3 or any(a.ndim != 2 for a in arrays):
        raise ValueError("Expected three [N, hidden_size] feature matrices.")
    if len({a.shape[1] for a in arrays}) != 1 or arrays[0].shape[1] == 0:
        raise ValueError("Feature hidden dimensions must match and be positive.")
    for name, array in zip(FEATURE_FILES, arrays):
        if array.dtype != np.float16:
            raise ValueError(f"{name}: expected original capture dtype float16, got {array.dtype}")
        for start in range(0, len(array), BLOCK_SIZE):
            if not np.isfinite(array[start:start + BLOCK_SIZE]).all():
                raise ValueError(f"Non-finite original features in {name}")
    nv, ni, nl = map(len, arrays)
    ns = len(metadata["dataset_indices"])
    if metadata["dataset_indices"].ndim != 1 or metadata["request_ids"].shape != (ns,):
        raise ValueError("Dataset/request identifiers must have one entry per sample.")
    _integer_vector(metadata, "dataset_ordinals", ns)
    ids = _integer_vector(metadata, "vocabulary_token_ids", nv)
    if len(np.unique(ids)) != nv or np.any(ids < 0) or np.any(ids >= statistics["available_vocabulary_embeddings"]):
        raise ValueError("Vocabulary token IDs are duplicated or out of range.")
    positions = _integer_vector(metadata, "image_global_positions", ni)
    if len(np.unique(positions)) != ni or np.any(positions < 0) or np.any(positions >= statistics["available_image_features"]):
        raise ValueError("Image sample indices are duplicated or out of range.")
    for prefix, count in (("image", ni), ("latent", nl)):
        ordinals = _integer_vector(metadata, prefix + "_sample_ordinal", count)
        if np.any(ordinals < 0) or np.any(ordinals >= ns):
            raise ValueError(f"Invalid {prefix} sample ordinal.")
        for suffix in ("sequence_positions", "generation_steps"):
            _integer_vector(metadata, prefix + "_" + suffix, count)
    _integer_vector(metadata, "latent_indices", nl)
    steps = _integer_vector(metadata, "latent_trajectory_steps", nl)
    offsets = _integer_vector(metadata, "latent_sample_offsets", ns + 1)
    if offsets[0] != 0 or offsets[-1] != nl or np.any(np.diff(offsets) < 0):
        raise ValueError("Invalid latent trajectory offsets.")
    if not np.array_equal(np.diff(offsets), statistics["per_sample_latent_counts"]):
        raise ValueError("Latent counts disagree with trajectory offsets.")
    image_counts = np.asarray(statistics["per_sample_image_counts"])
    if image_counts.shape != (ns,) or np.any(image_counts < 0) or image_counts.sum() != statistics["available_image_features"]:
        raise ValueError("Invalid per-sample image counts.")
    expected_samples = np.searchsorted(np.cumsum(image_counts), positions, side="right")
    if not np.array_equal(expected_samples, metadata["image_sample_ordinal"]):
        raise ValueError("Image global positions disagree with sample ordinals.")
    for sample in range(ns):
        start, end = map(int, offsets[sample:sample + 2])
        if not np.all(metadata["latent_sample_ordinal"][start:end] == sample):
            raise ValueError("Latent rows do not follow sample offset order.")
        if not np.array_equal(steps[start:end], np.arange(end - start)):
            raise ValueError("Latent trajectory steps are not consecutive.")
        if np.any(np.diff(metadata["latent_sequence_positions"][start:end]) < 0):
            raise ValueError("Latent sequence positions are out of order.")


def save_features(output_path, arrays, metadata, records, capture_config, statistics, model_name, semantics):
    """Commit a verified archive; manifest.json is the completion marker."""
    output_path = Path(output_path)
    metadata = dict(metadata)
    metadata.update(
        dataset_indices=np.asarray([str(r["dataset_index"]) for r in records]),
        request_ids=np.asarray([str(r["request_id"]) for r in records]),
        dataset_ordinals=np.asarray([r["dataset_ordinal"] for r in records], dtype=np.int64),
    )
    validate_features(arrays, metadata, statistics)
    feature_dir = output_path / "features"
    feature_dir.mkdir(exist_ok=False)
    files = {}
    for filename, original in zip(FEATURE_FILES, arrays):
        path = feature_dir / filename
        with path.open("xb") as handle:
            np.save(handle, original, allow_pickle=False)
        saved = np.load(path, mmap_mode="r", allow_pickle=False)
        if saved.dtype != original.dtype or saved.shape != original.shape:
            raise ValueError(f"Feature round-trip failed: {filename}")
        for start in range(0, len(saved), BLOCK_SIZE):
            if not np.array_equal(saved[start:start + BLOCK_SIZE], original[start:start + BLOCK_SIZE]):
                raise ValueError(f"Feature round-trip failed: {filename}")
        files[filename] = {"shape": list(saved.shape), "dtype": str(saved.dtype), "sha256": _digest(path)}
        del saved
    with (feature_dir / "metadata.npz").open("xb") as handle:
        np.savez_compressed(handle, **metadata)
    with np.load(feature_dir / "metadata.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(metadata) or any(not np.array_equal(saved[k], v) for k, v in metadata.items()):
            raise ValueError("Metadata round-trip failed.")
    with (output_path / "results.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
    capture_info = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name, "config": capture_config, "statistics": statistics,
        "feature_semantics": semantics,
        "selected_dataset_ordinals": metadata["dataset_ordinals"].tolist(),
    }
    write_json(output_path / "run_config.json", capture_info)
    manifest = dict(capture_info, schema_version=1, status="complete", arrays=files,
                    metadata_sha256=_digest(feature_dir / "metadata.npz"),
                    kind_names=KIND_NAMES.tolist(),
                    sampling={"replacement": False, "seed": capture_config["RANDOM_SEED"],
                              "vocabulary": "Sorted original embedding-table row IDs (token IDs).",
                              "image": "Sorted indices into all captured image rows concatenated in sample order.",
                              "latent": "All captured latent rows, in sample and trajectory order."},
                    precision="Original float16 capture; no normalization or dimensionality reduction.")
    # Written last: partial captures are never accepted as reusable caches.
    write_json(feature_dir / "manifest.json", manifest)
    print(f"Original high-dimensional features saved: {feature_dir}", flush=True)
    return feature_dir


def load_features(feature_dir, expected_model=None):
    feature_dir = Path(feature_dir).expanduser().resolve()
    try:
        manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
            raise ValueError("Feature cache is incomplete or has an unsupported schema.")
        if expected_model is not None and manifest["model"] != expected_model:
            raise ValueError(f"Expected {expected_model} feature cache, got {manifest['model']}.")
        arrays = []
        for name in FEATURE_FILES:
            info = manifest["arrays"][name]
            if _digest(feature_dir / name) != info["sha256"]:
                raise ValueError(f"Feature checksum mismatch: {name}")
            array = np.load(feature_dir / name, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != info["shape"] or str(array.dtype) != info["dtype"]:
                raise ValueError(f"Feature shape/dtype mismatch: {name}")
            arrays.append(array)
        if _digest(feature_dir / "metadata.npz") != manifest["metadata_sha256"]:
            raise ValueError("Metadata checksum mismatch.")
        with np.load(feature_dir / "metadata.npz", allow_pickle=False) as data:
            metadata = {key: data[key] for key in data.files}
        validate_features(arrays, metadata, manifest["statistics"])
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid feature cache at {feature_dir}: {exc}") from exc
    return arrays, metadata, manifest


def joint_metadata(metadata, counts):
    nv, ni, nl = counts
    points = {"kind_codes": np.repeat(np.arange(3, dtype=np.uint8), counts),
              "kind_names": KIND_NAMES,
              "token_ids": np.concatenate((metadata["vocabulary_token_ids"], np.full(ni + nl, -1, dtype=np.int64))),
              "dataset_indices": metadata["dataset_indices"], "request_ids": metadata["request_ids"],
              "dataset_ordinals": metadata["dataset_ordinals"]}
    for key in ("sample_ordinal", "sequence_positions", "generation_steps"):
        points[key] = np.concatenate((np.full(nv, -1, dtype=np.int32), metadata["image_" + key], metadata["latent_" + key]))
    for key, source in (("latent_indices", "latent_indices"), ("trajectory_steps", "latent_trajectory_steps")):
        points[key] = np.concatenate((np.full(nv + ni, -1, dtype=np.int32), metadata[source]))
    return points


def run_tsne(feature_dir, output_dir, options, expected_model=None):
    """Fit all categories together on CPU, without importing any model code."""
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    validate_settings("tsne_only", 1, 1, options, "validation")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Projection directory already exists: {output_dir}. Change TSNE_RUN_NAME.")
    arrays, metadata, manifest = load_features(feature_dir, expected_model)
    counts = [len(a) for a in arrays]
    total, hidden_size = sum(counts), arrays[0].shape[1]
    components = min(options["pca_components"], total - 1, hidden_size)
    if components < 3 or not options["perplexity"] < total:
        raise ValueError("3D t-SNE requires at least four points, hidden_size >= 3, and perplexity < N.")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = {"status": "running", "feature_cache": str(Path(feature_dir).expanduser().resolve()),
              "feature_manifest_sha256": _digest(Path(feature_dir).expanduser() / "manifest.json"),
              "model": manifest["model"], "capture_config": manifest["config"],
              "point_counts": dict(zip(KIND_NAMES.tolist(), counts)),
              "options": options, "actual_pca_components": components,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__}}
    write_json(output_dir / "started.json", config)
    work_path = output_dir / "joint_input.float32.npy"
    work = None
    try:
        print(f"Joint PCA: N={total:,}, D={hidden_size} -> {components}", flush=True)
        work = np.lib.format.open_memmap(work_path, mode="w+", dtype=np.float32, shape=(total, hidden_size))
        offset = 0
        for array in arrays:
            for start in range(0, len(array), BLOCK_SIZE):
                block = array[start:start + BLOCK_SIZE]
                work[offset + start:offset + start + len(block)] = block
            offset += len(array)
        work.flush()
        pca = PCA(n_components=components, whiten=False, svd_solver="randomized", random_state=options["random_state"], copy=False)
        reduced = pca.fit_transform(work).astype(np.float32, copy=False)
        if not np.isfinite(reduced).all() or not np.any(np.var(reduced, axis=0) > 0):
            raise ValueError("PCA produced invalid or zero-variance input for t-SNE.")
        work._mmap.close()
        work = None
        work_path.unlink()
        # Useful for inspection; never presented as a replacement for raw features.
        np.save(output_dir / "pca_preprocessed.npy", reduced, allow_pickle=False)
        np.savez_compressed(output_dir / "pca_preprocessing.npz", components=pca.components_, mean=pca.mean_,
                            explained_variance_ratio=pca.explained_variance_ratio_)
        params = dict(n_components=3, perplexity=options["perplexity"], init="pca", learning_rate="auto",
                      random_state=options["random_state"], metric="euclidean", method="barnes_hut",
                      verbose=1, n_jobs=options["n_jobs"])
        iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
        params[iteration_key] = options["max_iter"]
        estimator = TSNE(**params)
        coordinates = estimator.fit_transform(reduced).astype(np.float32, copy=False)
        if coordinates.shape != (total, 3) or not np.isfinite(coordinates).all():
            raise ValueError("t-SNE did not produce finite [N, 3] coordinates.")
        points = joint_metadata(metadata, counts)
        np.savez_compressed(output_dir / "joint_tsne_3d.npz", coordinates=coordinates, **points)
        trajectory = {"coordinates": coordinates[counts[0] + counts[1]:],
                      "sample_offsets": metadata["latent_sample_offsets"],
                      "dataset_indices": metadata["dataset_indices"], "request_ids": metadata["request_ids"]}
        for key in ("sample_ordinal", "sequence_positions", "generation_steps", "trajectory_steps"):
            trajectory[key] = metadata["latent_" + key]
        trajectory["latent_indices"] = metadata["latent_indices"]
        np.savez_compressed(output_dir / "latent_trajectories.npz", **trajectory)
        config.update(status="complete", tsne_parameters=params, kl_divergence=float(estimator.kl_divergence_),
                      n_iter=int(estimator.n_iter_), elapsed_seconds=time.perf_counter() - started,
                      pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist())
        write_json(output_dir / "run_config.json", config)
        print(f"3D t-SNE complete: {output_dir}", flush=True)
        return output_dir
    except Exception as exc:
        write_json(output_dir / "failure.json", {"error": str(exc), "feature_cache_preserved": str(feature_dir)})
        raise
    finally:
        if work is not None:
            work._mmap.close()
        if work_path.exists():
            work_path.unlink()
