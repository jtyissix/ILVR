"""Official Hugging Face sources, pinned to revisions inspected on 2026-11-12."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    patterns: tuple


SOURCES = {
    "mmvp": Source("MMVP/MMVP", "37eafecab8a3940c50c2ade5b36de69dbc99a8cf", ("Questions.csv", "MMVP Images/*")),
    "mmstar": Source("Lin-Chen/MMStar", "bc98d668301da7b14f648724866e57302778ab27", ("mmstar.parquet",)),
    "vstar": Source("craigwu/vstar_bench", "d9ae62c903da0c98336e85c5ee89cd863b04b4da", ("direct_attributes/*", "relative_position/*")),
    "hrbench_4k": Source("DreamMr/HR-Bench", "83b9013d6293b85dc507e87199ca52517536939c", ("hr_bench_4k.parquet",)),
    "hrbench_8k": Source("DreamMr/HR-Bench", "83b9013d6293b85dc507e87199ca52517536939c", ("hr_bench_8k.parquet",)),
    "mathvista": Source("AI4Math/MathVista", "2b6ad69445fbb5695c9b165475e8decdbeb97747", ("data/testmini-*.parquet",)),
    "mm_math": Source("THU-KEG/MM_Math", "7d4cfaabd7597b3495f6ad14a242ec92fc0e0d4d", ("MM_Math/MM_Math.jsonl", "MM_Math/MM_Math.zip")),
    "scienceqa": Source("derek-thomas/ScienceQA", "f18b0a70359ebfb41f658fd564208d0355b013f4", ("data/test-*.parquet",)),
    "emma": Source("luckychao/EMMA", "6c87aec9048c489108170088e50b768646b5bcf9", ("*/test-*.parquet",)),
    "blink": Source("BLINK-Benchmark/BLINK", "a3666eb249237ba3d5eca8db21176cc47967e040", ("*/val-*.parquet",)),
    "visualpuzzles": Source("neulab/VisualPuzzles", "e6bc28fb1adcd27f25476b917e8c35c67b64027b", ("data/train-*.parquet",)),
    "visulogic": Source("VisuLogic/VisuLogic", "3e483f8cca0bc766ecc5f70299f2b8a34b5035ca", ("data.jsonl", "images.zip")),
    "seedbench": Source("AILab-CVC/SEED-Bench", "81cb1571b8a3a30af2e62073b8d2054418350f0a", ("SEED-Bench.json", "SEED-Bench-image.zip")),
}
EMMA_MINI = Source("luckychao/EMMA-mini", "6b9ae9a74733bb57f0b741213f8d0ec9ebae067a", ("*/test-*.parquet",))
EXPECTED_COUNTS = {"mmvp": 300, "mmstar": 1500, "vstar": 191,
                   "hrbench_4k": 800, "hrbench_8k": 800, "mathvista": 1000,
                   "scienceqa": 2017, "emma": 2788, "blink": 1901,
                   "visualpuzzles": 1168, "visulogic": 1000, "seedbench": 14233}


def source_for(name, emma_variant="full"):
    if emma_variant not in ("full", "mini"):
        raise ValueError("EMMA_VARIANT must be 'full' or 'mini'")
    return EMMA_MINI if name == "emma" and emma_variant == "mini" else SOURCES[name]
