"""Edit the globals, then run python eval/visualize_latent.py from ILVR."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===================== 只修改此配置区 =====================
EVAL_ROOT = Path(__file__).resolve().parents[2] / "eval"
RUN_NAME = "sft3eval"             # qwen25vl7b / sftiieval / sft3eval
DATASET = "mmstar"               # 每次一个数据集、一个文件
INFERENCE_ID = "00000001"
SCORE_ID = "00000001"
RECORD_KIND = "scores"           # scores / predictions；不能混用两目录的编号
FILE_ID = "00001365"             # 文件名！也可填 1365 或 "00001365.json"

DATASET_DIR = "/home/fit/renjujty/WORK/jty/dataset/lm-eval"
MODEL_DIR = None                 # None 使用该次 inference_config.json 中的模型路径
DEVICE = "cuda:0"
LAYERS = [-4, -3, -2, -1]        # decoder 层，负数从末尾数；"all" 为所有层
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs/latent_attention"
OVERLAY_ALPHA = 0.65
DISPLAY_MAX_SIDE = 1024          # 仅限制导出 PNG 大小，不改变模型输入分辨率
VALIDATE_ONLY = False           # True 只检查文件/样本/图片路径，不加载模型
# =========================================================

from eval.latent_attention import visualize_record


if __name__ == "__main__":
    visualize_record(eval_root=EVAL_ROOT, run_name=RUN_NAME, dataset=DATASET,
                     inference_id=INFERENCE_ID, score_id=SCORE_ID,
                     record_kind=RECORD_KIND, file_id=FILE_ID, dataset_dir=DATASET_DIR,
                     model_dir=MODEL_DIR, device=DEVICE, layers=LAYERS, output_dir=OUTPUT_DIR,
                     alpha=OVERLAY_ALPHA, max_side=DISPLAY_MAX_SIDE, validate_only=VALIDATE_ONLY)
