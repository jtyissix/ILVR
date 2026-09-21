# Zebra-CoT Jigsaw / Visual Search 数据准备与评测

新增入口支持 baseline 和 Interaction-CE 导出模型，支持单卡、4 卡独立生成、断点续跑。默认直接用 **ILVR 仓库的规则评分**，无需裁判模型；开放答案也可以在生成完成后另用模型裁判评分。

## 1. 先明确测试范围

核对来源：[ILVR 论文 v3 §4.1、表 2](https://arxiv.org/html/2512.05665v3)、[官方 eval.py](https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/eval.py)、[Zebra-CoT 数据集](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT/tree/f6d1defc169d19180a69f517925ed6cbb06a1c97)。

| 任务 | 官方目录 | 公开 split | 完整条数 | Parquet 数量 / 下载量 |
| --- | --- | --- | ---: | --- |
| Jigsaw | `2D Visual Reasoning - Visual Jigsaw` | `train` | 21,899 | 26 个 / 约 12.1 GB |
| Visual Search | `2D Visual Reasoning - Visual Search` | `train` | 30,000 | 27 个 / 约 12.8 GB |

这里的 `train` 是公开数据的命名。论文把这两种 **2D 任务类型**留出作为 OOD 测试，训练采用其他任务的 Zebra-CoT 10k 数据；公开论文和仓库没有提供可核验的 Jigsaw/Search 测试题目 ID、抽样 seed、原始数据转换脚本或裁判 prompt。因此，本入口测试你实际准备的全部题目或指定抽样，不把它称为“论文官方 test split”。两个任务的完整公开数据也不等同于论文表 2 的具体测试集。

你现在的官方 CoMT checkpoint / CoMT 上的 Interaction-CE 继续训练模型，与论文 OOD 实验使用的 Zebra-CoT 10k 训练模型，训练来源也不同。可以公平比较这两个模型在同一份数据上的表现，但不能将结果视为表 2 的精确复现。输出始终记录 `ilvr_exact_reproduction=false`。

## 2. 下载：下面三种方式任选一种

在 Linux 项目根目录执行，保留现有 ILVR 的 torch、Transformers 和 flash-attn 环境：

```bash
python -m pip install -r requirements-zebra.txt
# 已安装 requirements-interaction.txt 时已有 hf 命令；若缺失，沿用项目版本：
python -m pip install huggingface-hub==0.35.3
```

### A. 先下载少量 Parquet 分片

例如各下载第一个分片（约 0.94 GB），即可测试其中的全部样本，或进一步抽样：

```bash
hf download multimodal-reasoning-lab/Zebra-CoT \
  "2D Visual Reasoning - Visual Jigsaw/train-00000-of-00026.parquet" \
  "2D Visual Reasoning - Visual Search/train-00000-of-00027.parquet" \
  --repo-type dataset --revision f6d1defc169d19180a69f517925ed6cbb06a1c97 \
  --local-dir data/zebra/raw
```

也可以从网页下载任意一个或多个 `.parquet` 文件。两个任务的分片文件名相似，**请放在各自的任务文件夹下**。无需额外下载图片压缩包，图片字节已在 Parquet 内。

### B. 完整下载这两个任务（推荐需要全量时使用）

```bash
hf download multimodal-reasoning-lab/Zebra-CoT \
  --repo-type dataset --revision f6d1defc169d19180a69f517925ed6cbb06a1c97 \
  --include "2D Visual Reasoning - Visual Jigsaw/*.parquet" \
            "2D Visual Reasoning - Visual Search/*.parquet" \
  --local-dir data/zebra/raw
```

两个任务合计约 24.9 GB，转换后还需要额外空间存放题目图。只有下载齐全部分片，才是上表的完整条数；程序允许少量分片，不会为了凑齐条数自动联网下载。

### C. 下载整个 Zebra-CoT

```bash
hf download multimodal-reasoning-lab/Zebra-CoT \
  --repo-type dataset --revision f6d1defc169d19180a69f517925ed6cbb06a1c97 \
  --local-dir data/zebra/raw
```

全数据约 71 GB。准备程序会忽略其他任务，只转换 Jigsaw / Visual Search。下载后布局如下（只下载部分分片也适用）：

```text
data/zebra/raw/
├── 2D Visual Reasoning - Visual Jigsaw/
│   ├── train-00000-of-00026.parquet
│   └── ...
├── 2D Visual Reasoning - Visual Search/
│   ├── train-00000-of-00027.parquet
│   └── ...
└── 其他任务目录（如果下载了全数据）
```

## 3. 转换成评测输入

默认读取已下载、可识别的两个任务的 **全部可用样本**。只有一个任务目录时也能运行。

```bash
python -m src.interaction.zebra prepare \
  --raw_dir data/zebra/raw \
  --output_dir data/zebra/prepared
```

如果只想先测少量题，例如每个任务最多 200 题：

```bash
python -m src.interaction.zebra prepare \
  --raw_dir data/zebra/raw \
  --max_samples_per_task 200 --seed 42 \
  --output_dir data/zebra/prepared_200
```

这是**从已下载分片中做的自定义随机抽样**，并非论文官方 200 题划分。下载分片不同时，抽样池也不同。样本少于 200 时保留全部，并在 manifest 写明实际数量。改变下载范围或 seed 后请使用新输出目录；baseline 和训练后模型务必复用同一份 prepared 数据。

也可以明确指定文件、通配符，或处理浏览器下载后改名的文件：

```bash
# 同时指定两个任务的若干分片：保留父目录名以识别任务。
python -m src.interaction.zebra prepare \
  --parquet "data/zebra/raw/2D Visual Reasoning - Visual Jigsaw/train-00000-of-00026.parquet" \
            "data/zebra/raw/2D Visual Reasoning - Visual Search/train-00000-of-00027.parquet" \
  --output_dir data/zebra/prepared_selected

# 文件放在平铺目录或已改名：明确告诉程序它属于哪个任务。
python -m src.interaction.zebra prepare \
  --parquet "data/zebra/downloads/*.parquet" \
  --subset visual_search --output_dir data/zebra/prepared_search
```

平铺目录里的文件必须全部属于 `--subset` 指定的同一个任务，程序无法从两种相同结构的 schema 自动辨别题型。也可使用 `--raw_dir data/zebra/downloads --subset jigsaw`。重复传入相同文件或同内容副本会报错。

转换产物：

```text
data/zebra/prepared/
├── TEST.jsonl
├── manifest.json
└── images/
    ├── jigsaw/
    └── visual_search/
```

`image_root` 必须指向 `data/zebra/prepared`（或你实际的 prepared 目录）。这里 `TEST.jsonl` 是本项目评测输入文件名，不代表 HF 提供了名为 TEST 的划分。

准备程序只读取 `Question`、`Final Answer`、`problem_image_1` 三列，按小批次提取原图，保留编码和像素。不读取或保存 `Text Reasoning Trace` / `reasoning_image_*`。哈希大型分片和提取图片可能需要几分钟，终端会逐分片报告。manifest 记录输入分片路径、内容哈希、抽样方式、实际条数和 TEST 哈希；每题记录来源分片与行号。

可选完整校验（CPU，无需模型）：

```bash
python -m src.interaction.zebra validate \
  --test_data_path data/zebra/prepared/TEST.jsonl \
  --image_root data/zebra/prepared
```

不单独运行 validate 也能评测：生成入口仍会检查 metadata、TEST 哈希、题目格式、数量和图片路径。validate 额外校验图片内容哈希与可读性。

## 4. 对 baseline 和训练后模型运行评测

四卡评测训练后模型：

```bash
CUDA_VISIBLE_DEVICES=2,3,5,7 torchrun --standalone --nproc_per_node=4 \
  -m src.interaction.evaluate --task zebra \
  --model_path outputs/interaction_ce/inference \
  --test_data_path data/zebra/prepared/TEST.jsonl \
  --image_root data/zebra/prepared \
  --output_dir outputs/eval_zebra_interaction \
  --max_new_tokens 4096 --attention_backend flash_attention_2
```

baseline 使用相同命令，仅替换：

```text
--model_path checkpoints/ilvr_comt
--output_dir outputs/eval_zebra_baseline
```

只评一个任务，追加 `--zebra_subset jigsaw` 或 `--zebra_subset visual_search`，使用不同的 output_dir。`all` 表示 prepared 中已有的全部任务，不强制要求两种任务都存在。

单卡直接用 `CUDA_VISIBLE_DEVICES=2 python -m src.interaction.evaluate ...`。设置四个可见设备再运行普通 `python` 仍只会用一张卡；四卡并行请用上面的 torchrun。每卡保留一份模型，样本按 rank 分配，结束用 CPU Gloo 同步，沿用现有 4 小时等待超时及 `--sync_timeout_seconds`。

中断后在**同一命令、同一模型、同一数据和卡数**上追加 `--resume`，已完整保存的题会跳过。不要在相同 checkpoint 路径下替换权重后续跑。`predictions_rankN.jsonl` 每题完成立即追加并 flush；`progress_rankN.json` 在一题生成过程中更新。生成默认 greedy，8/9 步 latent 从 checkpoint 配置读取，未更改原生成执行器。输入超出 `--max_input_tokens`（默认 32768）明确报错，不截断题目。

模型实际收到的是：**题目图 → 原始 Question 文字（移除图片占位标记）**，不追加新 CoT/boxed 指令，不把参考答案、文本推理或推理图放进输入。沿用 ILVR eval.py 的 PIL RGB 读取、checkpoint 自带 processor 和 chat template，不额外设置 EMMA 的 `add_vision_id`。原始 Zebra 到 ILVR 的转换未公开，移除占位标记是本项目明确记录的适配约定。首题 `prompt_rankN.json` 保存实际 messages、模板结果和 token IDs，便于核对。

## 5. 评分与指标

生成时已经执行 **非模型判断器**评分，无需再传参数启用。`correct` 直接是布尔值；结果位于 output_dir 的 `metrics.json`，包括：

- `by_task.jigsaw.accuracy` / `by_task.visual_search.accuracy`：分任务准确率，范围 0–1。
- `accuracy`：全部样本的加权总体准确率。
- `macro_task_accuracy`：两个任务准确率的算术平均；只有一个任务时为 null。样本数量不同时与总体 accuracy 不同。
- `samples`、`correct`、`token_limit_count`、数据指纹及非精确复现标记。

规则直接加载当前仓库 `eval.py` 的纯函数，并验证它们与上面固定的官方 revision 一致：优先最后一个简单 `\boxed{...}`，其次 final answer / answer 提示，最后使用官方回退规则；清理首行、标点、特殊 token，再做大小写、数值和 yes/no 归一化比较。**这是 ILVR 的规则，不是 EMMA 的 fast 规则。**

该官方通用规则保留了 `a/true/yes`、`b/false/no` 的等价映射及裸文本回退逻辑，在开放答案中可能有局限。例如没有 final answer/boxed 包装的 `A knife.` 可能被提取为 `a`；`a wheelchair` 与 `wheelchair` 也不保证判等。本实现不静默修改这些规则，逐题输出 `raw_output`、`prediction`、`prediction_normalized`、`gold_normalized` 供审计。论文的开放题使用 Qwen2.5-VL-72B 裁判；规则分数不能直接等同于其语义评分。

仅在需要重评分或多卡生成完成但合并失败时，用 CPU 入口，无需重新推理：

```bash
python -m src.interaction.score_zebra \
  --test_data_path data/zebra/prepared/TEST.jsonl \
  --predictions 'outputs/eval_zebra_interaction/predictions_rank*.jsonl' \
  --output_dir outputs/eval_zebra_interaction/fast --backend fast
```

也可传合并后的 `predictions.jsonl`，但不要同时传合并文件和 rank 文件，重复样本会报错。若生成使用了 `--zebra_subset`，重评分也传同样的参数。评分默认检查完整覆盖；`--allow_partial` 仅用于诊断中途结果，报告会标记 `complete_test_coverage=false`。不要对仍在写入的 rank 文件做最终评分。

如需语义裁判，可复用 [EMMA 文档](emma_evaluation_zh.md)中准备的本地 OpenAI-compatible Qwen2.5-VL-72B 服务：

```bash
python -m src.interaction.score_zebra \
  --test_data_path data/zebra/prepared/TEST.jsonl \
  --predictions outputs/eval_zebra_interaction/predictions.jsonl \
  --output_dir outputs/eval_zebra_interaction/judge --backend judge \
  --judge_base_url http://127.0.0.1:8000/v1 \
  --judge_model Qwen/Qwen2.5-VL-72B-Instruct --workers 4
```

此模式使用公开的本项目 `zebra_project_judge_v1` rubric，只将问题、模型回答和参考答案送入裁判；没有宣称获得了论文未公开的裁判 prompt。`scoring_config.json` 保存 rubric 和裁判设置。评分失败为 `correct=null`，总体 accuracy 也为 null，不计作模型答错；修复服务后原命令加 `--resume` 重试失败项。保留 `rule_correct` 以便比较规则与语义判分；baseline 和新模型须使用相同评分方式。

## 6. 验证范围

36 项 CPU 测试通过，覆盖部分/完整目录、抽样、解析、评测入口、8/9 步配置接线、续跑，以及已有 EMMA/VSP 回归。官方两个任务的真实 Parquet 各取 3 条，已验证提取图片与深度校验；使用真实 checkpoint processor，对照官方 eval.py 的预处理语句，6 条的模板文本、`input_ids`、`image_grid_thw`、`pixel_values` 均一致（给两边相同的适配后 text_input）。

另用这 6 条的金标准构造最终回答，跑通离线规则评分，6/6 判对；这只是程序自检，不是模型准确率。本机没有运行 CUDA 模型生成或 72B 裁判，实际模型准确率需要在服务器测得。

```bash
python -m unittest tests.test_interaction_zebra tests.test_interaction_emma \
  tests.test_interaction_vsp tests.test_interaction_vsp_audit \
  tests.test_interaction_evaluation_state -v
```
