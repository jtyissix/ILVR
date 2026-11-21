# 多模态模型评测

独立下载、离线模型推理、答案判分三部分。支持原始 **Qwen2.5-VL-7B-Instruct** 和本项目 **SFT-II / SFT3** 导出的 **backbone**。入口只需修改文件顶部全局变量，不需要命令行参数。

## 运行

本目录位于 **`ILVR/eval/`**，下面的命令均从 **ILVR 项目根目录**执行。复用已有 `ilvr` 环境时，只安装项目根目录的增量依赖：

```bash
conda activate ilvr
python -m pip install -r requirements-eval.txt
```

该文件在完整 `ILVR/requirements.txt` 环境上补充 `python-Levenshtein` 和 `antlr4-python3-runtime==4.11.1`，复用已有 Torch、torchvision、Transformers 分支及 qwen-vl-utils。如果只装过精简的 `requirements-interaction.txt`，还需确认 `requirements.txt` 中的数据处理依赖（如 pyarrow、qwen-vl-utils、SymPy）已经安装。**不要在 ILVR 训练环境中安装 `eval/requirements.txt`**，该文件供全新独立评测环境使用。

注意：旧 `requirements-emma.txt` 使用 `latex2sympy2` 和 ANTLR **4.7.2**，与 MM-Math 所需 **4.11.1** 冲突。安装增量依赖会切换到 4.11.1；如果仍需要旧 EMMA 快速判分，请使用单独的判分环境。新的 `eval/` 使用 API 判分 EMMA，不依赖旧快速判分器。

全新独立 Python 3.10+ 评测环境可以先安装 CUDA 匹配的 PyTorch，再执行 `python -m pip install -r eval/requirements.txt`。只在下载机器上准备数据，可以仅安装 `eval/requirements-data.txt`，不需要 GPU。推理兼容 ILVR 仓库中的 Transformers 分支和标准 Transformers 4.51.3；不加载训练辅助模块。测试依赖另装 `pytest`。

脚本默认路径根据自身位置计算：数据位于 `ILVR/eval/datasets`，结果位于 `ILVR/eval/outputs`。手动填写的绝对 `DATASET_DIR`、`MODEL_DIR`、`OUTPUT_DIR` 则继续使用指定位置。

`eval/` 是 ILVR Git 仓库中的普通目录，不是独立仓库或子模块。忽略规则统一位于 `ILVR/.gitignore`：默认下载目录、评测结果和 Python 工具缓存不会进入版本控制，源码、测试、依赖清单及 `vendor/` 参考代码可以正常提交。`ILVR/.gitattributes` 为评测源码固定 LF 换行。如果将数据或结果改到其他仓库内目录，需要为该自定义目录补充忽略规则。

1. 修改 `download.py` 顶部的 `DATASET_DIR`、数据集选择和可选的 `HF_ENDPOINT`、`HF_TOKEN`。
2. 执行 `python eval/download.py`。使用固定 Hugging Face revision，下载需要的文件，统一转换 PNG 并转换标注，生成文件清单；不会下载 SEED 视频包、ScienceQA 训练集、MathVista 隐藏答案的 test。
3. 修改 `evaluate.py` 顶部的 `DATASET_DIR`、`MODEL_DIR`，以及判分需要的 `API_KEY`。兼容第三方接口时同时填写 `API_BASE_URL` 和 `JUDGE_MODEL`。基座与训练模型用不同 `RUN_NAME`。
4. 执行 `python eval/evaluate.py`。

`MODEL_DIR` 可以是原始模型的本地 HF 目录、训练导出根目录，或其 `backbone/` 子目录。必须包含对应 tokenizer、processor 和完整 HF 权重；DeepSpeed 原始分片需要先用项目的 `export_stage3` / `export_stage2` / `export_warmup` 导出。不会退回基座权重或基座 tokenizer 掩盖缺失文件。

断点处理：保持原 `DATASET_DIR` 重跑即可跳过已完成的数据集；已有完整 `samples.jsonl` 但缺少清单时，会直接补齐清单。新版处理中断后按 `images/source_*.png` 逐图续接，存在的文件直接跳过，`.part` 临时文件重写。图片使用低压缩等级 PNG（`compress_level=1`），不做哈希、去重、内容比较或转换后的复检。上一版 `.img` 会就地转换为同名 `.png` 并更新标注，转换中断可继续，原 `.img` 保留。旧版只生成了部分去重图片、尚未写出标注时，没有原题到图片的映射，无法可靠恢复这些散图；它们会原样保留，未完成的数据集走新的 PNG 转换流程。

下载后的结构为：

```text
DATASET_DIR/
  mmvp/
    raw/                    # 官方文件，可保留用于复核
    prepared/
      samples.jsonl
      images/
      manifest.json         # revision、实际条数、图片列表
  mmstar/ ...
```

`ONLY_DATASETS`、`EXCLUDE_DATASETS` **必须恰好一个非空**，两个都空也会报错。下载脚本和评测脚本各自有这组配置：

```python
ONLY_DATASETS = ["all"]             # 测全部
EXCLUDE_DATASETS = []

# 或：只测三项，hrbench 展开为 4K、8K
ONLY_DATASETS = ["mmvp", "hrbench", "mathvista"]
EXCLUDE_DATASETS = []

# 或：排除两项
ONLY_DATASETS = []
EXCLUDE_DATASETS = ["seedbench", "mm_math"]
```

名称支持 `V*`、`Math-Vista`、`MM-Math`、`VisualPuzzle`、`SEED-Bench` 等别名，规范名称见下表。

## 默认数据与原始计分协议

以下计分列对应 `SCORING_PROTOCOL="official"`；当前默认 robust 的差异见“最终答案提取与评分口径”。

| 配置名称 | 默认范围 / 条数 | 原始计分方式 |
|---|---:|---|
| `mmvp` | 官方 300 题 / 150 对 | 原始 GPT judge 提示，两题都正确才算题对正确；另报逐题准确率 |
| `mmstar` | 1,500 | 官方前缀答案匹配，逐题准确率及六类成绩 |
| `vstar` | 191 | 官方原始选项完整文本的平均 token NLL，选最小值；分两类报告 |
| `hrbench_4k` | 800 条 / 200 题 × 4 轮 | 官方 yes/no judge 提示；先算各轮准确率，再平均四轮 |
| `hrbench_8k` | 800 条 / 200 题 × 4 轮 | 同上；不会误用“四轮全对才算对”的指标 |
| `mathvista` | `testmini` 1,000 | 官方答案抽取示例、选择题/整数/浮点/列表归一化、精确比较 |
| `mm_math` | 官方 JSONL 全量，论文口径 5,929 | 复用官方 `AutoScoringJudge`，默认精度 `1e-2`，比较 `\boxed{}` 最终答案 |
| `scienceqa` | test 的带图子集 2,017 | 官方 QCM→A 零样本格式、答案抽取与准确率 |
| `emma` | full test 2,788 | 官方 CoT 提示、按占位符交错插入图片、官方 Correct/Incorrect judge 提示 |
| `blink` | val 1,901 | 官方逐题 prompt、规则抽取及 GPT 选项匹配，14 个任务准确率的宏平均 |
| `visualpuzzles` | 作者 HF `train` split 的 1,168 道评测题 | 官方 CoT 提示、原版答案解析器、逐题准确率 |
| `visulogic` | 1,000 | 官方 CoT 提示、boxed/语言规则抽取，失败后使用官方 judge 抽取提示 |
| `seedbench` | SEED-Bench v1 Image 14,233 | 候选完整答案的平均 token NLL，选最小值；报告九个图像任务类别 |

MM-Math 的**图片数不等于题数**；最终条数以下载后 JSONL 和 `manifest.json` 为准，当前不把图片查看器计数硬编码为题数。准备阶段直接使用下载数据，不检查预期条数。HF revision 见 `registry.py`。

`EMMA_VARIANT="mini"` 可准备 400 题版本；`SCIENCEQA_FULL=True` 可准备包含无图题的完整 test 4,241 题。切换版本请使用新的 `DATASET_DIR`，避免混合已有数据。

## backbone 的 text / latent 交替

`MODEL_KIND="auto"` 识别项目 SFT-II 的标记，以及 SFT3 的 `sft3_latent_relaxation_backbone` 标记。SFT3 可填写 `export/` 或 `export/backbone/`。也可以显式设为 `"base"` 或 `"trained"`。

对于生成式测评，模型自行预测 `<|latent_start|>` 后，解码循环把**上一位置最终归一化 hidden state** 作为下一连续位置的输入，不将连续 latent 转换为词表 token。模型预测 `<|latent_end|>` 后恢复文本生成，可以进入多个 latent 段。边界 ID 从导出 tokenizer 和 config 校验读取，没有硬编码旧 ID。

默认每段至少一个 latent 位置，正常结束由模型预测；`MAX_LATENT_STEPS=64` 只是防止无限循环。强制结束、未闭合段、总生成预算截断均记录在预测文件中。`MAX_NEW_STEPS` 同时计入文本 token 和连续 latent 位置。

加载内容只有视觉编码器、语言 backbone 和 LM head；不会实例化 auxiliary decoder、InteractionCore / cross attention 或 latent projection。导出权重有缺失、意外键或 tokenizer 不一致时直接报错。

这里执行的是 **backbone-only 自回归推理**。SFT-II 训练轨迹中通过 InteractionCore 提供的 helper 信息在该路径中不存在，连续位置使用自身 hidden state 递推；不把它描述为完整复现带 helper 的训练轨迹。推理代码支持自动交替，但某道题是否真的进入 latent 段由模型预测决定，结果中的 `latent_segments` 可以直接核查。

SFT3 的 text/action、interaction、visual 三类 latent 均使用上一位置的最终 hidden state 回填，与此推理的递推计算一致。训练时的 4/3/16 位置用于确定训练监督布局；自由生成时保留由模型预测结束标记的方式，不额外强制固定段长。项目 Transformers 分支使用单卡推理 attention 路径，避免误入 Ulysses 训练路径。

**SEED-Bench 与 V* 默认是候选似然协议**，直接对候选答案做 teacher forcing，因此没有自由生成的 text/latent 链。这是其默认官方选择题评价方式。若要专门研究这些数据上的自主交替，把它们从 `LIKELIHOOD_DATASETS` 移除；该实验应标为生成式变体，不与原始候选 NLL 成绩混用。NLL 只平均答案 token，默认包括 Qwen 对话结束符，不计算问题和图片 token 的 loss；每个候选独立前向，避免共享可变缓存污染。

## 最终答案提取与评分口径

默认 `SCORING_PROTOCOL="robust"`，对基座、SFT-II、SFT3 使用完全相同的规则。原始 prompt、图片、生成预算、候选 NLL 和基准聚合方式不变；修改的是生成式输出的答案提取与比较，因此结果明确标记 `protocol_modified=true`，不能称为未经修改的官方协议成绩。

先识别嵌套 `\boxed{}` / `\fbox{}`、`The final answer is:`、`Answer:`、中文最终答案、独立选项字母及明确选项文本。优先最终结论，不从推理正文挑一个碰巧与标准答案相同的中间值。多种未解决的候选交给 API；提取 API 只收到问题、选项和模型回复，**不收到标准答案**，且必须提供回复中的原文证据。

数学答案统一根号、角度、分数等写法后比较。MM-Math 保留重复参考 boxed 的去重及所有不同答案部分；使用严格的数学解析和等价判断，不沿用原评分器隐式乘除 100 的百分比容差，也不把多解答案的一部分当完整答案。不能可靠作符号比较时，再由 API 仅比较已提取的最终答案与参考答案；不得替模型解题或补全缺失答案。MathVista 继续使用原版数字精度与答案归一化。

本地提取失败会调用 API，API 仍确认没有明确答案时计错并记录 `unparsed`；空回复直接计错。API 调用失败、返回无效证据或等价比较为 UNKNOWN 时保留为待判分，不计错、不从分母中悄悄排除；完整准确率保持空值。`RULES_ONLY=True` 时需要 API 的题也保持待判分。

若需原协议复现，设 `SCORING_PROTOCOL="official"`，保留原 MMStar 前缀规则、ScienceQA/VisualPuzzles 随机兜底及原 judge。不要对不同模型选择较高的一套成绩；报告比较时统一协议、judge、参数和覆盖率。

已有预测无需重新生成：保持原 `MODEL_DIR`、`RUN_NAME` 及推理配置，设置 `MODE="score"`、`SCORING_PROTOCOL="robust"`。新评分版本会分配新的判分目录，旧逐题评分仍保留。要另报原协议，再用 `SCORING_PROTOCOL="official"` 重评分；两种结果不会取最大值。修改模型权重或推理设置时另用 `RUN_NAME`。

## 配置、结果与恢复

- `MODE="all"`：生成并判分；`"infer"`：只保存模型输出；`"score"`：只判分已有输出，不加载模型；`"validate"`：只检查数据、图片与提示构建。
- `API_KEY` 仅用于答案抽取/判分。本地推理不需要它。robust 模式在无法明确提取或无法可靠比较时调用 API；official 模式沿用各基准原 judge 流程。
- `JUDGE_MODELS` 支持为不同基准指定不同模型；改变 judge 配置只产生新的判分记录。`JUDGE_MODEL` 是可配置替代模型，不声称复用了论文当年的已退役服务。
- `MIN_PIXELS` / `MAX_PIXELS` 默认使用 checkpoint 保存的 processor 值，实际值和每题 `image_grid_thw` 都会记录。比较基座与训练模型时应显式保持一致。降低像素上限会改变 HRBench / V* 的高分辨率条件；不会在 OOM 后偷偷降低它。
- 当前每个进程使用 `DEVICE` 指定的一张卡，逐题推理。多卡可按互不重叠的数据集拆分进程，并设置不同 `RUN_NAME`。
- `RESUME=True` 复用成功结果、重试失败题。API 超时、限流、无效格式不会被记为模型答错；已生成答案保留，未完成判分时主 `score=null`，并报告 `partial_score` 与实际覆盖量。
- `MAX_SAMPLES` 仅用于调试，结果标记 `diagnostic_subset`，且不会截断 MMVP 题对或 HRBench 四轮组。
- `RULES_ONLY=True` 在 robust 模式禁止 API，并把需要 API 的题保留为待判分；official 模式沿用原离线简化规则，MathVista 不支持该简化。
- 模型记录路径、文件大小和修改时间；配置以原始 JSON 直接比较，结果和缓存按编号保存。目录中的 `index.sqlite3` 保存编号映射，复制结果或缓存时须一起保留。图片统一保存为 PNG，不去重、不比较内容、不做额外验证。已有 PNG 直接跳过；下载时遇到已有 `manifest.json` 直接复用（含 `.img` 引用时自动转换并更新标注）。不检查数据集版本、预期题数或完整性，普通评测也不预扫描图片；只有显式 `MODE="validate"` 才进行数据检查。修改数据、源码或覆盖模型后请使用新的 `RUN_NAME`。

结果在 `OUTPUT_DIR/RUN_NAME/summary.json` 和 `summary.csv`，每完成一个数据集就更新两份汇总，并在终端打印该数据集的准确率。**全部数据集统一报告逐题准确率**，包括使用候选 NLL 的 V* 和 SEED-Bench（先选出候选，再与金标准比较）：

| 字段 | 含义 |
|---|---|
| `accuracy` / `accuracy_percent` | 逐题准确率，分别为 0–1 / 0–100；全部所选题目判分完成后给出 |
| `correct` / `incorrect` | 已判分题目中的正确 / 错误题数 |
| `scored` / `expected` / `pending` | 已判分 / 所选总题数 / 尚未完成判分的题数 |
| `coverage` / `scoring_complete` | 判分覆盖率（0–1）/ 是否全部判分完成 |
| `partial_accuracy` / `partial_accuracy_percent` | 未完成时，仅已判分题目的准确率；此时正式 `accuracy` 为 null |
| `metric` / `score` / `score_percent` | 保留原始基准指标名称及其 0–1 / 0–100 数值 |
| `sample_accuracy` | `accuracy` 的兼容别名 |
| `scoring_protocol` / `scorer_version` / `protocol_modified` | 评分口径、实现版本和是否修改原判分协议 |
| `api_fallback_count` / `unparsed_count` | 成功判分中使用 API 兜底的题数 / 最终仍无可提取答案的题数 |
| `extraction_methods` | 各提取方式的题数（JSON 汇总）；逐题记录另含最终答案和提取原文证据 |

例如 MMVP 4 题答对 3 题、只有 1 对全对时，逐题 `accuracy_percent=75`，官方 `pair_accuracy` 的 `score_percent=50`。BLINK 也同时报告逐题准确率与官方任务宏平均。API 失败不会当成答错；未完成时 CSV 对应正式准确率列留空，终端显示 `PENDING`。仅推理模式不产生准确率。调试子集继续标注 `diagnostic_subset`，其成绩只代表所选子集。

每个数据集子目录保存相同的准确率字段、推理配置、逐题预测、独立判分记录及错误清单。预测包含文本、原始特殊 token、latent 段、截断状态或各候选 NLL。API key 不写入结果文件。API 响应按完整请求缓存；失败请求和格式无效的响应不缓存。修改汇总展示代码不会使已有推理和判分缓存失效；保持原配置重新运行或使用 `MODE="score"` 即可更新报表。

如果先做推理后换机器判分，拷贝同一结果目录，保持模型目录身份及推理配置不变，设 `MODE="score"`。该模式仍读取本地模型文件的身份信息，但不加载权重到内存，不需要 GPU。

## 对原始协议的适配说明

这些基准没有统一的模型推理配置。这里使用 Qwen 的 chat template、动态分辨率处理、greedy 解码及统一默认生成预算；预算和像素条件必须与成绩一同报告。

1. **V***：复用候选 NLL 和原始标注（每题第一个选项正确），候选评分不向模型暴露正确标签；不复现作者 SEAL 模型专用的视觉搜索器、目标定位与裁剪流程。那会引入本次要求之外的模型。生成式变体保留原选项顺序，标签偏置尤其明显，应单独报告。
2. **HRBench**：保留作者 judge 提示和四轮聚合，用可配置的兼容 API 代替原 lmdeploy 服务。温度为 0.2，但后端 top-k 等设置不完全一致；记录判分模型，不声称数值等价。
3. **MMVP**：使用作者原始 HF 标签；不会默默采用其他框架对第 99/100、279/280 题的标签修订。judge 提示保持原版，服务模型可替换。
4. **ScienceQA、VisualPuzzles**：保留原解析器“解析失败随机选项”的行为，并逐题标记 `random_fallback`、汇总次数。为了恢复与执行顺序无关，随机种子按题 ID 派生，与作者一次性遍历时的随机序列不同。ScienceQA 原始抽取要求唯一的 `The answer is X.` 形式；请同时查看随机兜底比例。
5. **MathVista**：保留官方抽取后的编辑距离选择项、整数/精度归一化，包括原版默认的空抽取处理方式。提取 prompt 不含待测题金标准答案。
6. **MM-Math**：实现官方最终答案 outcome evaluation；未实现额外的 GPT-4V 错误步骤定位 / process evaluation。后者属于额外过程分析，不混入准确率。
7. **ScienceQA / BLINK / EMMA**：使用有公开答案的默认范围。没有拿隐藏 test 生成一个假的本地准确率。ScienceQA 的 lecture/solution 不进入输入；EMMA 的占位符图片保持顺序及重复引用。

原始代码来源和 commit 在 `vendor/sources.json`；许可证和来源说明在 `vendor/NOTICE.md`。提示与计分规则主要对应：[MMVP](https://github.com/tsb0601/MMVP)、[MMStar](https://github.com/MMStar-Benchmark/MMStar)、[V*](https://github.com/penghao-wu/vstar)、[HRBench](https://github.com/DreamMr/HR-Bench)、[MathVista](https://github.com/lupantech/MathVista)、[MM-Math](https://github.com/kge-sun/MM-Math)、[ScienceQA](https://github.com/lupantech/ScienceQA)、[EMMA](https://github.com/EMMA-Bench/EMMA)、[BLINK](https://github.com/zeyofu/BLINK_Benchmark)、[VisualPuzzles](https://github.com/neulab/VisualPuzzles)、[VisuLogic](https://github.com/VisuLogic-Benchmark/VisuLogic-Eval)、[SEED-Bench](https://github.com/AILab-CVC/SEED-Bench)。

## 验证

```bash
python -m pytest eval/tests -q
```

测试包含全部适配器的小型 CSV/JSONL/Parquet/图片样本、官方判分规则、题对与循环聚合、下载产物校验、API 缓存及失败恢复。另使用随机初始化的小型真实 Qwen2.5-VL，在 CPU 上比较普通前向、带图前向、连续 latent 缓存递推和候选答案 NLL；同时检查无辅助模块加载。测试不下载 7B 权重，不调用收费 API。完整基准下载和真实 7B/GPU 全量跑分需要在你的评测环境执行。
