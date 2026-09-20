# EMMA Bench 测试：数据准备、生成和判分

新增入口支持官方 EMMA-mini（400 题）和完整 EMMA（2788 题），可测试官方 ILVR checkpoint 以及本项目的 Interaction-CE、baseline_ct、no_refinement 推理导出。**只使用 test，不加入训练。** 所有命令在 Linux 服务器的 ILVR 项目根目录执行。

## 1. 与论文对齐到什么程度

核对来源：

- [ILVR v3，§4.1 和 Table 2](https://arxiv.org/html/2512.05665v3)：EMMA 包括 Chemistry、Coding、Math、Physics；开放题使用 Qwen2.5-VL-72B 判分。论文的 OOD 实验模型经过 Zebra-CoT 10k 子集训练。
- [ILVR 官方仓库](https://github.com/XD111ds/ILVR/tree/13a67ed2ee975d6ca56b4e9eba18d68b352b166b)：核对到该版本未发布 EMMA 专用推理/评分脚本。
- [EMMA 官方仓库](https://github.com/EMMA-Bench/EMMA/tree/47259952dd8e95bc1e3855a4164ccd3634f8eb1e)：使用其 `configs/gpt.yaml`、`data_utils.py`、`models/qwen.py` 和 `evaluation/` 中的公开协议。

**不能保证逐项复现 ILVR 表格分数**：ILVR 论文没有明确给出 EMMA-mini/完整集的选择、题目 ID 列表、专用 prompt 和裁判模板等细节。这里默认官方发布的 mini 400 题，完整集也支持；不会把自行抽样的数据称为论文子集。若当前模型仅做过 CoMT 训练，这次是它的 EMMA 泛化测试，训练来源也不同于论文的 Zebra-CoT OOD 实验。

实际实现：

| 项目 | 本入口行为 |
| --- | --- |
| 文字 prompt | 直接读取固定版本的 EMMA 官方 YAML，默认 `CoT`，要求最终答案放在 `\boxed{}` |
| 图片 | 按 `<image_n>` 在题目和选项中的出现顺序插入，支持重复引用及最多 5 张源图；保留透明度和 EXIF 方向 |
| Qwen 输入 | 沿用 checkpoint 的 processor/chat template，`add_generation_prompt=True`、`add_vision_id=True`；与官方 Qwen wrapper 一样先调用 `qwen_vl_utils.process_vision_info` |
| 生成 | 沿用本项目连续 latent 执行器；greedy；默认最多 4096 个生成位置，包含连续 latent 位置 |
| latent 步数 | 读取模型 `config.json`，baseline 通常 8，Interaction 通常 9；不因换数据集而修改 |
| 正式评分路径 | 官方 EMMA 裁判 prompt + 本地 Qwen2.5-VL-72B-Instruct；对所有题判分，属于公开 EMMA LLM 评分方式。ILVR 未公布选择题的具体评分分工 |
| 快速评分路径 | 直接导入固定版本的官方 `fast_extract_answer` / `is_equal`，用于无裁判时检查结果；与 LLM 分数分开保存 |

不把 `answer`、`solution` 加入待测模型的输入。准备好的 JSONL 保留金标准供**生成之后**评分，`solution` 不导出。裁判只接收官方模板要求的回答和金标准，不额外修改题意或推理评分标准。

## 2. 下载与转换数据

使用现有 ILVR 评测环境及仓库内修改版 Transformers。安装数据转换/评分依赖不会要求替换 Transformers：

```bash
python -m pip install -r requirements-emma.txt

git clone https://github.com/EMMA-Bench/EMMA.git data/emma/official
git -C data/emma/official checkout 47259952dd8e95bc1e3855a4164ccd3634f8eb1e

hf download luckychao/EMMA-mini --repo-type dataset \
  --revision 6b9ae9a74733bb57f0b741213f8d0ec9ebae067a \
  --local-dir data/emma/raw

python -m src.interaction.emma prepare \
  --raw_dir data/emma/raw \
  --output_dir data/emma/prepared \
  --emma_repo data/emma/official \
  --variant mini --strategy CoT
```

数据图片已经嵌在 Parquet 中，**不需要另找图片包或解压 tar**。转换器提取图片，检查四科各 100 条、题目 ID 唯一、选项/答案合法及所有图片引用；记录原始文件和输出 JSONL 的 SHA-256。官方 prompt 和评分源文件也校验 SHA-256，版本不匹配会报错。

`prepared` 应为新目录，转换器不覆盖已有数据。目录为：

```text
ILVR/
├── data/emma/
│   ├── official/                  # 固定版本的 EMMA 官方代码
│   ├── raw/
│   │   ├── Chemistry/test-00000-of-00001.parquet
│   │   ├── Coding/test-00000-of-00001.parquet
│   │   ├── Math/test-00000-of-00001.parquet
│   │   └── Physics/test-00000-of-00001.parquet
│   └── prepared/
│       ├── TEST.jsonl
│       ├── manifest.json
│       └── images/<subject>/<pid>/image_1.png ...
├── checkpoints/ilvr_comt/
└── outputs/interaction_ce/inference_2/
```

**`--image_root` 是 `data/emma/prepared`，不是 `images` 子目录。** 转移数据到其他磁盘时，整体复制 `prepared/`，保留 `manifest.json` 和相对目录结构。不要手改 TEST 中的题目、顺序或 prompt；校验不一致会报错。

可单独预检，不加载权重、不用 GPU：

```bash
python -m src.interaction.emma validate \
  --test_data_path data/emma/prepared/TEST.jsonl \
  --image_root data/emma/prepared
```

这条单独预检可以跳过；生成入口仍会在模型加载前检查数据和图片。mini 应为 400 条、四科各 100 条，包含 341 道选择题和 59 道开放题。

完整集的准备方式如下；它不是 mini 的可直接比较替代结果：

```bash
hf download luckychao/EMMA --repo-type dataset \
  --revision 6c87aec9048c489108170088e50b768646b5bcf9 \
  --local-dir data/emma_full/raw
python -m src.interaction.emma prepare \
  --raw_dir data/emma_full/raw --output_dir data/emma_full/prepared \
  --emma_repo data/emma/official --variant full --strategy CoT
```

完整集共 2788 条：Chemistry 1176、Coding 564、Math 892、Physics 156。后续命令将 TEST 和 image_root 一起换到 `data/emma_full/prepared`。

## 3. 生成待测模型的回答

四卡并行，每张卡一个完整模型副本、处理不同题目；不需要 DeepSpeed 配置：

```bash
CUDA_VISIBLE_DEVICES=2,3,5,7 \
torchrun --standalone --nproc_per_node=4 --module src.interaction.evaluate \
  --task emma \
  --model_path outputs/interaction_ce/inference_2 \
  --test_data_path data/emma/prepared/TEST.jsonl \
  --image_root data/emma/prepared \
  --output_dir outputs/eval_emma_interaction \
  --max_new_tokens 4096 --attention_backend flash_attention_2
```

使用你的实际推理导出目录替换 `inference_2`。baseline 使用相同命令，改为：

```text
--model_path checkpoints/ilvr_comt
--output_dir outputs/eval_emma_baseline
```

单卡可把命令开头改为 `CUDA_VISIBLE_DEVICES=2 python -m src.interaction.evaluate`，其余参数不变。只设置四个 `CUDA_VISIBLE_DEVICES` 再运行普通 `python` 仍只启动一个进程。

新实验使用新的输出目录；中断后用原目录加 `--resume` 续跑，见第 8 节。入口会拒绝直接覆盖已有预测。两个模型使用相同数据、策略、生成预算和评分方式。若想测试官方 Direct 策略，重新 prepare 到另一目录并传 `--strategy Direct`；不要混入 CoT 结果。

生成时每完成一题输出进度并刷新该 rank 的 JSONL。默认 `--max_input_tokens 32768`，超限带 pid 报错，不静默截掉选项或图片。输出：

- `predictions_rank*.jsonl`：各卡即时结果；`predictions.jsonl`：完成后合并。
- `prompt_rank*.json`：每卡首题实际渲染 prompt、图片顺序、扩展 token IDs，方便核对输入。
- `evaluation_config.json`：模型路径、latent 步数、数据指纹、prompt 版本和生成参数。
- `emma_responses.json`：按 pid 索引、兼容官方评分输入的回答。
- `metrics.json`：此时 `scoring=pending_offline_scoring`，`accuracy=null`，表示**尚未评分**，不是准确率为零。

## 4. 用 Qwen2.5-VL-72B 裁判评分

这一步在待测模型生成完成、释放 GPU 后运行，**不需要重新生成回答**。

准备裁判 checkpoint（与被评测的 7B 模型分开放置）：

```bash
hf download Qwen/Qwen2.5-VL-72B-Instruct \
  --local-dir checkpoints/emma_judge
```

裁判推荐通过独立环境的 vLLM 提供本机接口；不要把 vLLM 安装到依赖修改版 Transformers 的 ILVR 训练环境中。[vLLM 官方 Qwen2.5-VL 配方](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen2.5-VL.html)提供了 72B 四卡 tensor parallel 部署方式。以下为 4×80GB BF16 的起始配置，实际显存仍取决于软件版本和上下文长度：

```bash
# 建立独立环境，仅首次需要。保持当前目录是 ILVR 根目录。
python3 -m venv .cache/emma-judge-venv
.cache/emma-judge-venv/bin/python -m pip install vllm

# 在单独终端启动服务；此时四张卡应已释放。
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=2,3,5,7 \
  .cache/emma-judge-venv/bin/vllm serve "$PWD/checkpoints/emma_judge" \
  --served-model-name Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 8 \
  --host 127.0.0.1 --port 8000
```

服务器就绪后，在原来的 ILVR 环境另开终端执行，评分客户端本身不占 GPU：

```bash
python -m src.interaction.score_emma \
  --test_data_path data/emma/prepared/TEST.jsonl \
  --predictions outputs/eval_emma_interaction/predictions.jsonl \
  --output_dir outputs/eval_emma_interaction/judge \
  --emma_repo data/emma/official \
  --backend judge \
  --judge_base_url http://127.0.0.1:8000/v1 \
  --judge_model Qwen/Qwen2.5-VL-72B-Instruct \
  --workers 4
```

裁判使用官方 few-shot 模板，`temperature=0`、`seed=42`、默认输出上限 32 tokens，要求 `Correct` / `Incorrect`。这些解码参数是本实现明确固定的设置，并非 ILVR 已公开的裁判配置。所有题都走同一裁判；空回答直接判错。

结果目录中 `metrics.json` 才是最终评分；同时保存逐题 `predictions_scored.jsonl` 和官方格式 `emma_results.json`。`scored.jsonl` 是随评分写入的断点日志。

若服务断连、返回异常文本或裁判输出被截断，该题保持 `correct=null`；最终总准确率也为 null，命令非零退出，不把服务故障当模型答错。修复服务后原命令加 **`--resume`**，只重试未成功评分的题。输入/数据/评分配置改变时会拒绝复用旧缓存，应使用新目录。

若只有各卡的文件，也可把 `--predictions` 改为 `outputs/eval_emma_interaction/predictions_rank*.jsonl`；不要同时传合并文件和 rank 文件，以免重复。默认必须覆盖整个 test；`--allow_partial` 仅供诊断，并在 metrics 中标为覆盖不完整。

## 5. 暂时没有 72B 裁判：先跑官方快速规则

```bash
python -m src.interaction.score_emma \
  --test_data_path data/emma/prepared/TEST.jsonl \
  --predictions outputs/eval_emma_interaction/predictions.jsonl \
  --output_dir outputs/eval_emma_interaction/fast \
  --emma_repo data/emma/official --backend fast
```

不需要 API、裁判权重或 GPU。这条路径直接调用官方解析与等价判断：选择题字母、最后一个完整 boxed 答案、部分答案提示语、数字和 LaTeX/符号比较。**不使用 CoMT exact match，也不使用 VSP 的动作解析器。** 保留官方规则的行为，不额外扩展答案规则来抬分。

快速规则不能覆盖所有开放题自然语言等价表达，与 72B 裁判的分数可能不同。报告时注明 `fast` 或 `judge`；不要将 fast 分数直接称为论文判分结果。之后准备好裁判，直接复用同一 `predictions.jsonl` 运行第 4 节。

## 6. 看哪些指标

- `accuracy`：全部题目的正确率，取值 0–1；任何题尚未成功评分则为 null。
- `by_subject`：Chemistry、Coding、Math、Physics 的题数和准确率。
- `macro_subject_accuracy`：四科准确率的算术平均，便于查看论文式四科均分。mini 四科等量，因此与总体 accuracy 一致；完整集两者可能不同。
- `by_type`、`by_task`、`by_category`：题型、任务、类别统计。Coding 的多标签类别会分别计数，类别计数之和可能超过样本数。
- `token_limit_count`：达到生成预算的题数；不会从准确率分母里删除。
- `pending`、`scored`、`complete_test_coverage`：检查评分故障和是否只测了一部分。
- `ilvr_exact_reproduction=false`：记录前述未公开细节和训练来源差异，不表示程序报错。

保留 baseline 和训练后模型的完整输出目录，以及裁判模型下载版本和 vLLM 版本。两次评分应使用同一个裁判服务/权重。

## 7. 已做的验证

本地 CPU 环境已完成：

- 官方 mini 全部 400 题转换、图片可读性和计数检查。
- 与固定版本官方实现比较：800 个 CoT/Direct query、616 次图片引用经预处理后的像素内容、400 个裁判 prompt 均一致；五图题 `chem_82` 的真实 processor 输出 `input_ids`、`image_grid_thw`、`pixel_values` 逐项一致。
- 用金标准构造 400 条合成回答，完整跑通官方 fast 评分入口，结果为 400/400；这只验证评分程序，不是待测模型准确率。另验证错误答案、空回答、嵌套 boxed 和分数等价。
- EMMA 测试与原 VSP 回归共 20 项通过，包含数据错误、图片顺序、答案不进入 prompt、裁判异常不误算为错误、评分恢复与入口输出。

安装依赖后可运行：

```bash
python -m unittest tests.test_interaction_emma tests.test_interaction_vsp tests.test_interaction_vsp_audit -v
```

本地未运行真实 CUDA 模型生成或 72B 裁判，也未测得模型在 EMMA 上的准确率；这两步需要在你的服务器按上面的命令执行。完整 2788 题的数据计数依据官方数据集元信息，逐题比对在 mini 上完成。

## 8. 多卡完成不同步、超时与续跑

旧版在评测结束时才调用 NCCL barrier，默认超时 600 秒。如果某卡已完成、其他卡仍在处理较慢题目，快卡可能等待超时；torchrun 随后终止其余进程。日志中的 `rank 2 ...100/100` 只代表该卡完成，不代表四卡都完成；`GPU ... currently unknown` 也不能单独证明 GPU 映射出错。[PyTorch 超时说明](https://docs.pytorch.org/docs/2.6/distributed.html#torch.distributed.init_process_group)

新版仅用 **CPU Gloo** 协调各进程完成，模型计算仍在各自 GPU 上；默认等待上限 **4 小时**，通过 `--sync_timeout_seconds` 调整。没有降低 token 预算、跳题或更改 prompt/生成算法。若其他进程真实崩溃，仍需看它最早的异常，不把所有故障归结为同步。

保留服务器原输出目录里的 `evaluation_config.json` 和 `predictions_rank*.jsonl`，不要拿 FinalShell 的部分本地快照覆盖服务器文件。上传新版 `evaluate.py`、`generation.py` 和新增的 `evaluation_state.py` 后，用**原启动命令、原模型、原数据、原卡数、原生成参数**加 `--resume`。例如原来使用本文四卡命令时：

```bash
CUDA_VISIBLE_DEVICES=2,3,5,7 \
torchrun --standalone --nproc_per_node=4 \
  --log-dir outputs/emma_resume_logs --tee 3 \
  --module src.interaction.evaluate \
  --task emma \
  --model_path outputs/interaction_ce/inference_2 \
  --test_data_path data/emma/prepared/TEST.jsonl \
  --image_root data/emma/prepared \
  --output_dir outputs/eval_emma_interaction \
  --max_new_tokens 4096 --attention_backend flash_attention_2 \
  --resume --sync_timeout_seconds 14400 --progress_interval_seconds 30
```

上述路径和 `max_new_tokens` 必须与中断运行一致；若你原来显式设置了其他 `max_input_tokens`，也一并保留。不要更换模型目录内的权重或编辑 TEST 后再续跑。续跑检查数据报告、模型路径、生成配置及每条结果的样本身份；旧版没有权重内容指纹，无法识别同一路径被偷偷替换权重的情况。

完整写入的回答直接跳过，包括 `correct=null` 和达到 token 上限的回答。只补做缺失样本；若最后一行被中断写坏，先备份到 `*.partial-*` 再恢复，文件中间损坏或重复样本则报错。续跑完成后重新合并并校验全部题目。`resume_config.json` 记录本次续跑设置，原 `evaluation_config.json` 保留。

每卡新增 `progress_rankN.json`，并输出以下阶段：

| stage | 含义和排查方向 |
| --- | --- |
| `preprocessing` | 读取/缩放图片、tokenization；长时间停留需查图片和磁盘 IO |
| `vision` | 图像传入 GPU、视觉编码与 embedding；结合 GPU 利用率和本卡异常日志判断 |
| `prefill` | 处理完整输入 prompt；图片 token 多时可能更慢 |
| `decoding` | 自回归生成；循环每约 30 秒记录 `generated_tokens` 和耗时，数值增长表示仍在推进 |
| `saved` | 一题回答已写入并刷新到 JSONL |
| `waiting_for_other_ranks` | 本卡已完成，等待其他卡 |
| `complete` | 本卡已结束；rank 0 也已完成合并 |

如果 GPU 调用或 IO 本身阻塞，进度不会按 30 秒强行更新，最后一个阶段就是排查起点。新结果还记录 `preprocessing_seconds`、`vision_seconds`、`sample_seconds`、`input_tokens`；旧 `generation_seconds` 本来就不包含图片处理和视觉编码时间。

检查 rank 3：

```bash
cat outputs/eval_emma_interaction/progress_rank3.json
wc -l outputs/eval_emma_interaction/predictions_rank*.jsonl
nvidia-smi
```

`CUDA_VISIBLE_DEVICES=2,3,5,7` 时，local rank 3 对应物理 GPU 7。结合 `--tee 3` 保存的各进程日志，找 rank 3 自己**最早的 Traceback**，不要只看最后的 `ChildFailedError`。

如果四个 rank 文件其实已包含完整 400 题，合并阶段失败也不必重跑生成：第 4/5 节评分命令可直接使用 `--predictions outputs/eval_emma_interaction/predictions_rank*.jsonl`。评分入口会检查覆盖范围和重复题；文件不完整时先续跑，不用 `--allow_partial` 冒充完整集结果。

本次修复通过 27 项 CPU 测试，包括中断续跑、已完成题不重复生成和进度回调不改变 8/9 步生成结果；另用真实两进程 Gloo，让第二个进程晚 2 秒完成，验证等待和合并。Linux 四卡 CUDA 场景仍需服务器复测。
