# ILVR Interaction-CE：数据准备与训练

本实现从官方 CoMT checkpoint 继续训练，目标平台为 **Linux、单机 8×80GB NVIDIA GPU**。
当前实现采用你确认的 **独立、冻结、在线 reference**：reference 与主干来自同一份初始 checkpoint，
reference 不更新、不做 EMA。它与 PDF 第 8.3 节的“当前模型 reference”有所区别。

训练图为：

```text
输入图像 --冻结视觉编码器，只计算一次--+--> 冻结 reference：局部 T_ref + 8 步 Z_ref
                                      |                     |
                                      |             Round-trip fusion -> b_train
                                      |                     |
                                      +--> 可训练主干：Text -> b_train -> 8 步新 Z -> 后续文字
                                                                                |
                                                                       唯一目标：CE
```

`Z_ref` 来自 reference 的连续自回归轨迹，不是 helper 图像池化结果。训练主干的 8 步 Z 在注入 b 后重新生成。
reference 提供冻结输入，CE 经 b 反向更新融合模块及主干；没有 latent 对齐、蒸馏、cosine 或 bridge loss。
推理只保留主干，连续生成 9 步（第一步是 interaction），不加载 reference/fusion。
这个训练与推理之间的差异是研究假设，准确率是否提高需通过对照实验验证。

## 1. 环境

在训练服务器上进入项目根目录。建议使用独立环境，安装与仓库兼容的依赖：

```bash
conda create -n ilvr-interaction python=3.11 -y
conda activate ilvr-interaction
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-interaction.txt
MAX_JOBS=8 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -c "import torch, transformers, deepspeed; print(torch.__version__, torch.cuda.is_bf16_supported()); print(transformers.__file__); print(deepspeed.__version__)"
```

FlashAttention 编译需要与上述 torch CUDA 版本匹配的 CUDA toolkit/nvcc 和编译器。
`transformers.__file__` 必须指向本项目的 `transformers/src/transformers/`；入口会检查这一点。
默认训练不用 TRL、LoRA、旧 EMA trainer 或旧 Ulysses 训练分支。

训练和导出也使用 CPU 内存。尤其 CPU 离线导出需要展开 optimizer 分片，建议至少预留 64GB 可用主机内存；
8 个进程同时加载初始模型的瞬时主机内存需求可能更高。完整训练 checkpoint 包含 optimizer 状态，磁盘占用明显大于推理权重。

## 2. 下载与放置

数据来自 [shuai22/comt](https://huggingface.co/datasets/shuai22/comt)，
初始化模型来自 [shuai22/comt_ckpt](https://huggingface.co/shuai22/comt_ckpt/tree/main)。
这是 ILVR 已训练的 CoMT checkpoint，不是 Qwen base。公开权重为 7 个 `.bin` 分片，总仓库大小约 31.8GB。

```bash
hf download shuai22/comt --repo-type dataset --local-dir data/comt
hf download shuai22/comt_ckpt --local-dir checkpoints/ilvr_comt
tar -xzf data/comt/comt.tar.gz -C data/comt
```

期望目录：

```text
ILVR/
├── data/comt/
│   ├── TRAIN.jsonl
│   ├── TEST.jsonl
│   ├── comt.tar.gz
│   └── images_comt/
│       ├── creation/
│       ├── deletion/
│       ├── selection/
│       └── ...
├── checkpoints/ilvr_comt/
│   ├── config.json
│   ├── pytorch_model.bin.index.json
│   ├── pytorch_model-00001-of-00007.bin
│   ├── ...全部 7 个分片
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── preprocessor_config.json
│   └── ...其余官方 tokenizer、chat template 等文件
└── outputs/
```

**`image_root=data/comt`，不是 `data/comt/images_comt`。** 官方 JSONL 中的路径已经含有 `images_comt/`：

```json
{
  "text_input": "题目与选项",
  "image_input": ["images_comt/creation/10003.png"],
  "sequence_plan": [
    {"type": "text", "content": "当前推理文字"},
    {"type": "latent", "helper_image": "images_comt/creation/10003.png"},
    {"type": "text", "content": "后续推理文字"},
    {"type": "text", "content": "The final answer is: 70°"}
  ],
  "original_final_answer": "70°"
}
```

无需改写官方字段或手动插入 token。代码将每个 `latent` step 转为对应数量的连续占位位置。
保留已有答案文字，不重复追加 `original_final_answer`；该字段用于评测。
helper 图像参与预检，但不进入训练图像编码器。推理只使用 `image_input` 和 `text_input`。

只需下载一份初始化 checkpoint。默认 `reference_path=null` 表示从 `model_path` 再加载一份冻结 decoder 到每张 GPU，
不需要在磁盘复制第二份权重。若显式填写 `reference_path`，其内容必须与初始 checkpoint 相同。
新代码显式执行连续递推，不会因官方 config 中的 `stage: stage1` 而启用旧对齐训练。

## 3. 数据预检和 token 缓存

在训练服务器上生成缓存（包含该服务器上的图像绝对路径）：

```bash
python -m src.interaction.data \
  --model_path checkpoints/ilvr_comt \
  --data_path data/comt/TRAIN.jsonl \
  --image_root data/comt \
  --prepared_dir data/comt/prepared
```

这一步不加载完整模型权重到 GPU。它检查 checkpoint 文件、Git LFS 未下载的指针、token ID、
图像路径、轨迹结构、标签、图像网格和长度，输出样本数、latent 段数及长度分位数。
具体错误保存在 `data/comt/prepared/validation_errors.json`；任何坏样本都会使准备失败，不会静默丢弃。

缓存始终保存 8 步参考布局，训练 collator 按实验模式增加 interaction slot。
每条样本预留增加 interaction 后的长度，上限默认32768，超长会报错，不截断 rationale 或最终答案。
图像 resize 配置沿用 checkpoint。图像、TRAIN 内容、processor 或根目录变化后要重新准备缓存。
首次启动还会读取权重计算 SHA-256；之后来源文件未改变时复用指纹缓存，恢复时校验来源。

## 4. 启动训练和三组对照

默认配置：

| 项目 | 默认值 |
|---|---|
| DeepSpeed | ZeRO-2，BF16，无 CPU/NVMe offload |
| 每卡 batch / 累积 | 1 / 2；8卡有效 batch=16 |
| epoch | 3 |
| 主干 LR / fusion LR | 2e-6 / 5e-5 |
| optimizer | fused AdamW，weight decay=0.01；一维参数不衰减 |
| scheduler / warmup | cosine / 5% |
| gradient clipping | 1.0 |
| seed | 42 |
| fusion | 1024维，8 heads，SwiGLU 4096，dropout=0，4个 residual scales 初值0.1 |
| 保存 / 日志间隔 | 每100个 optimizer steps / 每10步 |

```bash
# 完整方案：冻结在线 reference + 训练期 fusion；推理9步。
bash run_interaction_training.sh configs/interaction/interaction_ce.json

# 原始8步 CE 继续训练。
bash run_interaction_training.sh configs/interaction/baseline_ct.json

# 不使用 fusion，训练和推理都直接自回归9步。
bash run_interaction_training.sh configs/interaction/no_refinement.json
```

三个配置继承 `src/interaction/config.py` 中的公共默认值。修改学习率等参数可在相应 JSON 中显式设置，
公平对照时公共参数保持一致。上述三条命令分别启动独立实验，不应在同一组8卡上同时运行。
脚本尊重 `CUDA_VISIBLE_DEVICES`，不会擅自覆盖显卡选择。

短跑检查与等效更大 micro-batch：

```bash
bash run_interaction_training.sh configs/interaction/interaction_ce.json \
  --max_steps 5 --log_steps 1 --output_dir outputs/smoke

bash run_interaction_training.sh configs/interaction/interaction_ce_mb2.json
```

有效 batch=`GPU数 × micro_batch_size × gradient_accumulation_steps`。
默认 sampler 使每条真实样本每个 epoch 恰好出现一次；最后一个窗口用零 CE 的占位样本补齐，
不会重复训练真实样本。CE 依据整个分布式累积窗口中的有效标签数归一化。
`--max_steps` 是提前停止阈值，不改变3个 epoch 对应的学习率时间表，便于短跑后原样恢复。

## 5. 恢复与导出

训练自动产生完整 DeepSpeed checkpoint 和单独的推理权重：

```text
outputs/interaction_ce/
├── train_config.json
├── run_info.json
├── metrics.jsonl
├── checkpoints/
│   ├── latest
│   └── global_step00000100/
│       ├── ...DeepSpeed model/optimizer/scheduler 分片
│       ├── rng_rank0.pt ... rng_rank7.pt
│       └── complete.json
├── inference/
│   ├── model*.safetensors
│   ├── config.json
│   ├── ...processor/tokenizer
│   └── interaction_recipe.json
└── completion.json
```

```bash
bash run_interaction_training.sh configs/interaction/interaction_ce.json \
  --resume outputs/interaction_ce/checkpoints
```

也可将 `--resume` 指向某个 `global_step...` 子目录。模型、optimizer、scheduler、随机状态、epoch 和下个窗口位置一起恢复。
数据、reference、世界大小及重要训练配置不匹配时拒绝恢复。保持 `model_path` 指向初始 checkpoint，
不要把它替换为训练后权重；训练后权重由 `--resume` 加载。
已有 checkpoint 的输出目录若不指定 `--resume` 会报错，避免覆盖实验。

所有 rank 参与保存；`complete.json` 仅在完整保存后写入，`latest` 只指向完整 checkpoint。
checkpoint 默认全部保留，请按磁盘预算调整 `save_steps`；每份完整状态可能占用约百 GB，三个实验要分别预留空间。

推理导出不包含 reference 或 fusion，配置分别保存视觉步数8、interaction步数1、兼容字段 `latent_size=9`。
中断训练后也可单独进行 CPU 离线导出：

```bash
python -m src.interaction.export \
  --checkpoint outputs/interaction_ce/checkpoints \
  --config outputs/interaction_ce/train_config.json \
  --output_dir outputs/interaction_ce/inference
```

只有 `completion.json` 中 `finished_all_epochs=true` 才表示完整训练完成；短跑导出是当前进度的模型。

## 6. 初始和最终评测

VSP 空间规划测试已接入同一入口，通过 `--task vsp` 启用；下载、放置、预检和单卡/四卡命令见
[VSP 测试准备与评测](vsp_evaluation_zh.md)。以下 CoMT 命令保持不变。

不使用 TEST 训练或选择学习率。预先固定 max_new_tokens 和评分规则后，评测初始 checkpoint 与各实验的最终模型：

```bash
torchrun --standalone --nproc_per_node=8 --module src.interaction.evaluate \
  --model_path checkpoints/ilvr_comt \
  --test_data_path data/comt/TEST.jsonl --image_root data/comt \
  --output_dir outputs/eval_initial --max_new_tokens 1024

torchrun --standalone --nproc_per_node=8 --module src.interaction.evaluate \
  --model_path outputs/interaction_ce/inference \
  --test_data_path data/comt/TEST.jsonl --image_root data/comt \
  --output_dir outputs/eval_interaction --max_new_tokens 1024
```

另两组替换 `model_path` 和输出目录。每卡处理不同样本，按样本编号合并；生成采用 greedy decoding。
每段强制恰好8或9个连续步骤，不依赖旧全局 batch latent controller 或硬编码 ID。
输出包含 token ID、每段步数、是否被长度上限截断、生成文字和耗时。

`metrics.json` 汇报分任务准确率、整体准确率、任务宏平均和截断数量。
评分复用本仓库 `eval.py` 的 `extract_final_answer` / `normalize_for_match`，属于规则化 exact match；
不声称与外部 LLM judge 的分数等价。所有对照使用相同规则。

## 7. 效率机制、性能测试与显存备选

- 冻结 reference 独立在线运行，每个 GPU 只有一套训练主干和一套冻结 reference decoder。
- 输入图像每批编码一次；reference 不保留视觉 tower 或 LM head 的第二份 GPU 权重。
- 文本按块计算，latent 使用增量 KV；历史 KV 保存为不可变小段，避免每一步保存完整前缀副本。
- 非 reentrant checkpoint 保留主干跨 latent 步的完整梯度。没有通过 detach 主干 cache 换取速度。
- FlashAttention 使用 GQA 和变长序列，排除已结束的 padding 行；fusion 使用 SDPA。
- 只对有效监督位置计算 LM logits，并对分块 CE 做重计算以降低大词表显存。
- ZeRO-3 会跨 rank 对齐执行边界及 LM head 调用次数，避免不等长轨迹导致 gather 次序不一致。

用同一模式、同一有效 batch、同一数据次序比较增量计算和完整前缀重算：

```bash
bash run_interaction_training.sh configs/interaction/interaction_ce.json \
  --execution recompute --max_steps 20 --log_steps 1 --output_dir outputs/bench_recompute

bash run_interaction_training.sh configs/interaction/interaction_ce.json \
  --execution incremental --max_steps 20 --log_steps 1 --output_dir outputs/bench_incremental

python -m src.interaction.benchmark outputs/bench_recompute outputs/bench_incremental \
  --warmup_steps 5 --output outputs/benchmark_comparison.json
```

重算路径仅供正确性/性能对照，长轨迹可能超出显存，遇到这种情况应记录 OOM，不能把更短样本结果当作等价对比。
日志提供跨 rank 最大 step 耗时、reference/主干/反向耗时、有效 CE tokens/s、samples/s 和峰值 allocated 显存。
每个累积窗口末同步一次，以使计时和显存数据可靠。首次加载、数据准备和最终保存时间不混入 GPU 前向耗时。
`--profile_steps 1` 生成各 rank 的 Chrome trace；profiler 有开销，不应与不启用 profiler 的吞吐直接比较。

默认优先 ZeRO-2。显存不够时可显式测试：

```bash
bash run_interaction_training.sh configs/interaction/interaction_ce.json \
  --deepspeed configs/interaction/zero3.json --output_dir outputs/interaction_zero3
```

ZeRO-3 减少主干参数显存，但连续递推会增加参数通信；不承诺比 ZeRO-2 快。
reference 仍为每卡冻结副本。使用前先运行下述 ZeRO-3 smoke test，实际8卡性能需在目标机器测量。

## 8. 验证命令

无需真实 checkpoint 的 CPU 测试（安装环境依赖后）：

```bash
python -m unittest tests.test_interaction tests.test_interaction_qwen tests.test_interaction_checkpoint -v
```

覆盖增量与重算的 hidden/CE/梯度等价、真实 Qwen 前向、图像 mRoPE、真实 processor、
冻结 reference、fusion 梯度、空文本与无 latent 样本、尾批归一化、8/9步生成、HF导出重载以及恢复状态。

Linux CUDA 上的小模型 DeepSpeed 集成检查：

```bash
torchrun --standalone --nproc_per_node=2 --module tests.test_interaction_deepspeed --zero_stage 2
torchrun --standalone --nproc_per_node=2 --module tests.test_interaction_deepspeed --zero_stage 3
torchrun --standalone --nproc_per_node=2 --module tests.test_interaction_deepspeed \
  --zero_stage 2 --attention_backend flash_attention_2
```

集成检查使用不同 rank 的不等长轨迹和零标签尾批，检查有限 loss、参数更新、reference冻结、
跨卡参数一致，以及保存恢复后的下一步与连续训练一致；不会下载7B模型。
本次开发环境的 CPU/Qwen 检查已运行，真实多卡 DeepSpeed、FlashAttention CUDA 和 CoMT 准确率需在上述服务器验证。

## 9. 实现入口与来源

- EMMA 测试：[EMMA 数据准备与评测指南](emma_evaluation_zh.md)，支持官方 mini/完整集、多图输入和独立判分。
- 数据：`src/interaction/data.py`；训练：`src/interaction/train.py`。
- 模型与效率实现：`model.py`、`fusion.py`、`execution.py`。
- 分布式与恢复：`distributed.py`、`checkpoint.py`。
- 推理与评测：`generation.py`、`evaluate.py`；独立导出：`export.py`；性能对比：`benchmark.py`。
- 方法依据：[ILVR 原论文 v3](https://arxiv.org/html/2512.05665v3)，以及你提供的 interaction latent 定稿 PDF。
- [DeepSpeed 配置](https://www.deepspeed.ai/docs/config-json/)与[checkpoint 文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。
