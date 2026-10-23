# SFT I：原图交错 warm-up

在 `ILVR` 目录运行：

```sh
python run_sft1.py
```

所有选项集中在 `configs/sft1/warmup.json`；需要另一个配置时，仅加 `--config 文件路径`。路径相对 **ILVR 根目录**解析，与运行时所在目录无关。这里提供的是代码与配置，当前工作区没有完整正式训练数据和主干权重，必须先填写服务器上的实际路径。

## 先填写的配置

| 字段 | 含义 |
| --- | --- |
| `model_path` | 本地完整 Qwen2.5-VL / ILVR 初始化检查点，默认 `checkpoints/ilvr_comt`；不会联网下载 |
| `sources` | 四套结构化改写结果的 `path`、图片根目录 `image_root` 及字段映射 |
| `prepared_dir` / `output_dir` | 数据缓存和此次训练的独立输出目录 |
| `gpu_ids` | 如 `[0,1,2,3]`；`null` 保留现有 GPU 可见性设置 |
| `num_gpus` | 使用的可见 GPU 数量；`null` 使用全部可见 GPU |

默认 CoMT 读取 `thought_rewritten`，答案字段为 `original_final_answer`；ReFocus 读取完成裁图的 `sequence_plan`，题目/题图/答案字段为 `question/image/answer`。Visual CoT 读取完成 grounding 的 `train.comt.jsonl`，Zebra 读取 v2 的 `hard_rewritten.jsonl`。CoMT 同时支持 hard 文件和混合的 augmented 文件：`skip_missing_plan: true` 明确排除没有 `thought_rewritten` 的记录，`skip_empty_answer: true` 排除原始答案为空的记录；每条排除记录均写入报告。字段名称配错仍报错，不会用旧 `sequence_plan` 替代改写结果。题图和 helper 必须真实存在；已有绝对图片路径迁移到服务器后需要先修正。

结构化文字使用 `analysis.content` 与 `action.original_text`，不把动作标签或元数据当正文。四个源默认 `require_structured_actions: true`：含 helper 的记录必须有结构化动作，避免误读旧的扁平计划。进入训练的序列末尾必须含 final answer，且与配置的答案字段一致。CoMT、ReFocus 已启用 `allow_append_final_answer: true`：末尾没有答案且没有显式 final 标记时，将配置的原始答案字段追加为末尾文字并计算 CE；已有答案不重复追加，原数据文件不修改。显式 final 标记与答案冲突、非法 helper 或演示记录仍报错。Visual CoT、Zebra 默认要求改写计划本身带 final answer。

## 执行与超参

`mode` 为 `train`、`prepare_only` 或 `smoke`。`prepare_only` 只严格校验并缓存数据，不需要 GPU；`smoke` 使用真实训练流程、默认在 2 个 optimizer steps 后保存，可用 `smoke_steps` 调整。建议 smoke 使用单独 `output_dir`。`max_steps` 是总 optimizer step 上限，`null` 按 `epochs` 完整训练；上限不改变按完整 epoch 规划的学习率曲线。

类默认值为微批 1、梯度累积 2、3 epochs、BF16、ZeRO-2；主干 LR `2e-6`，interaction 与 adapter LR `5e-5`，实际以 JSON 为准。`adapter_lr: null` 跟随 `fusion_lr`。`min_image_pixels/max_image_pixels` 控制图像处理，`max_seq_length: 32768` 是完整交错序列上限。短边不足 Qwen patch factor（通常 28）的图片按比例放大，原文件不修改。全局默认 `shrink_overlong_images: true`：仅对超长样本逐步降低图片像素预算，按比例缩放并对齐 Qwen 网格，直到实际完整序列不超过 32768；保留全部 analysis、action 原文、图片顺序和 final answer。不要求恰好填满 32768。每条样本的图片预算写入缓存，在训练时原样复用。

`skip_overlong_samples: true` 是缩图后的兜底：文字本身过长，或图片缩到最低预算仍无法满足上限时，才排除完整样本，并写入 `skipped_records.jsonl`。两项同时为 `true` 时先 resize 再 skip；两项都为 `false` 时超长报错。该策略对 CoMT、ReFocus、Visual CoT、Zebra 全部生效，先前因超长被跳过的样本会在重建缓存时重新尝试缩图。切换 ZeRO 阶段不改变这套策略或序列上限。

预处理默认 `prepare_workers: 4` 个 CPU 进程，`prepare_log_seconds: 5` 秒输出进度；`prepare_workers: 1` 可串行排查。与训练 DataLoader 的 `num_workers` 独立。任务和待写结果数量有上限，结果按原始源/行号顺序流式写出，不把全量 token 序列留在内存。终端显示完成/总数、成功、排除、错误、降分辨率数量、速度、用时和预计剩余时间。数据准备和所有源/图片/权重核查完成后才启动多卡进程，避免预处理触发分布式超时；不要绕过入口直接启动训练模块。

运行期间可查看 `prepared_dir`：

- `progress.json`：启动即创建，持续更新阶段、全局进度和各数据源统计。
- `validation_errors.jsonl`：发现错误即追加；`validation_errors.json` 每次进度更新时保存快照。
- `skipped_records.jsonl`：按配置排除的原始源、行号及原因。
- `resized_records.jsonl`：逐条记录超长样本缩图前后的总长度、视觉/非视觉 token 数，以及每张图的原始尺寸、常规处理尺寸、最终尺寸、token 数和像素预算；尺寸统一为 `[宽, 高]`。同一张图在序列中出现多次也逐次计数。每个数据源的第一个缩图案例会在终端打印摘要。
- `records.partial.jsonl`：持续写入成功结果，定期刷盘；全量通过后才发布为 `records.jsonl` 和 `manifest.json`。

有真实校验错误时仍停止，不将错误记录悄悄丢弃。中断后的 partial 文件可排查，但当前不支持自动从 partial 续跑。`prepare_only` 会重建缓存；成功后改回 `train` 即可复用。

需要 Linux CUDA/BF16 环境、仓库已有的 `requirements-interaction.txt` 依赖和该仓库 Transformers；启动器自动优先加载本仓库 Transformers。`attention_backend` 默认 `flash_attention_2`，无 FlashAttention 环境可改 `sdpa`。

多卡默认使用 **ZeRO-2**（`configs/sft1/zero2.json`），锁定项目依赖中的 **DeepSpeed 0.18.0**。`src/sft1/zero2.py` 将已就绪的梯度按各卡一致的参数顺序交给原生 ZeRO-2 reducer，避免不同 helper 数量造成动态桶错配。它只暂存参数引用，不复制整模型梯度；反传期间持续归约、释放非本卡分片，裁剪、优化器更新和保存仍由 DeepSpeed 完成。完整轨迹只反传一次，文字状态与 KV 的计算图保留。

同时兼容修复 0.18.0 的 ZeRO-2 梯度累积问题：原生 epilogue 用只在累积边界填充的 `averaged_gradients` 判断是否已有梯度，导致之前微批的梯度被覆盖。此处移植 [DeepSpeed 0.18.3 的修复](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.3/deepspeed/runtime/zero/stage_1_and_2.py)，改为检查真实累积缓冲 `all_grad_tensors`，继续使用原生分片布局与生命周期。无需修改安装包或更换依赖版本。

固定顺序优先 LM head、倒序 decoder 层，较小的 interaction/adapter 可以等待；不是等全部梯度就绪后才统一归约。每个微批检查全部参数恰好就绪一次，启动时核对各卡参数名称、形状、组内位置和归约顺序。日志 `zero2_pending_grad_peak_gib` 记录等待排序的梯度量，`peak_allocated_gib` 是实际显存峰值。配置使用 `overlap_comm: false`、`reduce_scatter: true`、`contiguous_gradients: true`，归约/聚合桶为 2000 万元素。BF16 参数使用 BF16 通信，规避该版本超桶参数在混合通信精度下的限制。暂不支持 offload、ZenFlow、DeepCompile、torch_autocast、模型/序列并行。仍可显式选择 `zero1.json`；ZeRO-3 尚未实现。依据 [DeepSpeed 0.18 官方实现](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.0/deepspeed/runtime/zero/stage_1_and_2.py)。

**这次 ZeRO-1 → ZeRO-2 不需要重做 prepare。** 保持 `prepared_dir`、数据源与图片/序列配置，`mode: train` 会直接校验并复用现有版本 4 缓存，只更新启动预检凭据。不要为切换 ZeRO 设置 `prepare_only`。ZeRO-1 的优化器断点不能当作 ZeRO-2 断点直接恢复；若原输出目录已保存 checkpoint，应设置一个新的 `output_dir`，`resume: null`，继续使用原 `prepared_dir`。ZeRO-2 自身的断点可正常续训。

## 模型和监督

```text
题目、题图 → analysis → action.original_text → b_i,1 → b_i,2 → b_i,3 → 完整 helper 图像 tokens → … → final answer
```

仅一个可训练主干；沿 helper 边界分段，保留 KV cache 和文字状态的梯度。第 i 次 interaction 使用已经包含此前 b 的真实前缀，不加载 reference model，也不执行 reference pass。`execution: incremental` 为默认；`recompute` 对同一前缀重算，仅用于验收。

每个微批把题图和 helper 一次编码，重复图片复用同次结果。冻结视觉编码器与 merger，训练 decoder、LM head、2 层 Transformer image adapter 和 interaction。adapter 将变长 helper 图像特征转为 16 个 H 维槽位，由 `adapter_slots` 配置。interaction 的共享 `visual_projection`（W_Z）与拼接 MLP 位于其后；默认拼接 MLP 为 `16384 → 1024 → 3072`，最后重排为 3 个 1024 维 query。每个 query 分别读取局部文字、16 个视觉槽位和题图，使用各自的多图权重及门控，输出 3 个 H 维 interaction token。数量由独立配置 `interaction_tokens: 3` 控制，三个 token 一次生成、按固定顺序插入，并非复制同一个向量。题图单独走多图分支；helper 不进入题图列表。

只计算 analysis、action 原文、final answer 和回答结束符的 CE；提示词、全部 interaction token、图像及边界、padding 不计 loss。三个 interaction token 均占用实际序列位置且参与因果注意力。一个样本完整分段前向后才反传；分布式按整个累积窗口的有效文字 token 总数归一化。没有 latent 回归、蒸馏或额外辅助 loss。

interaction token 数量、图片处理策略、序列长度预算或超长排除策略改变后，旧数据缓存必须重建：将 `mode` 设为 `prepare_only` 运行入口，完成后改回 `train`。当前缓存版本为 4，会拒绝此前缓存，避免旧的全局 skip 结果遗漏可通过缩图保留的样本；旧单 token 模块检查点也不能直接续训为三个 token。

## 保存、续训和后续阶段

`resume` 填 `outputs/sft1_warmup/checkpoints` 可自动选择 latest，也可指定完整 `global_step...` 目录。恢复主干、两个新模块、optimizer、scheduler、每个 rank 的随机数与数据窗口。数据、初始权重、训练语义和 world size 变化会拒绝续训；可调整日志、停止上限等运行控制项。

`output_dir/export/` 包含：

- `backbone/`：HF 格式的 decoder、LM head、冻结视觉模块和 processor。
- `interaction.safetensors` 与 `interaction_config.json`：独立共享核心；JSON 的 `kwargs` 可直接传给 `InteractionCore`。
- `warmup_adapter.safetensors` 与 `warmup_adapter_config.json`：可剥离图像适配器，`kwargs` 对应 `WarmupImageAdapter`。
- `warmup_recipe.json`：训练配置、接口和来源说明。

这是 **warm-up 组件包，不是可直接生成 16 个 latent 的推理模型**。后续去掉 adapter，把真实 16 个自回归 latent 接到 `interaction.visual_projection`，保留主干和 interaction 并继续训练。CE-only 只保证接口可继承，不保证两类表示已经对齐。修改 `adapter_slots` 会改变 adapter 查询和拼接 MLP 的参数形状，旧 8 槽位训练检查点不能直接续训为 16 槽位。

## 验证

单机回归覆盖数据、模块、配置与断点恢复，以及真实小型 Qwen 的分段前向、梯度和 CUDA BF16 图文训练链路。

Linux 多卡环境另提供独立验收入口，无需下载模型或准备正式数据：

```sh
python -m tests.test_sft1_deepspeed
```

GPU 数量等选项位于该文件顶部。此脚本默认依次验证 ZeRO-2、ZeRO-1，使用 FP32 对照单模型处理所有 rank 样本的结果，检查不同 helper 数量、无题图、连续 helper、纯文字及全 mask 样本、全局 CE 归一化、实际更新梯度与动量断点续训。257 元素的小桶会触发多个桶及超桶参数路径，并实际检查各卡梯度就绪顺序不同。该脚本不等同于正式 32768 长度的 BF16 显存压力测试；环境不满足时显示 `skipped`，不视为通过。

另有 `python -m tests.test_sft1_zero2_bf16`，默认 4 卡、BF16、AdamW、8 次累积、裁剪与断点重放。先用独立的普通模型反传/all-reduce 检查分片梯度，再用核验后的梯度独立执行 FP32 master AdamW，避免把 BF16 近零梯度舍入导致的符号变化误判为优化器错误。它也会自动启动 torchrun，参数集中在文件顶部。

本次本地验证已通过原生 DeepSpeed 0.18.0 的 2/4 进程 CPU/Gloo FP32 对照：ZeRO-2 四进程最大参数差约 `5.96e-8`，各 rank 参数一致，断点继续训练一致。另通过四进程 BF16、AdamW、8 次梯度累积和实际触发的梯度裁剪：归约梯度相对 L2 差小于 `0.051%`（BF16 舍入），使用核验后的梯度独立执行 FP32 master AdamW，更新权重完全一致，断点重放逐位一致。真实小型 Qwen 的单卡 CUDA BF16 前后向也通过。尚未在 4×A800 上运行正式模型、32768 长度的 NCCL/FlashAttention 显存压力测试。
