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

类默认值为微批 1、梯度累积 2、3 epochs、BF16、ZeRO-1；主干 LR `2e-6`，interaction 与 adapter LR `5e-5`，实际以 JSON 为准。`adapter_lr: null` 跟随 `fusion_lr`。`min_image_pixels/max_image_pixels` 控制图像处理，`max_seq_length: 32768` 是完整交错序列上限。短边不足 Qwen patch factor（通常 28）的图片按比例放大，原文件不修改。全局默认 `shrink_overlong_images: true`：仅对超长样本逐步降低图片像素预算，按比例缩放并对齐 Qwen 网格，直到实际完整序列不超过 32768；保留全部 analysis、action 原文、图片顺序和 final answer。不要求恰好填满 32768。每条样本的图片预算写入缓存，在训练时原样复用。

`skip_overlong_samples: true` 是缩图后的兜底：文字本身过长，或图片缩到最低预算仍无法满足上限时，才排除完整样本，并写入 `skipped_records.jsonl`。两项同时为 `true` 时先 resize 再 skip；两项都为 `false` 时超长报错。该策略对 CoMT、ReFocus、Visual CoT、Zebra 全部生效，先前因超长被跳过的样本会在重建缓存时重新尝试缩图。保持 ZeRO-1，未增加序列上限。

预处理默认 `prepare_workers: 4` 个 CPU 进程，`prepare_log_seconds: 5` 秒输出进度；`prepare_workers: 1` 可串行排查。与训练 DataLoader 的 `num_workers` 独立。任务和待写结果数量有上限，结果按原始源/行号顺序流式写出，不把全量 token 序列留在内存。终端显示完成/总数、成功、排除、错误、降分辨率数量、速度、用时和预计剩余时间。数据准备和所有源/图片/权重核查完成后才启动多卡进程，避免预处理触发分布式超时；不要绕过入口直接启动训练模块。

运行期间可查看 `prepared_dir`：

- `progress.json`：启动即创建，持续更新阶段、全局进度和各数据源统计。
- `validation_errors.jsonl`：发现错误即追加；`validation_errors.json` 每次进度更新时保存快照。
- `skipped_records.jsonl`：按配置排除的原始源、行号及原因。
- `resized_records.jsonl`：逐条记录超长样本缩图前后的总长度、视觉/非视觉 token 数，以及每张图的原始尺寸、常规处理尺寸、最终尺寸、token 数和像素预算；尺寸统一为 `[宽, 高]`。同一张图在序列中出现多次也逐次计数。每个数据源的第一个缩图案例会在终端打印摘要。
- `records.partial.jsonl`：持续写入成功结果，定期刷盘；全量通过后才发布为 `records.jsonl` 和 `manifest.json`。

有真实校验错误时仍停止，不将错误记录悄悄丢弃。中断后的 partial 文件可排查，但当前不支持自动从 partial 续跑。`prepare_only` 会重建缓存；成功后改回 `train` 即可复用。

需要 Linux CUDA/BF16 环境、仓库已有的 `requirements-interaction.txt` 依赖和该仓库 Transformers；启动器自动优先加载本仓库 Transformers。`attention_backend` 默认 `flash_attention_2`，无 FlashAttention 环境可改 `sdpa`。

多卡固定使用 **ZeRO-1、`overlap_comm: false`**，在反传结束后按固定参数顺序归约。不同样本的 helper 数量使梯度产生顺序不同，普通 ZeRO-2 的动态桶可能错配；因此拒绝 ZeRO-2，ZeRO-3 的模块调用对齐也尚未实现。ZeRO-1 仍分片 optimizer 状态，但每卡保留完整梯度，显存需求高于 ZeRO-2。配置还锁定 `reduce_scatter: false`、`contiguous_gradients: true`，暂不支持 optimizer offload、ZenFlow 或 DeepCompile。依据 [DeepSpeed 0.18 官方实现](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.0/deepspeed/runtime/zero/stage_1_and_2.py)。

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

GPU 数量等选项位于该文件顶部。此脚本使用 FP32，对照单模型处理所有 rank 样本的结果，检查不同 helper 数量、空样本、全局 CE 归一化、ZeRO-1 更新及断点续训。当前工作区尚未执行多卡验收；环境不满足时显示 `skipped`，不视为通过。
