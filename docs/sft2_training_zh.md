# 第二阶段：action 4 + interaction 3 + visual 16

本实现使用已确认的单路径训练语义：

```text
题目与题图 → analysis → <|latent_start|> → 4 个自回归 action latent
                     → 3 个 interaction token（读取该 helper 的真实 pooled 特征先验）
                     → 16 个自回归 visual latent → <|latent_end|> → 下一段 analysis / final answer
```

真实 helper 特征既是 interaction 的先验，也是 visual latent 的监督目标。这是有 helper 条件的训练设定，当前实现保留这一信息通路。它不声称实现只给题图、无需 helper 的自由推理。导出文件也明确要求 helper images 与对应的事件计划。

标准 action/helper 对由一组 start/end 包住，内部为连续的 4+3+16 个向量，没有 action 与 interaction 之间的换行，共占 25 个序列位置。start/end 是真实特殊 token，参与下一词 CE；23 个内部槽位的 labels 仍为 -100。首个 action latent 使用 start 的输出 hidden，最后一个 visual latent 的输出用于预测 end。

数据中没有紧接 helper 的独立 action 使用 `start → action×4 → end`；没有紧邻前置 action 的 helper（包括连续 helper）使用 `start → interaction×3 → visual×16 → end`。这些情形保留原始事件顺序，不凭空补 action。边界 token 不进入 interaction 的局部文字 memory，四个 action 输出仍进入 memory。

## 从预热权重继承

第二阶段 processor 注册 `<|latent_start|>` 与 `<|latent_end|>`；已有 token 复用原 ID 和权重。缺失的 token 按固定顺序追加，学生按需扩展词表；新 token 对应的输入 embedding 和 LM-head 行用原矩阵均值初始化，包括模型预留的词表行。辅助 decoder 在词表迁移后复制，导出会保存更新后的 tokenizer、词表尺寸和边界 ID。第一阶段导出和冻结 teacher 不增加边界、不改变权重。

`warmup_path` 指向第一阶段的 `output_dir/export/`，其中必须包括 `backbone/`、`interaction.safetensors`、两个模块配置以及 `warmup_recipe.json`。学生继承主干与完整 `InteractionCore`，包括共享 `visual_projection`、16 槽位有序拼接 MLP、文字/latent/题图分支与输出投影；学生不加载 `WarmupImageAdapter`。

冻结的离线 teacher 仍使用完整第一阶段组件（包括旧 adapter），重建原始文字与完整 helper 的预热轨迹，取得消费完每个 `action.original_text` 的输出隐状态。缓存绑定预热权重、数据与图像身份。不要删除用于生成 teacher 的第一阶段 adapter 文件。

辅助 decoder 从当前预热 backbone 的 Qwen2.5-VL 语言 decoder 和 LM head 初始化，使用独立的层、norm 与 head；这里是 Qwen2.5-VL 的语言 decoder，不是另一套 Qwen2-VL。它没有视觉编码器，也没有第二份词 embedding。输入文字 embedding 由学生提供，辅助 decoder 的层、norm 与独立 LM head 训练并保存。

## 数据与四项 loss

第二阶段必须从保留结构的原数据重建缓存。第一阶段缓存已经把 `analysis.content` 与 `action.original_text` 展平成文字，不能恢复动作边界和 `action.content`。第二阶段分别保存普通文字、4 个 action latent 位置、16 个 visual latent 位置、3 个 interaction 位置、helper 映射与辅助动作词目标。

teacher 按 analysis/action 的显式边界分段 tokenize，并将 `TOKENIZATION_POLICY` 绑定到缓存身份，避免 BPE 跨边界合并导致动作终点不明确；因此不复用第一阶段平铺后的 token 缓存。

局部 interaction memory 显式包含普通文字及 4 个 action latent 的已计算输出隐状态。它不再根据 CE label 选择 memory，因此 label 为 `-100` 的 action latent 也能参与注意力。

总目标为：

```text
ce_weight × L_ce
+ visual_weight × L_visual
+ action_words_weight × L_action_words
+ action_state_weight × L_action_state
```

- `L_ce`：保留的 analysis、final answer、latent start/end 与结束符的下一词 CE；连续 latent 与 interaction 不作为 CE 目标。
- `L_visual`：16 个自回归 visual latent 输入向量与 helper 的 4×4 空间 pooling 特征逐槽 cosine loss。
- `L_action_words`：辅助 decoder 消费 4 个 action latent 后预测 `action.content` 及结束符的 CE。
- `L_action_state`：消费第 4 个 action latent 后的主干输出 hidden，与冻结 teacher 消费原 `original_text` 后的输出 hidden 对齐，使用按 teacher 均方能量归一的均方误差。

四项 loss 各自按整个梯度累积窗口、所有数据并行 rank 的有效计数归一化：分别是文字目标 token、visual 槽位、action-word token、action 个数。空事件与分布式补齐样本贡献零；各项加权后每个微批仅反传一次。日志分别记录四项均值、计数及加权和，避免动作稀疏时把所有项除以文字 token 数。

## 启动

2026-10-27 的边界版本将 prepared cache 升为 v2，并更新 checkpoint 格式。旧的无边界第二阶段缓存必须先用 `mode: "prepare_only"` 重建，再切回 `smoke` 或 `train`；teacher 缓存会按新的记录身份重新生成。旧第二阶段 checkpoint 不能直接 resume 到本版本，应从第一阶段 export 初始化，设置 `resume: null` 并使用新的输出目录。第一阶段权重本身无需重训。

修改 `configs/sft2/latent.json` 的 `warmup_path`、四个数据源路径和输出目录，然后从 ILVR 目录执行：

```bash
python run_sft2.py
```

启动顺序固定为：CPU 多进程数据检查/缓存 → 多卡并行离线 teacher 缓存 → 所有 teacher 进程退出并校验合并缓存 → 启动分布式学生训练。前两阶段不创建 NCCL 进程组，耗时预处理不会占用学生训练的 NCCL 初始化等待窗口。

CPU 预处理与 warm-up 使用相同的 spawn 进程池和有界任务队列，逐样本的结构检查、tokenization、图像处理与长度预算并行执行。`prepare_workers` 控制进程数，默认 4；目录统计、来源指纹和最终有序写入由主进程完成。它与训练 DataLoader 的 `num_workers` 是两个不同参数。

teacher 默认使用与学生训练相同的 `gpu_ids` / `num_gpus`：每张选定 GPU 一个独立推理进程，各自加载完整冻结的阶段一 teacher；按原始 teacher 轨迹长度估计计算量分配样本，单个样本的完整历史始终留在同一张卡。缓存内容是每个 action 的 `original_text` 最后一个 token 的输出 hidden，文件张量形状为 `[该样本 action 数, hidden_size]`，用于 `L_action_state`，不缓存全部 token 的 hidden 或 KV。可复用的缓存和无 action 样本不启动 GPU 推理，待处理样本少于卡数时只启动必要的进程。

每个 GPU 写独立进度日志，协调进程验证分片归属、覆盖范围和 tensor 校验和后才发布完整 manifest。任一进程失败会终止并回收其它进程，启动器不会继续学生训练；重启时复用已完成且校验通过的样本，允许更换 GPU 数量。多卡仅并行处理不同样本，不使用 DDP/NCCL，也不将一套 teacher 的参数切分到多卡。

并行缓存回归使用两个真实 spawn 进程和确定性 CPU 测试目标，验证分片结果与串行一致、失败回收、改变卡数后续算及缓存修复；实际小型 Qwen 的单卡 CUDA BF16 teacher→学生更新→导出流程也已通过。物理多 GPU 上的 7B teacher 并行吞吐与显存尚未实测。测试入口：`python -m unittest tests.test_sft2_teacher_parallel tests.test_sft2_teacher tests.test_sft2_training -v`。

- `mode: "prepare_only"` 只准备结构化数据，不需要 GPU、不生成 teacher 隐状态。
- `mode: "smoke"` 使用真实模型与真实数据短训，默认两个 optimizer steps，仍需要已准备或可生成的 teacher 缓存。
- `mode: "train"` 正常训练，复用验证通过的数据和 teacher 缓存。
- 首次从阶段1迁移时 `resume: null`，使用独立 `output_dir`；阶段1 optimizer 不能作为阶段2 optimizer 恢复。
- 阶段2中断后，`resume` 可指向本阶段 `checkpoints/` 或具体完整 step 目录。

显存需求包括可训练主干、辅助 decoder、interaction、冻结视觉编码器、优化器与全部轨迹计算图。teacher 缓存阶段每卡一套冻结模型，所有独立进程退出后才加载学生，避免训练时额外常驻第三套 decoder。正式 7B、多卡、最长序列的显存上限需要在目标训练机器实测。

## 分布式、保存与验证

复用已锁定 DeepSpeed 0.18.0 的 ZeRO-2 固定归约顺序与累积修复。第二阶段显式优先归约辅助 LM head、辅助 decoder 倒序各层，再归约主干与较小模块，避免把整个辅助 decoder 的已就绪梯度留到最后。各 rank 校验的顺序指纹仍包含参数名、形状、dtype、optimizer 组和组内位置；第一阶段的默认顺序保持不变。

完整训练 checkpoint 保存主干、interaction、辅助 decoder、optimizer、scheduler、每个 rank 随机数与数据窗口。恢复签名覆盖四项 loss 权重、数据、teacher 缓存、预热主干与 interaction、训练配置和 world size。

`export/` 包括：

- `backbone/`：主干、LM head、冻结视觉编码器和 processor。
- `interaction.safetensors` 与配置。
- `auxiliary_decoder/`：辅助 decoder 层、norm、head；使用主干输入 embedding 的组件包。
- `stage2_recipe.json`：训练语义和 helper/计划依赖。

这些组件不等同于可直接调用普通 HF `generate()` 的完整条件生成器。`backbone/config.json` 的 `stage` 保留为旧框架的安全主干加载模式，第二阶段语义由独立 `stage2_recipe.json` 标识，避免误触仓库原有另一套 `stage2` forward。

CPU 组件测试可执行：

```bash
python -m unittest tests.test_sft2_training tests.test_sft2_checkpoint -v
```

这些测试验证独立 loss 归一化、辅助参数归属与顺序、完整状态恢复和导出约束，不代替真实预热权重、GPU/BF16、FlashAttention 或多卡 NCCL 验收。

原生多进程 ZeRO-2 验收入口为 `python -m tests.test_sft2_deepspeed`（默认两张 CUDA 卡），CPU/Gloo 对照使用 `python -m tests.test_sft2_deepspeed --cpu`。它实际执行 `Stage2Model`、四项 loss、辅助 decoder、原生 DeepSpeed 0.18.0 的分片归约及 optimizer，并在不等 action/helper 数量、连续 helper、纯文字、补齐样本和两次梯度累积下，与独立单模型处理所有 rank 样本的结果比较，检查完整 checkpoint 续训。解码器是随机初始化的小型可微测试模型，不下载 7B 权重。缺少环境时输出 `skipped`，不得视为通过。

初始无边界版本曾合并通过 56 项测试，包括数据、teacher 缓存、实际小型 Qwen 的 CUDA BF16 四 loss 前后向、阶段1导出到阶段2的图像处理/teacher/学生更新完整流程，以及阶段1原有回归。

随后新增并通过训练入口回归：实际调用 `train()` 至 CUDA 启动边界，确认完整 teacher 身份校验先于训练内存中原教师轨迹的释放。

边界变更前，两进程 CPU/Gloo FP32 验收已通过：三次更新的最大参数差约 `1.19e-7`，首步有效梯度最大差约 `5.17e-6`；各 rank 参数完全一致，含辅助 decoder/optimizer 动量的断点续训一致。257 元素桶实际覆盖多桶和大参数路径，且观察到不同 rank 的梯度就绪顺序。尚未使用真实 7B 预热权重验收，也未完成 NCCL/多 GPU 或 32768 长上下文的显存验证。

2026-10-27 边界版本已通过数据/模型/teacher/词表/导出恢复与阶段一数据兼容回归。验证包括：25 位置布局、边界 CE 及反向梯度、缺失边界拒绝、旧缓存拒绝、预留词表行与扩展词表两种迁移、独立 action/连续 helper，以及实际小型 Qwen CUDA BF16 训练。新增的完整流程还验证了训练后导出并重新加载时，边界 token、词表大小和 embedding 权重保持一致。
