# 第三阶段：Latent Relaxation

入口为 `python run_sft3.py`，配置为 `configs/sft3/relaxation.json`。
接第二阶段 `run_sft2.py` 或 `run_sft2_offload.py` 的 export，训练参数只有 backbone 的 decoder、输入 embedding 和 LM head。
视觉编码器保持冻结，仅编码问题图片；不实例化或加载 interaction module、辅助 decoder、warmup adapter、teacher。

## 训练序列与 CE

复用第二阶段已经分词的 student 序列及其 4/3/16 布局：

```text
analysis text -> <|latent_start|>
              -> text/action latent × 4
              -> interaction latent × 3
              -> visual latent × 16
              -> <|latent_end|> -> analysis / final answer
```

第二阶段允许的独立 action 块（4 个）和独立 interaction+visual 块（3+16 个）也保留。
每个 latent 位置都取 `input_embedding[t] = final_hidden[t-1]`，包括 3 个 interaction latent。
连续状态和 KV cache 都不 detach，后续文本 CE 可回传整个 latent 链。
所有 latent 槽的 label 都是 `-100`；问题提示、问题图片和 padding 同样不算 CE。
普通 assistant 文本、最终答案、EOS 和离散的 `<|latent_start|>` / `<|latent_end|>` 边界标记仍计算 CE。

这里的 text latent 指第二阶段压缩 action 原文得到的 4 个连续向量，不是普通文本 token 的 hidden states。
原 action 文本不再作为额外 CE/回归目标。没有 visual alignment、action-word CE 或 action-state regression。
这个训练方式对应 [ILVR §3.3 的 Text-Only Supervision with Latent Relaxation](https://arxiv.org/html/2512.05665v3#S3.SS3)。

## 防止 latent collapse

参考 [ProLaViT §3.4，式 (7)–(9)](https://arxiv.org/pdf/2607.02907)，按**同一样本、同一 latent 类型的不同推理步骤**比较均值向量。
不在不同样本之间计算相似度，也不直接惩罚同一步内所有 token 的相似度。

对类型 `c`，设该样本有 `K` 次该类型的 latent 块，按发生顺序编号：

```text
g[c,i] = mean(latent_inputs[c,i], dim=tokens)
w[i,j] = epsilon + (1 - epsilon) * (abs(i-j) / (K-1)) ** alpha
pair_loss[c,i,j] = w[i,j] * max(0, cosine(g[c,i], g[c,j]) - margin), i < j
```

text/action 每步平均 4 个向量，interaction 每步平均 3 个，visual 每步平均 16 个。
`K < 2` 的样本在该类型上的损失和步骤对数都为 0；不会跨样本配对。
各类型先对所有 rank 和整个梯度累积窗口的有效步骤对取平均，再对选中的类型取平均：

```text
L = CE_sum / global_text_token_count
  + collapse_weight / number_of_selected_types
    * sum(global_pair_loss_sum[c] / global_pair_count[c] for selected c)
```

若某个已选择类型在整个窗口没有有效步骤对，它的均值记为 0，选择类型的分母保持不变。
梯度归一化补偿 DeepSpeed 的 rank 平均和梯度累积，不会让短序列、dummy 样本或 micro-batch 划分改变目标权重。
均值和余弦用 FP32 计算，支持 BF16 backbone。

在 JSON 中设置：

```json
{
  "collapse_targets": "123",
  "collapse_weight": 0.2,
  "collapse_margin": 0.8,
  "collapse_alpha": 1.0,
  "collapse_epsilon": 0.1
}
```

| collapse_targets | 计算位置 |
| --- | --- |
| `0` | 完全关闭，不调用防塌缩损失 |
| `1` | text/action latent |
| `2` | interaction latent |
| `3` | visual latent |
| `12` | text + interaction |
| `13` | text + visual |
| `23` | interaction + visual |
| `123` | 三类都启用 |

数字或字符串均可，顺序不影响选择（如 `21` 等价于 `12`）；重复数字、夹带 `0` 或其他字符会报错。
`collapse_targets=0` 时，无论 `collapse_weight` 多大都不启用；`collapse_weight=0` 也会跳过计算。
默认 selector 是 `0`，系数预设 `0.2`。

论文使用可学习的 margin 和距离指数。这里为遵守“只训练 backbone”，二者作为固定超参数：
`margin∈[0,1]`，`alpha>=1`，`epsilon∈[0,1]`，不添加任何额外可学习参数。
`margin=0.8`、`epsilon=0.1`、系数 `0.2` 来自论文；固定 `alpha=1.0` 是本实现的默认选择。

## 路径与启动

先修改 `configs/sft3/relaxation.json`：

```json
{
  "stage2_path": "outputs/sft2_latent/export",
  "prepared_dir": "data/sft2/prepared",
  "output_dir": "outputs/sft3_relaxation",
  "collapse_targets": "23",
  "collapse_weight": 0.2
}
```

以上是需要编辑的字段示例，完整文件已经提供。相对路径均相对于 ILVR 根目录解析。

- `stage2_path` 指向包含 `backbone/`、`stage2_recipe.json` 的第二阶段 export，不是 DeepSpeed checkpoint 分片目录。
- `prepared_dir` 指向第二阶段的 `manifest.json`、`records.jsonl`。本阶段直接复用分词缓存，不重新处理原始 sources，也不需要 teacher cache。
- 保留缓存中的问题图片路径可访问。helper 图片可以不存在；读取记录时即删除 helper、teacher trajectory、action target 和 interaction 的 helper/text 引用，送入视觉编码器的只有问题图片。
- `min_image_pixels`、`max_image_pixels` 和 tokenizer 必须与缓存一致；不重新缩放或截断第二阶段的 token 序列。
- 使用不同的 `output_dir` 保存不同 selector/系数组合的实验。

在 ILVR 目录执行：

```bash
python run_sft3.py
# 或
python run_sft3.py --config configs/sft3/relaxation.json
```

启动器先做 CPU preflight，核对缓存、问题图片、tokenizer 和 backbone，再启动分布式训练。
没有任何 teacher 生成进程。
`mode="prepare_only"` 仅验证输入，不需要 GPU；`mode="smoke"` 运行 `smoke_steps` 个真实 optimizer steps。
常规训练需要 BF16 CUDA，沿用前两阶段的依赖和仓库自带 Transformers。
支持现有 ZeRO-1 / ordered ZeRO-2（默认），ZeRO-2 沿用仓库对 DeepSpeed 0.18.0 的版本约束。
`execution="incremental"` 使用可微 KV cache；`"recompute"` 可用于数值对照。

## 保存、日志与恢复

`metrics.jsonl` 包含 `ce_loss`、`collapse_text_loss`、`collapse_interaction_loss`、`collapse_visual_loss`、
`weighted_loss` 和各项 `loss_counts`。未选择类型为 0；count 为 0 表示该窗口没有该类型的有效步骤对。

`output_dir/checkpoints/` 保存 backbone、optimizer、scheduler、各 rank 的 RNG 和下一训练窗口。
将 `resume` 设置为 checkpoint 根目录或某个 `global_step...` 子目录可以精确恢复。
改变 selector、系数、margin、alpha、epsilon、数据、初始 backbone、world size 或训练语义会拒绝断点恢复。

`output_dir/export/` 只导出 `backbone/` 和 `stage3_recipe.json`。
backbone 包含训练后的 decoder/LM head 和冻结的视觉编码器，不包含 interaction 或辅助 decoder。
推理仍需按 4/3/16 布局驱动连续 hidden-state 回填；普通 HF `generate()` 不会自动执行本阶段的 latent 调度。
现有 vendored Transformers 的旧 `stage3` 路径也不是这个新训练入口。

## 验证

在已配置仓库自带 Transformers 的环境中：

```bash
python -m unittest tests.test_sft3_model tests.test_sft3_data_checkpoint tests.test_sft3_qwen -v
```

覆盖 selector/系数、距离加权公式、各类型梯度、text-only CE、latent 连续回传、dummy/变长批次、
不同执行模式的数值与梯度、仅问题图片的数据路径、checkpoint 恢复、backbone 导出、启动顺序，
以及真实 Qwen 层与独立逐 token 原生前向的数值和梯度对照。
