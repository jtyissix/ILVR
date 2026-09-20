# VSP 空间规划测试

本入口评测 ILVR 论文使用的 VSP **spatial planning**，支持官方 ILVR checkpoint、
Interaction-CE、baseline_ct 和 no_refinement 的推理导出。仅添加测试，不加入 VSP 训练数据。
如果模型仅在 CoMT 上继续训练，这次结果属于跨任务泛化测试；不能当作 VSP 专门训练后的论文复现分数。

## 1. 下载测试集

使用 [Mirage 官方发布的 VSP 测试包](https://github.com/UMass-Embodied-AGI/Mirage/tree/53f26de2682025e146781c5b198ec93bdfe4c4d6/data/vsp_spatial_planning)。
只需测试包，不需要训练包或 Mirage 的模型权重。下面命令固定仓库版本和压缩包校验值。
在 Linux 服务器的 ILVR 项目根目录执行：

```bash
mkdir -p data/vsp_spatial_planning
curl -fL --retry 3 \
  https://raw.githubusercontent.com/UMass-Embodied-AGI/Mirage/53f26de2682025e146781c5b198ec93bdfe4c4d6/data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz \
  -o data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz
echo '0272164f29dbed8b191977f7fa71cff8964b701bb84af293bf7525ada8394956  data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz' | sha256sum -c -
tar -xzf data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz \
  -C data/vsp_spatial_planning
```

解压后的关键目录：

```text
ILVR/
├── data/vsp_spatial_planning/
│   ├── vsp_spatial_planning_test.tar.gz
│   ├── test_direct.jsonl
│   └── imgs_test/
│       ├── level3/img/31.png
│       ├── level4/img/...
│       ├── level5/img/...
│       └── level6/img/...
├── checkpoints/ilvr_comt/
└── outputs/interaction_ce/inference/    # 或你另行导出的 inference_2/
```

原始文件有 **400 条**，3×3、4×4、5×5、6×6 地图各 100 条。保留 JSONL 原样。
字段为 `text_input`、`image_input`、`map_id`、`map_desc`；不需要 `sequence_plan` 或 `original_final_answer`。
图像路径类似 `./data/vsp_spatial_planning/imgs_test/level3/img/31.png`。
评测器识别并移除固定仓库前缀，因此 **`--image_root` 指向含 `imgs_test/` 的目录**，
即 `data/vsp_spatial_planning`；整个数据目录也可放到其他磁盘并传入其绝对路径。

## 2. 预检

使用原训练环境及仓库内 Transformers。数据预检不加载模型，不需要 CUDA：

```bash
python -m src.interaction.vsp \
  --test_data_path data/vsp_spatial_planning/test_direct.jsonl \
  --image_root data/vsp_spatial_planning \
  --expected_samples 400
```

检查 JSONL、初始图像存在且可解码、地图形状/格子编码、唯一的起点终点、同尺寸内重复 map_id，
以及 400 条和各尺寸 100 条的计数。坏样本会带行号报错，不会静默跳过。
输出数据 SHA-256、总数及各尺寸计数；TEST 不需要经过训练 token 缓存。

## 3. 运行评测

统一入口新增 `--task vsp`，原 CoMT 命令仍默认 `--task comt`。
每段 latent 步数来自模型自己的 `config.json`：CoMT baseline 通常为 8，Interaction 导出为 9；不要为了换测试集改步数。
新加载器兼容旧导出中的 `_attn_implementation_autoset` 标记，在内存中重置并验证视觉注意力类型；不用手改已有权重目录。

**2026-09-20 prompt 对齐更新：默认使用 `--vsp_prompt_style ilvr_eval`，对齐 ILVR 官方 `eval.py`。**
图像在用户消息开头，随后接未改写的 `text_input`，使用 checkpoint 的 chat template 和
`add_generation_prompt=True`。原文中的字面量 `<image>` 也保留；这正是官方评测代码的行为，
它不是这里额外插入的第二张图片。无需编辑或重新下载 TEST。
旧版把图像移到正文占位符处的布局可通过 `--vsp_prompt_style mirage_marker` 复现。
这两个选项只切换 prompt 构造；生成仍为 greedy、评分仍为 strict v1，便于单独检查 prompt 的影响。

单卡：

```bash
CUDA_VISIBLE_DEVICES=2 python -m src.interaction.evaluate \
  --task vsp \
  --vsp_prompt_style ilvr_eval \
  --model_path outputs/interaction_ce/inference_2 \
  --test_data_path data/vsp_spatial_planning/test_direct.jsonl \
  --image_root data/vsp_spatial_planning \
  --output_dir outputs/eval_vsp_interaction_ilvr_prompt \
  --max_new_tokens 1024 --attention_backend flash_attention_2
```

四卡：

```bash
CUDA_VISIBLE_DEVICES=2,3,5,7 \
torchrun --standalone --nproc_per_node=4 --module src.interaction.evaluate \
  --task vsp \
  --vsp_prompt_style ilvr_eval \
  --model_path outputs/interaction_ce/inference_2 \
  --test_data_path data/vsp_spatial_planning/test_direct.jsonl \
  --image_root data/vsp_spatial_planning \
  --output_dir outputs/eval_vsp_interaction_ilvr_prompt \
  --max_new_tokens 1024 --attention_backend flash_attention_2
```

两种命令选择一种。`inference_2` 按你实际导出目录替换；每张 GPU 加载完整模型，四卡各处理 100 条。
无需 DeepSpeed 配置；单独列出四张可见 GPU 再运行普通 `python` 不会启用四进程。

评测官方 CoMT baseline，使用同样命令并替换：

```text
--model_path checkpoints/ilvr_comt
--output_dir outputs/eval_vsp_initial_ilvr_prompt
```

评测继续训练的 baseline_ct，则换成 `outputs/baseline_ct/inference` 和独立的输出目录。
所有对照使用同一 TEST、相同生成上限、相同评分实现。生成采用 greedy decoding，不使用 reference/fusion。
相同输出目录再次运行会重写结果，不支持中途断点续评。
重测请使用上述新的输出目录。启动后会显示实际 prompt 风格、latent 步数，每完成一题打印进度。

## 4. 评分和结果

模型仅接收 `text_input` 和初始地图 `image_input`；默认图片在前，旧版占位符位置需显式选择 `mirage_marker`。
`map_desc` 只供生成后的评分使用，不放入提示词；不读取 helper 图像、参考动作或训练轨迹。

地图判定沿用 [Mirage 的模拟规则](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/task.py)：

- `1` 为起点，`2` 为终点，`0` 为安全格，`-1` 为洞。
- 越界动作原地不动，进入洞立即失败。
- 执行完整动作序列后位于终点才算正确；经过终点又离开也会失败。
- 成功率不强制最短路径，也不与某一条参考路径做字符串匹配。

答案提取顺序：最后一个完整 `\boxed{...}`，最后一行 `Final answer...`/`Answer:...`，
最后尝试整段输出本身是纯动作序列。支持 `DLLU`、`D, L, L, U`、`DOWN LEFT LEFT UP`。
空答案、截断的 boxed 答案及含未知动作/自然语言的答案计错，仍进入分母。
这里的解析器比旧脚本“从任意文本筛出 U/D/L/R 字母”的做法严格，且使用 greedy decoding；
地图规则相同，但不宣称完整复现 Mirage 的采样/答案解析协议。不同模型应统一使用本入口比较。

输出目录包含：

| 文件 | 内容 |
|---|---|
| `metrics.json` | `samples`、`correct`、`accuracy`、`by_grid_size`、`macro_grid_accuracy`、`invalid_answer_count`、`status_counts`、`token_limit_count` |
| `predictions.jsonl` | 逐题 map_id、预测动作、完整生成文本、模拟状态/最终位置、latent 步数、生成耗时 |
| `predictions_rank*.jsonl` | 各 GPU 的原始结果 |
| `evaluation_config.json` | 参数、prompt 风格、greedy 标记、对齐的官方源码版本、GPU 进程数、latent 步数、实际视觉后端、数据 SHA-256 与计数 |
| `prompt_rank*.json` | 每张卡第一题的实际 messages、模板展开文本、展开图像 token 后的 input_ids、image_grid_thw 和图像路径 |

逐题预测还记录 `prompt_style` 和 `prompt_input_sha256`（input_ids 与 image_grid_thw 的哈希），
避免只凭文件名判断用的是哪种 prompt。该哈希不包含图像像素，不能代替图像文件一致性校验。

```bash
python -m json.tool outputs/eval_vsp_interaction_ilvr_prompt/metrics.json
```

重点比较整体 `accuracy` 和四种尺寸的准确率，并检查无效答案和截断数。
这个入口不新增离散 token、不修改已有模型文件、不参与训练。

## 5. 代码检查

```bash
python -m unittest tests.test_interaction_vsp tests.test_interaction_vsp_audit tests.test_interaction_qwen -v
```

包含地图边界/洞/终点规则、非最短有效路径、多种动作格式、无效/截断答案、路径适配、
防止评分信息进入提示词、评测入口输出，以及旧导出配置的后端重载回归检查。
实际 CUDA/FlashAttention 和模型准确率需在服务器运行上述评测确认。

## 6. 低分和答案解析排查

CoMT checkpoint 直接测试 VSP 是跨任务评测，与论文使用 VSP 训练集的 IID 设置不同。
早先图片位置沿用 Mirage 的原文占位符；新默认布局已对齐 ILVR 原 `eval.py`。
完整调查见 [VSP 低分排查](vsp_failure_analysis_zh.md)，其中低分统计来自更新前的快照。

已有预测无需再调用 GPU，可同时按原严格规则、Mirage 官方 boxed/字母规则和扩展动作语法复核：

```bash
python -m src.interaction.audit_vsp \
  --test_data_path data/vsp_spatial_planning/test_direct.jsonl \
  --predictions outputs/eval_vsp_interaction_ilvr_prompt/predictions.jsonl \
  --output_dir outputs/eval_vsp_interaction_ilvr_prompt/audit
```

使用新输出目录。部分结果需显式加 `--allow_partial`；不同运行的文件不能混合。
扩展解析结果仅用于诊断，不能冒充官方成绩。工具不改写原 `metrics.json` 或预测文件。

## 7. 官方 prompt 核对依据

固定 ILVR commit `13a67ed2ee975d6ca56b4e9eba18d68b352b166b`：

- [eval.py 的 run_one_example](https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/eval.py#L292) 和 [src/evaluate_deepseed.py](https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/src/evaluate_deepseed.py#L200)：图片 content 在前、原文在后，生成模板开启，processor 使用 `padding=True`。
- [论文 v3](https://arxiv.org/pdf/2512.05665v3) PDF 第 12 页（附录页码 2）的 B.2 示例也显示图片在前，但这是通用训练样例，没有给出可替代 TEST 原文的 VSP 专用提示词。
- 官方训练 [src/main.py](https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/src/main.py#L30) 还调用 `place_input_image`：遇到 `<image>` 会把图片移到占位符位置。官方评测不调用它。因此此模式准确名称是 **ILVR eval prompt**，不是声称所有官方训练/推理路径完全一致。

本次使用官方 `shuai22/comt_ckpt` 的 tokenizer、chat template 和 image processor，在 400 条官方
VSP TEST 及真实测试图片上，分别执行两个官方入口到 `processor(...)` 为止的源码，逐条对比新模式。
`rendered_prompt`、`input_ids`、`attention_mask`、`image_grid_thw`、`pixel_values` 均完全一致。
没有载入模型权重或运行 GPU 生成；这证明输入构造对齐，不证明两个生成执行器的输出完全相同，也不保证准确率提升。
