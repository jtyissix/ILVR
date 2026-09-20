# VSP 低分排查（2026-09-20）

后续更新：新入口默认已切换到 `--vsp_prompt_style ilvr_eval`；本文的快照统计及“当前布局”表格
描述的是切换前的版本。重测命令与输入一致性验证见 [VSP 评测文档](vsp_evaluation_zh.md)。

结论：存在提示词布局差异，原有严格解析器也确实拒绝了一些明确的动作表达；但本次保存回复的离线复核中，扩展解析没有新增正确答案。多数失败涉及错误路线、只输出步数，以及沿用 CoMT 删改图像的回答方式。目前没有证据证明仅改 prompt 就能恢复论文成绩。

## 1. 数据范围与实测

本次读取 FinalShell 临时目录中的预测文件，并在 `.cache/vsp-audit/snapshot/` 固定副本。它们不是两份完整的 400 条评测，文件名也不足以确认 checkpoint 身份。以下按记录中的 `segment_steps` 分组，仅用于定位故障，不能作为两个模型的公平成绩比较。

| 分组 | 固定快照文件 | 条数 |
|---|---|---:|
| 8 步 | `rank0(2)`、`rank1(2)`、**`rank2`**、`rank3(2)` | 18 + 6 + 65 + 26 = 115 |
| 9 步 | `rank0`、`rank1`、`rank3` | 54 + 100 + 43 = 197 |

表中的文件均为 `gpus_my_predictions_*.jsonl`。注意不带 `(2)` 的 rank2 是 8 步，不能直接合并到其余同名的 9 步记录中。检查期间这个临时文件还发生过增长，最终以固定副本为准。

使用 Mirage 官方 test archive 中的地图，逐条验证 `index + map_id + grid_size`。原始文件的 `prediction/correct` 与当前严格解析器重算结果全部一致。没有发现本次记录与官方测试地图错配。

| 计分方式 | 8 步快照（115 条） | 9 步快照（197 条） |
|---|---:|---:|
| 当前 strict v1 | 5 / 115，4.35% | 6 / 197，3.05% |
| Mirage 官方 boxed 提取 + 字母过滤 | 0 / 115 | 0 / 197 |
| 扩展明确动作格式，诊断用 | 5 / 115，4.35% | 6 / 197，3.05% |
| strict v1 无法解析 | 12 | 88 |
| 扩展解析后仍无法解析 | 10 | 77 |
| 最终答案只有数字 | 8 | 66 |
| 回复包含字面量 `\boxed{` | 7 | 54 |
| 命中生成长度上限 | 0 | 6 |

扩展诊断支持 `move right`、`D then R`、`Up 2 times`、`3Rights`、`4R1U` 等。新增解析成功的 13 条全部仍未通过地图模拟。它不从推理正文中挑选能走通的路线，也不根据地图修改模型答案，因此不会把评分过程变成解题过程。这不涵盖所有可能的自然语言表达，不能据此声称任意更强解析器都不会改善分数。

本次官方兼容实现与下载的 Mirage 模拟函数、MathRuler boxed 提取函数，在全部 **312 条**保存回复上逐条核对，动作序列和成功判定一致。官方规则只保留 boxed 内容中的 U/D/L/R 字母，不会展开数字次数；未找到完整 boxed 时返回的 `None` 字符串不包含动作。官方得分更低，是因为严格解析器还支持 `Final answer:` 等额外格式，而这批 boxed 内的路线也没有走通。

具体例子：9 步组 index=381 的 `4R1U`，官方得到 `RU`；扩展语义得到 `RRRRU`。该地图起点为零基坐标 `(1,1)`，展开后的最后一个 U 进入 `(0,3)` 的洞，仍然错误。不能把这个例子的低分归咎于拒绝解析数字。

## 2. Prompt 到底对齐了什么

| 环节 | 当前 `src.interaction.evaluate --task vsp` | ILVR 仓库 `eval.py` | Mirage 官方 `src/test.py` |
|---|---|---|---|
| 用户文字 | 官方 `test_direct.jsonl` 原文 | 数据记录中的原文 | 官方测试原文 |
| 图片位置 | 原文 `<image>` 位置 | 图片在用户消息开头，随后附上原文 | `place_input_image` 把图片移到 `<image>` 位置 |
| assistant 前缀 | checkpoint 模板，`add_generation_prompt=True` | 同样使用生成模板 | 手工追加 `<\|im_start\|>assistant` |
| 解码 | greedy | 默认 temperature=0，greedy | sampling，temperature=0.7、top_p=0.9 |

所以当前布局与 **Mirage** 的图片位置一致，与 **ILVR 原推理入口以及本项目 CoMT 训练**的图片在前布局不同。此前如果将其统称为“完全对齐官方”，表述不准确。

进一步核对发现：ILVR 官方 `src/main.py` 的训练 collator 还调用 `place_input_image`，当原文有
`<image>` 时会改变图片位置；两个官方评测入口不调用它。因此不能从 task 层的 content 顺序直接推断
所有官方训练输入都把图放在开头。本项目 CoMT 继续训练仍使用图片在前。

原始 VSP 文字已经要求输出移动计划并放进 boxed，没有要求删除墙壁、删除洞或者仅回答步数。当前输入构造没有暴露 `map_desc` 或 helper 图像。因此并不是完全漏写任务要求。具体服务器 checkpoint 的最终模板文本、processor 配置以及实际启动参数没有随这些预测文件提供，无法只凭回复还原完整 token 输入。

改变图片位置有合理的对照实验价值，但不能仅据代码差异断言它造成了低分。greedy 与 Mirage 的采样差异也应记录，不过当前 greedy 与 ILVR 原脚本默认值一致。更换采样参数后单次得到更好结果，不等于证明 prompt 问题。

## 3. 更强的证据指向训练任务不匹配

读取到的 `gpus_interaction_ce.json` 训练路径是 `data/comt/TRAIN.jsonl`；它没有 VSP 训练数据。当前会话使用的是 CoMT checkpoint，随后继续在 CoMT 上训练。实际导出模型的精确训练来源仍应以它对应的 `train_config.json` 为准。

ILVR 论文 §4.1 将 VSP 列为 IID 评测，附录表 8 给出 **1,000 训练 / 400 测试**。论文 Qwen2.5-VL 的 81.5% 对应其 Stage 2 IID 设置；CoMT checkpoint 直接测 VSP 是不同的跨任务设置，不能把 81.5% 当作这个 checkpoint 的复现目标。见[论文 §4.1 和附录 C](https://arxiv.org/html/2512.05665v3)。

8 步快照中 108/115 条、9 步快照中 184/197 条回复含 `delete/deleted/deleting/deletion` 词形。人工抽查可见“删除墙壁/洞”“裁剪图像”“计数”等 CoMT 式推理。尤其 9 步组有 66 条最终只给数字。词频只是诊断信号，不等同于每条都已确认同一种故障，但结合回复内容，更支持任务行为迁移不良的解释。8 步组同样存在这些现象，不能把全部问题直接归于 interaction 机制。

当前临时训练配置还设置了 10 epochs、backbone LR=1e-5，超过原计划的 3 epochs、2e-6。它可能加强 CoMT 专化；目前没有 checkpoint 对照或训练曲线证明过拟合，不能据此直接下结论。

## 4. 无需 GPU 的复核方法

已新增独立入口，保留原始预测文件和原评测默认规则。它同时报告三种规则，扩展规则明确标为诊断结果：

```bash
python -m src.interaction.audit_vsp \
  --test_data_path data/vsp_spatial_planning/test_direct.jsonl \
  --predictions outputs/eval_vsp_interaction/predictions.jsonl \
  --output_dir outputs/eval_vsp_interaction/audit
```

也可以给 `--predictions` 传同一次运行的多个 `predictions_rank*.jsonl`。不要同时传合并文件与分片文件；不要混合不同 checkpoint 的输出。工具拒绝重复 index、地图身份错配和混合 latent 步数。相同 latent 步数仍不保证来自同一模型，需要使用者保证文件来源。

默认要求覆盖所给 TEST 的全部记录。分析运行中的固定副本时显式加 `--allow_partial`；残缺的 JSON 行会报错，不静默丢弃。输出目录必须是新目录。

- `audit_metrics.json`：三个评分口径、覆盖率、恢复/丢失的正确样本、原评分一致性、输入 SHA256。
- `rescored.jsonl`：逐条动作、地图模拟结果和失败原因。

不需要加载 checkpoint、读取图片或启动 torchrun。现有数据目录已有 TEST JSONL 即可。

## 5. 后续实验顺序

1. 先对服务器上每个模型的完整 400 条输出运行离线审计，确认精确 checkpoint 和配置；临时下载副本不能代替完整结果。
2. 固定同一个 checkpoint、样本、greedy 和生成长度，比较 ILVR 图片在前与当前占位符布局。做 prompt 选择时使用 VSP 训练集划出的开发子集；最终 TEST 留作一次锁定配置的评测。
3. 固定相同输入 token，比较原生 `eval.py` 与新连续执行器，才能区分 prompt 差异和推理实现差异。本次只有 CPU 离线计分，没有执行服务器模型对照，尚不能排除其他推理差异。
4. 如果目标是论文的 VSP IID 成绩，需要明确加入 VSP 训练，或使用经确认在 VSP 上训练的 checkpoint。是否改变训练范围应单独决定；本次没有恢复此前取消的 VSP 训练工作。

## 源码与验证

- [Mirage test.py](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/test.py)、[task.py](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/task.py)、[utils.py](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/utils.py)。
- [ILVR eval.py](https://github.com/XD111ds/ILVR/blob/main/eval.py)、[task_deepseed.py](https://github.com/XD111ds/ILVR/blob/main/src/task_deepseed.py)。
- [MathRuler boxed 提取](https://github.com/hiyouga/MathRuler/blob/main/mathruler/grader.py)：本次源码快照 SHA256 `dbc8a73cf48e3a449c52125218e1400d616186549bea36fe31b2fe0b495e3eff`。Mirage 没有在该入口锁定 MathRuler 版本，因此报告指定了本次验证的实现。
- 官方 TEST JSONL SHA256：`3a0868f2731edba7eada1f2895a8509ade4f1ce3d5d984376124d6e60ae02971`。
- VSP 相关 12 项 CPU 测试通过，覆盖官方提取语义、扩展次数、拒绝正文猜测、最终答案优先、地图身份、重复和缺失样本、不同运行混合；312 条真实回复与官方提取/模拟函数一致。
