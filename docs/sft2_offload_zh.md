# SFT-II optimizer CPU offload

这是独立训练入口。原 `run_sft2.py`、`src/sft2`、`src/sft1` 和原配置保持不变。模型、interaction 主路径、23 个连续槽位及 start/end、四项 loss、参数分组和学习率规则均复用原 SFT-II 实现。只把 ZeRO-2 的优化器状态和更新放到 CPU；GPU 仍保存完整 BF16 模型权重并执行前反向。

## 启动与缓存复用

编辑 `configs/sft2_offload/latent.json`，核实它的 `warmup_path`、`sources`、`prepared_dir`、`teacher_cache_dir` 与已经完成准备的训练配置一致。默认 prepared 仍为 `data/sft2/prepared`，teacher_cache_dir 为 null 时仍指向其中的 `teacher`。不要把它们改到 offload 输出目录。

```bash
python run_sft2_offload.py --config configs/sft2_offload/latent.json
```

独立 DeepSpeed 配置是 `configs/sft2_offload/zero2.json`，独立输出是 `outputs/sft2_offload`。GPU 数量默认 null，使用 Slurm 实际分配的可见卡。原配置文件继续用于原训练入口，不能混用。

启动顺序：CPU 数据校验/必要时并行准备 → CPUAdam 编译与一次 CPU 更新自检 → 原 teacher 缓存校验/仅补缺失条目 → teacher 进程全部退出 → torchrun 分布式 offload 学生训练。CPUAdam 首次编译也放在 NCCL 初始化之前。

已有 prepared manifest 时只校验，不重新 tokenize，不重写 records/manifest；预处理参数不匹配则报错，避免覆盖共享缓存。teacher 使用原实现和原身份规则，完整有效的缓存不会重新做 GPU 推理。GPU 数量、micro_batch_size、梯度累积、优化器 offload 不改变这两类缓存的身份。源数据、图片、warmup 权重、相关 teacher 代码或图像/tokenization 参数改变则不能保证命中。

原 GPU 优化器 checkpoint 不能直接恢复进此分支；初次使用 `resume: null`，从同一个 stage-I export 初始化。新分支自己的完整 checkpoint 支持在相同卡数和训练设置下恢复。prepared/teacher 缓存与优化器 checkpoint 是不同的文件。

## 运行依赖和 CPU 资源

继续使用 `requirements-interaction.txt` 固定的 DeepSpeed 0.18.0、Linux/CUDA/BF16 环境。CPUAdam 需要可用的 C++ 编译工具链及 CPU 指令支持。CPU 自检失败会在启动 teacher 和学生 NCCL 之前终止。

新分支保留固定顺序梯度归约，使用 native CPU offload epilogue。DeepSpeed 0.18.0 的 offload 中间累积只在 GPU 相加、未回写 CPU，本分支对单个 optimizer 实例补上回写，确保 GAS>2 不丢中间 microbatch；不修改已安装的 DeepSpeed，也不修改旧训练分支。

默认 `pin_memory=true`。主机内存不足时不能依赖 swap。以约 14.7B 可训练参数为例，全节点的模型状态/缓冲估算如下；实际数量会打印到启动日志和 `run_info.json` 的 `memory_budget`：

| CPU 项目（全节点合计） | 约 GiB |
|---|---:|
| FP32 主参数 + Adam 两份动量 | 164 |
| FP32 梯度分片 | 55 |
| BF16 参数更新缓冲 | 27 |
| GAS>1 时 BF16 累积缓冲 | 27 |
| 合计，不含数据和加载/保存峰值 | 274 |

还需每 rank 暂存区、跨分片边界的整参数冗余、模型加载/导出峰值、数据进程和预取窗口。建议节点内存从 512 GiB 起评估；8 卡、较多 DataLoader workers 应留更大余量（如 640/768 GiB），最终以实际进程内存峰值为准。这不是最低硬件保证。

原来的 20 个 CPU 核需同时供多个 CPUAdam 进程和 DataLoader 使用，可能限制速度。可从每 rank 4 个优化器线程，加数据加载余量起测，例如 6 卡 32/48 核、8 卡 48/64 核；按集群可用资源选择。启动器保留 `OMP_NUM_THREADS=4` 默认值，可以在 Slurm 脚本中先设置它。不要让每个 rank 都占用节点全部 CPU 核。若主机内存或 CPU 紧张，可将新配置的 `num_workers` 从 4 调至 1 或 2，保持缓存路径不变。

## GPU 显存与 batch size

以下按可训练参数 P≈14.7B、BF16、当前完整辅助 decoder 估算，单位 GiB，不是 7B 多卡实测。

| 项目 | 4 卡每卡 | 6 卡每卡 | 8 卡每卡 |
|---|---:|---:|---:|
| GPU 完整 BF16 训练权重，2P | 27.4 | 27.4 | 27.4 |
| 移到 CPU 的主参数和 Adam，12P/N | 41.1 | 27.4 | 20.5 |

GPU 还包含约 1.3 GiB 冻结视觉权重、约 1 GiB offload 最大参数暂存区、通信及待归约梯度、激活和前反向临时张量。不能用“总显存除以卡数”估算。

对原 6 卡约 79.2 GiB 的反向失败位置，仅扣除确定移出的主参数及动量、加上 offload 暂存区：`79.2 - 27.4 + 1.0 ≈ 52.8 GiB`。部分原 GPU 梯度累积还会移到 CPU。这只是同一失败位置的对照估算，不是完整训练峰值；失败后尚未执行的算子仍可能继续分配。

可将 micro_batch_size 增至 2，但默认保留 1。32768 长度下，主 decoder 可直接计算的保存量为：

| 项目 | batch=1 | batch=2 |
|---|---:|---:|
| 28 层 checkpoint 输入近似量 | 6.13 | 12.25 |
| 28 层可微 K/V | 1.75 | 3.50 |
| 两项合计 | 7.88 | 15.75 |

这些未包含层内重计算/反向、辅助 decoder、interaction、视觉编码和其他临时张量。若用每条满 32K 的激活及临时张量 15–25 GiB 作为规划假设，则总 GPU 预算可粗略按 batch=1 约 45–60 GiB、batch=2 约 60–85 GiB 考虑。此范围未经本机 7B 实测，也不是上限；两条长序列或大图片仍可能超过 80GB。

batch 内序列补齐到最长长度，多数 decoder 张量按 `B × T_max` 分配；FlashAttention 内部去 padding 不会消除 MLP/checkpoint/KV 的补齐占用。不同样本的 latent 位置取并集，视觉编码器一次处理 batch 内所有去重后的原图和 helper，16 个 pooled latent 不限制原图编码时的像素数量。

保持全局 batch 不变的例子：

| 卡数 | micro_batch_size | gradient_accumulation_steps | 全局 batch |
|---|---:|---:|---:|
| 6 | 1 | 8 | 48 |
| 6 | 2 | 4 | 48 |
| 8 | 1 | 4 | 32 |
| 8 | 2 | 2 | 32 |

先使用默认 batch=1 验证多个完整更新，再在独立输出目录尝试 batch=2。`mode: "smoke"`、`smoke_steps: 3` 可检查启动、累积和 checkpoint；默认 smoke 只覆盖头几个窗口，不代表最长样本通过。要验证容量，还需覆盖最长序列、多 action 和大图片样本。不同 batch/GAS/world_size 的 checkpoint 不可直接互相 resume。

## 验证范围

CPU 单元测试：

```bash
python -m unittest tests.test_sft2_offload
```

真实多卡 CPUAdam 验收：

```bash
python -m tests.test_sft2_offload_deepspeed
python -m tests.test_sft2_offload_deepspeed --bf16
```

CPU-only 数值代理验收可追加 `--cpu --cpu-test-optimizer`。FP32 模式与独立参考梯度、更新逐项比较；BF16 模式检查 CPU FP32 master/梯度/动量、有限梯度、跨 rank 参数一致及 checkpoint 恢复逐位一致。CPU-only 代理使用 torch AdamW 替代 CPUAdam 内核，能验证 native ZeRO offload 的分片、归约、累计和恢复，但不能证明 CUDA 传输、CPUAdam 编译/性能或真实 7B 显存容量。训练日志继续提供 `peak_allocated_gib` 和 `zero2_pending_grad_peak_gib`。

本次本地验收：新分支 10 项单测及原 SFT-II 配置/训练 10 项回归通过；两个进程的 FP32/Gloo 代理验收中，最大权重误差约 1.86e-7、最大梯度误差约 1.49e-7；BF16/Gloo 代理验收也通过，覆盖 GAS=4、三个更新窗口、空事件和保存恢复。真实 CPUAdam 内核、NCCL 及 7B 多卡容量尚未在本机验收；生产启动前会自动编译并自检 CPUAdam。

参考：[DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)、[DeepSpeed 内存估算](https://deepspeed.readthedocs.io/en/latest/memory.html)。
