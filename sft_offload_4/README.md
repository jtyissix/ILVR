# SFT-II offload：8 卡转 4 卡续训

新增代码、入口和测试均在本目录；原 8 卡代码不变。支持本项目 DeepSpeed 0.18.0、ZeRO-2、CPUAdam offload、BF16、纯数据并行的完整训练 checkpoint。不能使用 export。

## 1. CPU 离线转换

在 ILVR 根目录、原训练环境运行，替换 checkpoint 和原配置路径：

```bash
python -m sft_offload_4.convert \
  --checkpoint /你的8卡output/checkpoints/global_step00000300 \
  --config /你的8卡output/train_config.json \
  --output sft_offload_4/converted
```

也可指定原 configs/sft2_offload/latent.json，前提是训练参数和它引用的 DeepSpeed JSON 未改变。checkpoint 可为具体 step 目录或含 latest 的 checkpoints 目录。转换输出必须是新目录，不能覆盖源文件。

不需要 GPU，不初始化 NCCL，不运行 teacher 推理，不重做 prepare。只读取 prepared 的小型 manifest 取得样本数，不读取或 hash records、图片、teacher 张量，不修改缓存。原 output 下的 run_info.json、完整 checkpoint、原训练配置和 DeepSpeed JSON 必须存在。

转换会检查全部 8 个分片的 FP32 主权重、Adam exp_avg/exp_avg_sq、step、参数映射、padding、学习率和窗口位置。拒绝 NaN/Inf、负二阶动量、缺失分片和配置不一致。

每个参数组按完整坐标重新分为 4 份，并重新计算 DeepSpeed 的 2×DP 对齐与 padding，不能简单拼接两个文件。保存后重新读取，逐元素验证主权重和两份动量；模型权重、scheduler、global_steps、global_samples、epoch、next_window 均保留。完整验证通过后才写 complete.json 和 latest。失败产物不能训练，选择新目录重新转换。

默认生成：

```text
sft_offload_4/converted/
  latent.json
  zero2.json
  latest
  global_step00000300/
    mp_rank_00_model_states.pt
    bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
    ... rank 1–3 ...
    rng_rank0.pt ... rng_rank3.pt
    migration.json
    complete.json
```

latent.json 自动设置 num_gpus=4、gpu_ids=null，梯度累积翻倍；每卡 batch、学习率、loss 权重、seed、epochs、interaction 均不变。原 8×1×8 变为 4×1×16，全局 batch=64，训练窗口和 LR 总步数保持一致。

采用 mmap 读取源分片，逐目标 rank 分配，不拼接完整 FP32 模型。仍需充足 CPU 内存和磁盘：14.7B 参数的完整新 checkpoint 约 14P bytes，即约 192 GiB，另留开销和后续 checkpoint 空间。单个目标 rank 的主权重与两份动量约 41 GiB，页缓存、加载与写出还有额外占用，此值不是进程内存上限。--threads 控制数值检查线程数，默认 4。

## 2. 独立启动单机 4 卡训练

Slurm 申请 --gres=gpu:4、--ntasks-per-node=1，然后在 ILVR 根目录运行：

```bash
python -m sft_offload_4.run --config sft_offload_4/converted/latent.json
```

默认训练输出在 converted/training；也可转换时用 --training-output 指定新目录。不要复用原 8 卡 output_dir。

该入口不调用旧 prepare、teacher 生成器或内容 hash 扫描。prepared_dir 和 teacher_cache_dir 保持原路径，逐样本读取 teacher 时也不计算 SHA256。保留文件存在性、样本数和记录覆盖、格式、latent 边界、图像网格、张量归属/形状/dtype/有限值检查。配置签名只检查小型 JSON 参数和原有 provenance 字段，不重新 hash 数据、图片或模型文件。

启动器单进程检查 CPUAdam 环境后启动 4 个训练进程。恢复后逐元素检查内存中的 FP32 主权重、Adam 动量和 step 是否等于转换文件，并核对 optimizer 超参数、参数映射、scheduler、step 和样本计数。继续使用原 offload 的 FP64 范数裁剪及跨 rank 数值保护；诊断日志仍在 output_dir/numerics。

再次 resume 4 卡训练时，把生成的 latent.json 的 resume 改为新 4 卡 checkpoint 或 checkpoints 目录，运行相同训练命令即可，无需再次转换。

## 一致性与验证

转换后的有效主权重、两份动量、step、模型权重和 scheduler 必须与源文件完全一致。新 rank r 继承旧 rank r 的 RNG 状态；8 个随机流不能无损合并为 4 个。浮点归约顺序和 BF16 累积路径也改变，所以后续训练不保证与 8 卡逐位重放。

```bash
python -m unittest sft_offload_4.test_migration sft_offload_4.test_training
python -m sft_offload_4.acceptance --output sft_offload_4/test-results/fp32
python -m sft_offload_4.acceptance --bf16 --output sft_offload_4/test-results/bf16
```

acceptance 依次启动 8 个和 4 个 CPU/Gloo 进程，使用原生 DeepSpeed ZeRO-2 offload，以 torch AdamW 替代 CPUAdam 内核，检查保存、转换、精确恢复、下一步更新、scheduler 及 4 卡保存再恢复。它不能代替真实 7B 的显存/吞吐或 CPUAdam/NCCL 验收。具备 8 张 GPU 时可加 --cuda，在小模型上使用真实 CPUAdam/NCCL。

本地验收：8 项迁移/缓存/训练入口测试通过；真实 8→4 进程的 FP32 与 BF16/Gloo 代理测试通过，主权重和 Adam 恢复误差为零。FP32 下一步权重最大差异约 3.73e-9；该 BF16 小模型测试的下一步权重与 loss 差异均为零。这不保证大模型跨卡数逐位一致。
