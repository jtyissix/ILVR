"""Build and exercise CPUAdam once before teacher workers or student NCCL start."""


def check_runtime():
    import gc
    import torch
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from ..sft1.zero2 import check_zero2_version

    check_zero2_version(deepspeed.__version__)
    parameter = torch.nn.Parameter(torch.ones(1, device="cpu", dtype=torch.float32))
    optimizer = DeepSpeedCPUAdam([parameter], lr=0.01, adamw_mode=True, fp32_optimizer_states=True)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    if not bool(torch.isfinite(parameter).all()) or parameter.item() >= 1.0:
        raise RuntimeError("DeepSpeedCPUAdam preflight update failed")
    del optimizer, parameter
    gc.collect()
    print("SFT-II offload runtime passed: DeepSpeedCPUAdam built and CPU update verified before NCCL.", flush=True)
