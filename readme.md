# Toy-ZeRO-1

Toy-ZeRO-1 是一个单 GPU 教学项目，用 PyTorch 从零实现 ZeRO-1 的核心优化器状态管理思路。

本项目的硬件前提固定为：

```text
1 x NVIDIA GPU
WSL2 Ubuntu
Conda environment: toy-zero
PyTorch + CUDA
```

因此，本项目不再以多 GPU 显存节省作为验证目标。后续实现会围绕单 GPU 展开：

- 用标准 `torch.optim.Adam` 建立 baseline。
- 用手写 `ZeroAdam` 复现 Adam 更新路径。
- 在 `world_size=1` 下验证 `ZeroAdam` 与标准 Adam 的 loss 对齐和参数更新正确性。
- 保留 flatten / shard / write-back 结构，为理解 ZeRO-1 做准备。

## 环境

项目最终应放在 WSL Linux 原生文件系统：

```bash
~/Toy-ZeRO-1
```

推荐检查：

```bash
cd ~/Toy-ZeRO-1
conda activate toy-zero
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

期望 CUDA 可用：

```text
True
```

## 单 GPU 下的 ZeRO-1 定位

完整 ZeRO-1 在多 GPU 环境中会把 Adam optimizer states 按 rank 切分：

```text
parameters: full copy
gradients:  full copy
exp_avg:    1 / world_size shard
exp_avg_sq: 1 / world_size shard
```

但本项目只有一张 GPU，所以：

```text
world_size = 1
```

这意味着 optimizer state shard 等价于完整 state：

```text
exp_avg:    full local shard
exp_avg_sq: full local shard
```

所以本项目的重点不是证明显存下降，而是证明：

- flatten 参数和梯度的逻辑正确。
- 本地 shard 的 Adam 更新逻辑正确。
- 更新后的 flat parameter 能正确写回原模型。
- `ZeroAdam` 的 loss 曲线能和标准 Adam 对齐。

## 目录结构

```text
Toy-ZeRO-1/
├── model/
│   ├── __init__.py
│   └── mlp.py
├── optimizer/
│   ├── __init__.py
│   └── zero_adam.py
├── utils/
│   ├── __init__.py
│   ├── data.py
│   ├── distributed.py
│   ├── memory.py
│   └── seed.py
├── scripts/
│   ├── smoke_distributed.py
│   ├── run_smoke_distributed.sh
│   ├── smoke_model.py
│   ├── run_smoke_model.sh
│   ├── run_baseline.sh
│   ├── check_zero_adam.py
│   ├── run_zero_adam_check.sh
│   └── run_zero.sh
├── train_baseline.py
├── train_zero.py
├── requirements.txt
└── readme.md
```

`utils/distributed.py` 和 distributed smoke test 会保留，用于确认 CUDA/NCCL 基础环境。但主训练路径后续默认使用单进程、单 GPU，不再依赖多进程 DDP。

## 已有验证

分布式环境 smoke test，单 GPU 默认启动 1 个进程：

```bash
bash scripts/run_smoke_distributed.sh
```

模型和 synthetic data smoke test：

```bash
bash scripts/run_smoke_model.sh
```

可选模型压力测试：

```bash
bash scripts/run_smoke_model.sh --hidden-dim 1024 --num-hidden-layers 4 --batch-size 64
```

`ZeroAdam` 与标准 Adam 的逐步参数对齐测试：

```bash
bash scripts/run_zero_adam_check.sh
```

## 建设路线

### Phase 0: 环境确认

目标：确认 WSL2、conda、PyTorch、CUDA 可以用于单 GPU 训练。

已完成：

- CUDA 可用性检查。
- 单进程 smoke test。
- 模型 forward / backward / Adam step smoke test。

### Phase 1: 基础工具

目标：提供后续训练入口共享的工具模块。

已实现：

```text
utils/distributed.py
utils/seed.py
utils/memory.py
```

其中 `utils/distributed.py` 主要用于环境 smoke test 和未来扩展；单 GPU 主训练脚本会直接使用 `cuda:0`。

### Phase 2: 模型与数据

目标：建立可复现的 synthetic classification 任务。

已实现：

```text
model/mlp.py
utils/data.py
scripts/smoke_model.py
scripts/run_smoke_model.sh
```

当前模型：

```text
input_dim -> hidden_dim -> ... -> hidden_dim -> num_classes
```

数据由固定 seed 的随机 teacher 生成，不依赖外部下载。

### Phase 3: 单 GPU Adam Baseline

已实现。

计划文件：

```text
train_baseline.py
scripts/run_baseline.sh
```

训练流程：

1. 固定 seed。
2. 构建 synthetic dataset。
3. 构建 MLP。
4. 使用标准 `torch.optim.Adam`。
5. 训练固定 step 数。
6. 记录 loss、参数量、梯度范数和 CUDA 显存。
7. 输出 baseline loss 曲线，作为后续 `ZeroAdam` 对照。

这一步不使用 DDP。

运行命令：

```bash
bash scripts/run_baseline.sh
```

可选快速测试：

```bash
bash scripts/run_baseline.sh --steps 10 --log-interval 1
```

### Phase 4: 单 GPU ZeroAdam

已实现：

```text
optimizer/zero_adam.py
scripts/check_zero_adam.py
scripts/run_zero_adam_check.sh
```

实现内容：

1. 收集所有 trainable parameters。
2. 将参数 flatten 成一个一维 tensor。
3. 将梯度 flatten 成一个一维 tensor。
4. 在单 GPU 下令 shard 范围覆盖完整 flat tensor。
5. 只维护本地 Adam states：

```text
local_exp_avg
local_exp_avg_sq
step
```

6. 用 Adam 公式更新 flat tensor。
7. 将更新后的 flat tensor 写回原模型参数。

单 GPU 下这不会节省显存，但能验证 ZeRO-1 optimizer state 管理的核心代码路径。

第一版约束：

- 只支持一个参数组。
- 所有参数必须位于同一设备并使用同一浮点 dtype。
- 每个 trainable parameter 在每一步都必须具有 dense gradient。
- 支持标准 Adam 的 `lr`、`betas`、`eps` 和 L2 `weight_decay`。
- 暂不支持 AMSGrad、maximize、foreach、fused 和 mixed precision master weights。

运行正确性测试：

```bash
bash scripts/run_zero_adam_check.sh
```

该测试使用相同初始模型和相同 batch，逐步比较 `ZeroAdam` 与
`torch.optim.Adam(foreach=False, fused=False)` 的 loss 和所有模型参数。

### Phase 5: ZeroAdam 训练入口

计划文件：

```text
train_zero.py
scripts/run_zero.sh
```

训练流程与 baseline 保持一致，只把 optimizer 从标准 Adam 换成手写 `ZeroAdam`。

目标：

- 使用相同 seed。
- 使用相同 synthetic data。
- 使用相同 MLP 配置。
- 使用相同学习率和 step 数。
- 对比 baseline loss 与 zero loss。

### Phase 6: 正确性对比

最终验证项：

- baseline loss 和 zero loss 接近。
- `ZeroAdam` step 后参数仍是有限值。
- 梯度 flatten / parameter write-back 无 shape 错误。
- 在同一配置下重复运行结果稳定。

## 开发约束

- 所有主训练脚本默认单 GPU。
- 所有脚本默认在 WSL2 Ubuntu 中运行。
- 所有路径使用 Linux 风格 `/`。
- 主训练入口使用 `python train_*.py`，不使用 DDP。
- 不引入 DeepSpeed、FairScale 等高级封装。
- 第一版优先数学正确性和可读性。

## 当前进度

- [x] 明确单 GPU 作为固定硬件条件
- [x] 明确 Toy-ZeRO-1 的单 rank/reference 实现定位
- [x] 实现基础工具
- [x] 实现 distributed smoke test
- [x] 实现 MLP 与 synthetic data
- [x] 实现模型 smoke test
- [x] 实现单 GPU Adam baseline
- [x] 实现单 GPU ZeroAdam
- [x] 完成 ZeroAdam 与标准 Adam 的逐步参数更新对齐
- [ ] 实现 ZeroAdam 训练入口
- [ ] 完成 baseline 与 ZeroAdam loss 对比
