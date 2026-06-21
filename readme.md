# Toy-ZeRO-1

Toy-ZeRO-1 是一个从零实现 ZeRO-1 优化器状态分片的教学项目。项目目标不是复刻 DeepSpeed 的完整工程能力，而是用 PyTorch 原生 `torch.distributed` 和 NCCL，把 ZeRO-1 的核心机制拆开、写清楚、跑通，并用标准 DDP + Adam 作为正确性基线。

本项目默认开发与运行环境是 WSL2 Ubuntu，项目最终应放在 Linux 原生文件系统：

```bash
~/Toy-ZeRO-1
```

推荐运行环境：

```bash
conda activate toy-zero
python --version
python -c "import torch; print(torch.cuda.is_available()); print(torch.distributed.is_nccl_available())"
```

期望 CUDA 和 NCCL 均可用：

```text
True
True
```

当前项目按单 GPU 环境优先建设。单 GPU 下 `world_size=1`，ZeRO-1 的 optimizer state shard 等价于完整 state，因此可以验证代码路径、Adam 更新逻辑和 loss 对齐，但不能观察到跨 GPU 分片带来的显存节省。真正的 `1 / world_size` optimizer state 显存收益需要至少 2 张 GPU。

## 核心目标

ZeRO-1 只切分优化器状态，不切分模型参数，也不切分梯度。

在标准 Adam 中，每个 rank 都保存完整的优化器状态：

```text
parameters: full copy
gradients:  full copy
exp_avg:    full copy
exp_avg_sq: full copy
```

在 ZeRO-1 中，每个 rank 仍保存完整参数和梯度，但 Adam 状态按 rank 切分：

```text
parameters: full copy
gradients:  full copy
exp_avg:    1 / world_size shard
exp_avg_sq: 1 / world_size shard
```

因此，ZeRO-1 的主要收益是降低 optimizer states 的显存占用。

## 计划目录

```text
Toy-ZeRO-1/
├── model/
│   └── mlp.py
├── optimizer/
│   └── zero_adam.py
├── utils/
│   ├── data.py
│   ├── distributed.py
│   ├── memory.py
│   └── seed.py
├── scripts/
│   ├── smoke_distributed.py
│   ├── run_smoke_distributed.sh
│   ├── run_baseline.sh
│   └── run_zero.sh
├── train_baseline.py
├── train_zero.py
├── requirements.txt
└── readme.md
```

## 建设路线

### Phase 0: 环境确认

目标：确认 WSL2、conda、PyTorch、CUDA、NCCL 可以用于多进程训练。

检查命令：

```bash
cd ~/Toy-ZeRO-1
conda activate toy-zero
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.distributed.is_nccl_available())"
```

项目提供了一个分布式环境 smoke test。单 GPU 默认启动 1 个进程，确认进程能正确拿到 `rank`、`world_size`、`local_rank` 和 GPU device。

```bash
bash scripts/run_smoke_distributed.sh
```

单 GPU 机器不要设置 `NUM_PROCESSES=2`，否则第二个进程会尝试绑定不存在的 `cuda:1`。

如果未来有多张 GPU，可显式指定进程数：

```bash
NUM_PROCESSES=2 bash scripts/run_smoke_distributed.sh
```

### Phase 1: 分布式基础工具

目标：封装所有训练入口都会用到的 distributed 初始化逻辑。

计划文件：

```text
utils/distributed.py
utils/seed.py
utils/memory.py
```

关键能力：

- 从 `LOCAL_RANK` 设置当前 CUDA device。
- 调用 `torch.distributed.init_process_group(backend="nccl")`。
- 提供 `rank`、`world_size`、`local_rank`、`device`。
- 只允许 rank 0 打印主日志。
- 训练结束时正确 `destroy_process_group()`。

### Phase 2: 模型与数据

目标：先使用可控的小型 MLP 和 synthetic data，保证正确性验证足够稳定。

计划文件：

```text
model/mlp.py
utils/data.py
```

设计原则：

- 数据由固定 seed 生成，不依赖外部下载。
- 输入、标签、模型初始化都要可复现。
- 第一版模型使用 MLP，避免 Transformer 细节干扰 ZeRO-1 逻辑。

初始模型形态：

```text
input_dim -> hidden_dim -> hidden_dim -> num_classes
```

后续可通过 `hidden_dim` 和 `num_layers` 放大参数规模，用于观察 optimizer states 的显存差异。

### Phase 3: DDP + Adam 基线

目标：建立正确性对照组。

计划文件：

```text
train_baseline.py
scripts/run_baseline.sh
```

训练逻辑：

1. 初始化 distributed。
2. 固定 seed。
3. 创建模型并移动到当前 GPU。
4. 使用 `torch.nn.parallel.DistributedDataParallel` 包装模型。
5. 使用标准 `torch.optim.Adam`。
6. 训练固定 step 数，记录 loss。
7. 记录 `torch.cuda.max_memory_allocated()`。

这条 loss 曲线是后续 `ZeroAdam` 必须对齐的基线。

### Phase 4: ZeroAdam 核心实现

目标：手写 ZeRO-1 optimizer states 分片。

计划文件：

```text
optimizer/zero_adam.py
```

核心步骤：

1. 收集模型中所有 trainable parameters。
2. 把参数 flatten 成一个一维大 tensor。
3. 把梯度 flatten 成一个一维大 tensor。
4. 根据 `rank` 和 `world_size` 计算当前 rank 负责的 shard 范围。
5. 每个 rank 只为自己的 shard 初始化 Adam 状态：

```text
local_exp_avg
local_exp_avg_sq
```

6. 每个 step 只更新本地 shard。
7. 使用 `dist.all_gather_into_tensor` 收集所有 rank 更新后的参数 shard。
8. 把完整 flat parameter 写回原模型参数。

注意：第一版只支持所有 rank 形状一致、使用 dense gradients、单 optimizer group，先保证主线正确。

### Phase 5: 手动梯度同步 + Zero 训练入口

目标：不用 DDP 的 optimizer 行为，显式控制梯度同步和参数广播。

计划文件：

```text
train_zero.py
scripts/run_zero.sh
```

训练逻辑：

1. 每个 rank 拥有完整模型副本。
2. forward / backward 得到本地梯度。
3. 对每个参数梯度执行 `dist.all_reduce`。
4. 梯度除以 `world_size`，得到全局平均梯度。
5. 调用 `ZeroAdam.step()`。
6. `ZeroAdam` 内部更新本地 shard 并 all-gather 完整参数。
7. 记录 loss 和显存。

理论上，在相同 seed、相同 batch、相同 Adam 超参数下，`train_zero.py` 的 loss 应和 `train_baseline.py` 接近对齐。

### Phase 6: 正确性与显存对比

目标：先在单 GPU 上证明 Toy-ZeRO-1 的数学正确性，再在多 GPU 环境下验证显存节省预期。

验证项：

- Baseline loss 和 Zero loss 在固定 step 内接近。
- 每个 rank 的模型参数在 `ZeroAdam.step()` 后保持一致。
- 单 GPU 下，ZeRO-1 的 optimizer state 本地元素数量等于完整 Adam state，作为退化正确性测试。
- 多 GPU 下，ZeRO-1 的 optimizer state 本地元素数量约等于完整 Adam state 的 `1 / world_size`。
- 多 GPU 且较大模型配置下，ZeRO-1 的 `max_memory_allocated` 应低于 baseline。

建议命令：

```bash
bash scripts/run_baseline.sh
bash scripts/run_zero.sh
```

## 开发约束

- 所有脚本默认在 Linux/WSL2 Ubuntu 中运行。
- 所有路径使用 Linux 风格 `/`。
- 多进程训练使用 `torchrun`。
- 分布式 backend 默认使用 NCCL。
- 初期避免引入 DeepSpeed、FairScale 等高级封装。
- 第一版优先正确性，再优化性能和边界情况。

## 当前进度

- [x] 明确 Toy-ZeRO-1 项目目标
- [x] 明确 WSL2 + CUDA + NCCL 作为目标运行环境
- [x] 重写 README 与建设路线
- [x] 实现分布式初始化工具
- [x] 实现分布式 smoke test
- [ ] 实现 MLP 与 synthetic data
- [ ] 实现 DDP + Adam baseline
- [ ] 实现 ZeroAdam optimizer states 分片
- [ ] 实现 ZeRO-1 训练入口
- [ ] 完成 loss 与显存对比
