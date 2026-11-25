# AMD 执行手册：DSV4 四层 GRPO

目标是让测试人员只修改 `site.json` 的节点、网络和本地路径，之后使用同一入口
完成准备、执行、独立校验和结果汇总。当前提供单机 8 卡与四节点 32 卡，
**都是四层减层模型**。四节点扩展尚待 AMD 实机验收。

## 1. 资源与前提

- 每节点 8 张空闲 MI300X，或同型号的 gfx950 GPU；参与节点架构一致。
  宿主机须有可用 ROCm 驱动、`rocm-smi`、Python ≥3.10、Git、Docker 权限。
  本镜像是原有 ROCm 工具链，不能假定兼容任意宿主机驱动。
- 从**第一节点**运行统一入口；第一节点 `ssh` 填 `local`，其余节点配置无交互
  SSH。端口、密钥等放在用户自己的 `~/.ssh/config`，不要提交到仓库。
- 四节点的 `work_dir` 必须是同一份共享文件系统，路径一致且支持原子 rename。
  在不同机器上各建同名本地目录不满足条件。模型和基础数据可以是各节点同路径的
  本地副本，但内容校验必须一致。
- 节点 IP 必须是 Ray/RCCL/vLLM 可互通的真实 IP。按集群要求填写网卡，选择是否
  暴露 `/dev/infiniband`。需要开放专用 Ray 端口和该端口之后 1…299 的相关端口，
  并允许分布式 PyTorch/vLLM 所需的节点间动态连接；不要照抄另一集群的网卡名。
- `build`/下载阶段需要访问 Docker Hub、GitHub、Hugging Face。长跑保留权重版本、
  优化器和原始比较证据，建议准备 TB 级磁盘空间并监控配额。镜像归档也较大。

入口会拒绝选中卡上存在进程或 ≥1 GiB 未解释显存占用的情况，但它不能替代集群调度器
锁定卡。请预留整个执行窗口。不会停止别人的容器、现有 Ray 服务或重置 GPU。

## 2. 选择配置

```bash
git clone https://github.com/RL-Align/recipes.git
cd recipes

# 二选一
cp models/deepseek-v4/rocm/configs/single-node-8gpu.json site.json
# cp models/deepseek-v4/rocm/configs/four-node-32gpu.json site.json
```

编辑 `site.json`：

| 字段 | 填什么 |
| --- | --- |
| `work_dir` | 单机本地工作盘；四节点为共享工作目录，例 `/shared/dsv4-recipe` |
| `model_dir` | 原始四层 checkpoint 与 tokenizer 目录；下载方式可沿用模板路径 |
| `baseline_data_dir` | 基线 `train.parquet`、`val.parquet`、inventory、固定序列和审计 prompt 目录 |
| `nodes[].ip` | 实际参与计算的节点 IP，不能填示例 IP |
| `nodes[].ssh` | 第一节点 `local`；其余为 SSH 配置别名或 `user@host` |
| `network.ifname` | 实际通信网卡；留空则使用框架自动选择 |
| `network.rdma` | 确认集群已配置 RDMA 后设 `true`；默认 `false` |
| `network.ray_port` | 当前空闲的专用端口，默认 16379 |
| `gpu_arch` | MI300X 填 `gfx942`；MI350X 填 `gfx950`，构建时会重编译对应依赖二进制 |
| `numa_interleave` | 默认 `false`；仅在已确认需进程 NUMA interleave 的 MI35x 主机设 `true` |

卡号固定使用各节点物理 0…7。单机 actor 0…3、rollout 4…7；四节点前两台是 actor、
后两台是 rollout。32 卡配置为 actor PP2×DP8，rollout 共 8 个独立 PP2 实例，
每台 rollout 节点 4 个。TP/EP/CP 均为 1，不是全量 DSV4 或 EP 扩展。

```bash
python3 run.py plan --config site.json
```

`plan` 不访问 GPU、不创建容器，只显示实际拓扑、工作量和命令模板。

## 3. 一次性构建、准备资产

```bash
python3 run.py build --config site.json
python3 run.py prepare --config site.json --download
```

原镜像 `vensenmu/rl-kernel:pr711-grpo-stress` 是依赖底座，入口使用锁文件中的 registry
digest 构建带有配套新代码的派生镜像。vLLM 与 Megatron 版本保持锁定；代码必须同时包含
RL-Kernel 和 verl 的配套补丁。构建后生成：

- `WORK_DIR/runtime/runtime.lock.json`：实际新镜像 ID、源代码锁和 recipe 版本。
- `WORK_DIR/runtime/<image-id>.tar`：同一份镜像归档，启动时自动载入所有参与节点。

不需要自行给四个节点临时拼不同版本；也不需要预先发布派生镜像到 registry。
首次载入归档需要时间。发布 registry 镜像后可进一步简化分发，但必须记录其 digest。

特别注意：原 PR #711 镜像中的部分 vLLM 和 Transformer Engine 二进制只有 gfx942。
`gpu_arch: gfx950` 会按照 #717 的构建步骤，使用相同依赖源码重编译四个 vLLM 扩展及
TE common library，记录 `.so` 校验值；这不是升级外部框架。构建时间、内存和磁盘
需求明显增加，失败时保留日志，不能只修改配置伪装成已支持 gfx950。
选择 `numa_interleave: true` 会为本轮容器加入 SYS_NICE capability 并在训练入口使用
`numactl --interleave=all`，不修改宿主机 NUMA 策略。派生镜像构建尚待 AMD 环境实测。

构建要求 recipe checkout 干净且已提交。镜像锁绑定 recipe commit；更新 recipe 后需
重新构建，避免宿主机入口与容器脚本混用。`site.json` 和 `*.local.json` 已忽略，不影响
本地填写节点与路径。

下载入口引用固定资产：

| 资产 | 引用 |
| --- | --- |
| 四层模型 + 基础数据包 | `RL-Align/DeepSeek-V4-Flash-4L`，revision `4cdbf146efa5e2248349cfa33d4bfb54447c6dfd` |
| 包名 | `dsv4-4l-pr711-model-data.tar.zst` |
| 包 SHA256 | `35537489e57688e4497894ba6ad006fb89785d0315df1de26ab19207c051e132` |
| 权重 SHA256 | `1bd423ab1d98f43d40b19617fcb30d973d94436cc7b150fe3f744de79c2de086` |

已持有模型和 `final-data` 时，将两个路径指向原目录，省略 `--download`。
准备过程不使用 GPU。已成功准备的相同数据目录不覆盖；继续长跑时直接复用，
不要重复运行 `prepare`。准备中途失败的 `.partial` 或数据目录需先检查后另选新工作目录。

## 4. 三步正确性验证

保持 `profile: smoke`、`dataset: copy`，并选择一个从未使用过的输出目录：

```bash
python3 run.py run --config site.json \
  --output /shared/dsv4-recipe/runs/smoke-001
```

改过 `work_dir` 时同步修改这里的输出路径。统一入口会依次执行：

1. 镜像 ID、共享文件、GPU 空闲状态、物理 PCI/DRM/KFD 映射检查。
2. 每节点八卡健康检查、源码 tree/dependency 版本核对，建立本轮专用容器和 Ray 集群。
3. 三次真实 GRPO/PPO optimizer 更新，v0…v3 的训推逐位一致性与完整同步校验。
4. 独立 verifier，汇总结果，最后只停止本轮创建的容器并释放本轮资源。

这一步使用可解释的功能 copy 任务，验证计算和更新链路。四节点为了覆盖 DP8，
每步是 8 prompt×2 回答，基础六条样本重复至 24 条，重复事实记录在 manifest；
不能把这个结果称作真实数学大负载验证。

## 5. 同负载数学短跑，再跑 200 步

复制 `site.json` 为 `math.local.json`，修改：

```json
{
  "profile": "math-smoke",
  "dataset": "dapo",
  "gates": {
    "correctness_run": "/shared/dsv4-recipe/runs/smoke-001",
    "math_smoke_run": ""
  }
}
```

上面只是要修改的字段，其他节点/路径字段保留完整配置。也可将 `dataset` 设为 `gsm8k`。

```bash
python3 run.py prepare --config math.local.json
python3 run.py run --config math.local.json \
  --output /shared/dsv4-recipe/runs/math-smoke-001
```

数学数据固定引用：DAPO-Math-17k `65877096c24ffa7abc4e4fa5edb95cf3413a5674`；
GSM8K main `740312add88f781978c0658806c59bc2815b9866`。使用真实 prompt、真实采样回答、
原生规则 reward 和原生 GRPO advantage/optimizer。DAPO 前 128 条留作验证集，GSM8K
使用官方 test；训练按固定顺序、不 shuffle。按 tokenizer 模板过滤超长 prompt，
保留足够 200 步的样本，数据文件 hash 与实际行数写入 manifest。

数学短跑通过后，修改 `math.local.json` 的 `profile` 为 `math-200`，
`gates.math_smoke_run` 设为刚通过的目录，然后：

```bash
# 复用刚准备的数据，不重复 prepare
python3 run.py run --config math.local.json \
  --output /shared/dsv4-recipe/runs/math-200-001
```

目标是 **200 个实际 optimizer 更新步，不是 200 个 epoch**；每步 8 prompt×16 回答，
response 上限 7,168 tokens，prompt 上限 1,024。输出长度是采样结果，不能将长度上限
当作实际生成量。两个短跑门槛必须是相同镜像、模型、节点与网络；数学短跑还须是相同
数据集和目标负载。

如果数学 reward 全相同、advantage 全零，或长上下文产生任何 logprob 位差，运行会失败。
保留现场，分析根因后通过明确的新版本重跑。不能绕过有效更新保护、伪造 reward 或
把旧版本的 36 步等部分结果拼接成新版本的 200 步通过。

## 6. 验收与交付

```bash
python3 run.py verify --config math.local.json \
  --output /shared/dsv4-recipe/runs/math-200-001
python3 run.py report --config math.local.json \
  --output /shared/dsv4-recipe/runs/math-200-001
```

独立 `verify` 使用本轮输出保存的配置与镜像。`verification.json` 的 PASS 才能证明
该轮完整通过；`exit-code.json` 为 0 不够。交付以下文件，并保留原始证据路径：

- `site.json`、`runtime.lock.json`、`plan.json`、`launch.json`、每节点 preflight/probe。
- `driver.log`、`entrypoint.log`、`job-result.json`、`orchestration.json`。
- `verification.json` 与 `policy-sync/` 中的原始 tensor、优化器、PP/DDP、同步证据。
- `steps.csv`、`summary.json`、`report.md`。

表格参照 #377 的指标类别，记录本路径实际 rollout、actor train、权重同步、端到端时间、
吞吐、response 长度和 reward。KL 未启用标 N/A；没有的参考策略和 longest-sample
速度也标 N/A。审计和证据采集有开销，不能当作纯计算性能或直接宣称加速比。

异常时不会自动回退配置。中断控制台后应检查 `orchestration.json` 和容器 label，
确认本轮容器已经退出；原始数据不自动删除。清理失败的本轮容器时只操作记录中的确切
容器名，切勿停止其他用户的任务。
