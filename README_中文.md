# ScopeForm：代码与结果复现包

对应短文 **Context Before Calculation: Context-Faithful Formalization for Medical Evidence Agents**。本包包含 `deepseek-flash`、`Qwen3.8-27B` 在五个设置中的实际实验记录、冻结方法代码、CPM-Sandbox 运行时和评分数据。

直接查看：[结果表](results/table1.md) · [CSV](results/table1.csv) · [Agent 代码索引](code/README.md) · [复现检查记录](validation/README.md)。

最先运行 **离线复现**：不需要 API key、GPU 或模型权重，也不产生模型调用费用。它会从逐阶段的实际输出重新调用官方评分器，并验证论文表格、配对区间和纠错轨迹。

## 1. 三条复现路径

| 目的 | 入口 | 需要什么 |
|---|---|---|
| 重算论文结果、置信区间、成本与纠错统计 | `scripts/reproduce.py` | Python 依赖；包内数据足够 |
| 看懂形式化、未知值与按 ID 纠错 | `scripts/demo.py` | Z3；全程离线 |
| 重新调用模型生成实验结果 | `scripts/run_experiment.py` | DeepSeek key 或本地 Qwen 服务 |

### 安装

在本目录执行；记录的推理环境是 Python 3.10。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

模型服务单独使用 `requirements-server.txt`；离线复现无需安装 vLLM、PyTorch 或 Transformers。`paramiko` 是原评分器导入链的依赖，离线评分不会使用 SSH。

### 一条命令重算论文结果

```bash
python scripts/reproduce.py
```

程序会：

1. 校验交付文件的 SHA-256。
2. 读入两个模型各 4,920 个阶段，删除旧评分，再用原评分器从实际提交、证据与状态重新评分。
3. 核对 240 个实际初始检查点、4,800 对更新阶段和每阶段 40 次模型调用上限。
4. 重新计算 `M / E / J`、提交数、请求数、token 数与耗时。
5. 按来源簇做 20,000 次配对 bootstrap，核对区间。
6. 逐项核对论文合并表中的 **60 个数值**。
7. 从保存的前后图、诊断和求解结果重建纠错分析：2,372 对首轮记录，另列 254 对后续 Guard 轮次。

主要输出在 `output/reproduced/`：

- `table1.csv`、`table1.md`、`table1_rows.tex`：重算后的合并表；CSV 额外保留覆盖率和调用成本。
- `deepseek/`、`qwen/`：逐阶段新评分、各设置指标、配对区间、总成本。
- `repair_audit/summary.json`、`repair_audit/FINDINGS.md`：纠错统计及解释。
- `verification.json`：复现是否与记录一致。

只想快速核对主表、统计与成本：

```bash
python scripts/reproduce.py --skip-repair-audit
```

完整运行会将压缩的纠错记录展开到输出目录，需要约 0.5 GB 额外空间。输出可以删除后重新生成。`expected/` 是核对目标，不是生成结果的数据来源。

### 离线方法示例

```bash
python scripts/demo.py
python scripts/demo.py --backbone qwen --out output/demo_qwen
```

示例使用同一患者、同一药品的三条记录：过去实际用药 `5 mL × 2/天`、当前医嘱 `5 mL × 2/天`、当前实际用药 `2.5 mL × 4/天`。两个查询分别检查日总量与“单次量 AND 频次”。

- 错将实际角色绑定到过去：两项均得到 `supported`。
- 按 ID 修改 `obs.where.time` 为 `current`：得到 `supported / refuted`，来源记录不变。
- 当前频次未知：得到 `not_determined / refuted`。

这是论文插图的构造示例，诊断编辑由示例脚本给定；它用于检查可执行语义接口，不是一次新模型评测。真实案例见 `examples/real_case/`，完整纠错记录见 `artifacts/deepseek/repair_inputs.tar.gz`。

### 真实案例的完整执行回放

```bash
python scripts/replay_case.py
```

它把 S000384 / single_definite / Labels+Guard 的实际回包送回冻结运行时，重新执行两条 Agent 分支，并运行形式化求解。脚本核对每个生成的模型请求与保存请求完全一致，以及最终提交、状态、引用和调用数一致：基线 4 次、ScopeForm 5 次保存回包。它不调用模型，也不读取评分目标。

## 2. 重新运行模型

### DeepSeek：先准备，再执行

```bash
python scripts/run_experiment.py --backbone deepseek --scope pilot \
  --out runs/deepseek_pilot --prepare-only
python scripts/run_experiment.py --backbone deepseek --scope pilot \
  --out runs/deepseek_pilot
```

第二条命令会从 `DEEPSEEK_API_KEY` 环境变量读取 key；未设置时采用隐藏输入。key 不写入配置、日志或压缩包。默认地址为 `https://api.deepseek.com`，模型为记录时的 `deepseek-flash`。

- `--scope pilot`：按来源 ID 的固定哈希排序选 6 个对话来源、4 个药品来源，不按得分挑选。
- `--scope full`：120 来源、五设置、两方法，共 4,800 个更新阶段。
- `--source-ids S000384`：只跑指定来源。
- `--settings guard`：只跑 Labels+Guard；可列出多个设置。
- `--workers 4`：调整并发；不改变每阶段预算。

默认 `--initials recorded`：复用**对应模型真实执行过的初始检查点**，每个更新分支重新调用模型。这与论文配对设计一致，历史标签来自实际初始回答。`--initials fresh` 则连初始阶段也重新执行。

```bash
python scripts/run_experiment.py --backbone deepseek --scope full \
  --initials recorded --out runs/deepseek_full
python scripts/reproduce.py --run-dir runs/deepseek_full \
  --out runs/deepseek_full/evaluation
```

后一次新推理不保证逐 token 重现历史结果；离线复现入口精确复现已记录的结果。新跑结果在独立目录评分，不覆盖论文历史数据。相同命令可恢复已有检查点，调用数不重新归零。

### Qwen3.8-27B

记录的服务环境：A100-SXM4-80GB、`bfloat16`、上下文 65,536、vLLM 0.19.1、PyTorch 2.10.0、Transformers 5.20.1。完整服务参数与模型卡在 `provenance/qwen/`。

服务器上已有权重路径：

```text
/data/houwanlong/cpm_qwen_20261115/models/Qwen3.8-27B
```

原始权重、tokenizer 与配置的逐文件 SHA-256 位于 `provenance/qwen/Qwen3.8-27B.files.sha256`。

本包不复制模型权重。`scripts/serve_qwen.sh` 默认使用该目录和现有服务环境，缓存也在 `/data/houwanlong` 下；它不会下载模型。执行脚本才会启动服务：

```bash
bash scripts/serve_qwen.sh
```

可用 `MODEL_PATH`、`SERVER_PYTHON`、`CUDA_VISIBLE_DEVICES`、`TENSOR_PARALLEL_SIZE`、`GPU_MEMORY_UTILIZATION`、`PORT` 覆盖路径或硬件分配。原服务器使用的两份启动脚本也保存在 `provenance/qwen/scripts/`。其机器特定的 `LD_PRELOAD` 没有设为通用脚本的默认值。

在另一个终端运行：

```bash
python scripts/run_experiment.py --backbone qwen --scope pilot \
  --endpoint http://127.0.0.1:18227/v1 --out runs/qwen_pilot
python scripts/reproduce.py --run-dir runs/qwen_pilot \
  --out runs/qwen_pilot/evaluation
```

Qwen 请求使用 `seed=20261022`、`enable_thinking=false`、原生工具 `tool_choice=auto`。便携入口将配对方法放到同一 endpoint；历史实验的多副本调度和迁移记录另存于 `provenance/qwen/rebalance/`。

## 3. 实验与指标

共有 120 个来源簇（72 对话、48 药品），初始 519 条命题。每个来源有四个更新事件：`key_mixed`、`single_definite`、`single_uncertainty`、`neutral`。每个模型共享 120 个初始阶段，执行 `120 × 4 × 5 × 2 = 4,800` 个更新阶段，总计 4,920；两模型合计 9,840。

| 论文名称 | CLI 名称 | 原运行时分支 |
|---|---|---|
| Fresh | `fresh` | `fresh` |
| Labels | `labels` | `labels_only` |
| Labels+Index | `index` | `index_only` |
| Labels+Guard | `guard` | `labels_guard` |
| Index+Guard | `index_guard` | `evidence_constraint` |

**Guard 是 benchmark 的证据获取条件，不是 ScopeForm 组件。** 同一设置内两方法接受相同的初始状态、公开任务和 Guard。ScopeForm 接收其拒绝信息后可补读；Guard 不给标准答案。

每个设置、每个方法均有 480 个更新阶段：

- `M`：先在每阶段按全部计划命题计算标签正确率，再按来源取平均，最后在来源簇间取平均，报告百分数。未提交、失败和被初始失败阻塞的阶段保留在分母并记零。
- `E`：官方评分器认定证据通过的已提交包数量，分母 480。
- `J`：官方 `strict` 联合成功包数量，包含答案、证据与适用的状态/动作/回读要求，分母 480。
- 各项越高越好；完整定义由冻结的 `score_study15_local.py`、`score_workflow11_local.py` 执行。

配对推断按来源簇重采样，保留同来源的四个更新；20,000 次，种子 20261114。`M` 同时给 95% 区间与五比较 Bonferroni 99% 区间；`E/J` 的 95% 区间为描述性结果。十个 M 点估计均提高，不等于十项都达到校正后的显著性。

所有原生调用、形式化、诊断和纠错反馈共用**每阶段最多 40 次模型请求、40 次动作**。temperature=0、最大输出 8,192 tokens、关闭 thinking。失败仍计分，初始成本单列，不重复算到十个分支。

Qwen Labels+Guard 有一个实际中止的阶段：`S000056 / single_uncertainty / guard__formalization`，33 次调用后按未提交计零。已原样保留；它不是一次正常耗尽 40 次预算的失败。详情见 `provenance/qwen/evaluation/termination_audit.json`。

## 4. 代码怎么读

最直接的阅读顺序是：

1. `scripts/run_experiment.py`：便携实验入口，选择模型、五设置和来源。
2. `code/<backbone>/run_five_settings.py`：在相同原始环境分支上配对基线与 ScopeForm。
3. `code/<backbone>/frozen_code/run_methods.py`：原生读取、形式化入口、统一调用预算和 Guard 反馈。
4. `methods/frame_direct.py`：`solve_and_submit` 接口；模型交付形式化图，程序求解并真正保存、回读、提交。
5. `methods/frame_protocol.py`、`frame_context.py`：来源记录、引用、角色、查询的表示与编译接口。
6. `definitions/frames.json`、`core.json`：可执行语义定义；`methods/frame_library.py`、`record_solver.py`、`solver_worker.py` 负责展开和求解。
7. `methods/frame_repair.py`：四类诊断、按记录 ID 编辑、编译/执行错误反馈循环；无问题时返回空 issues，不编辑。
8. `definitions/error_taxonomy.json` 与 `docs/semantic_library/DESIGN.md`：分类及理论依据。

上述 `methods/`、`definitions/` 路径均相对于对应的 `frozen_code/`。理论依据包括框架语义、医疗显式语境和请求/事件区分、类型化临床表达式，以及条件逻辑与开放世界知识状态；执行真值以 JSON 定义与代码快照为准。

DeepSeek 对应 `frame_repair_id_fields_20261114`；Qwen 对应 `frame_repair_parse_recovery_20261115`。Qwen 快照另外包含结构化接口和解析适配（`frame_schema.py`、`repair_schema.py` 等）；两个快照均按实验配置里的哈希完整保留，不能互换后声称逐字节复现。同一模型五个设置使用同一快照。

冻结目录中保留了当时导入链需要的旧模块；主复现入口只选择 `frame_direct[_guard]_repair`。`code/baseline/evidence_notes.py` 是原基线文件的导入依赖，笔记功能在这些实验中关闭。新增的便携脚本处理路径、服务地址和打包数据，不修改原方法或评分公式。

## 5. 文件索引与数据边界

```text
scripts/                 离线复现、模型重跑、示例与 Qwen 启动入口
code/deepseek/            DeepSeek 对应的冻结代码
code/qwen/               Qwen 对应的冻结代码
code/baseline/           原生 Agent 与评分辅助函数
code/analysis/           原分析脚本
benchmark/               原运行时、120 来源 actor 输入、独立评分目标
artifacts/               两模型逐阶段记录、初始检查点、完整 DS 纠错轨迹
expected/                归档指标与论文 LaTeX 表，用于核对
examples/                插图示例、真实案例与实际请求/回包
provenance/              原配置、代码/数据哈希、模型身份、服务与中止记录
results/                 本次重算的主表、区间、成本与纠错汇总
validation/              本次独立环境复现记录
manifest.json            逐文件 SHA-256
```

本仓库是作者授权的公开复现版本，包含主实验的 actor 数据、独立评分目标及实际实验记录。历史文件名中的 `private` / `LOCAL_ONLY` 表示它们在评测时与被测 Agent 隔离，不表示 GitHub 仓库设为私有。在线推理入口只读取 actor 证据与实际初始检查点，不读取 evaluator targets；只有离线评分进程读取目标。原始 manifest 的发布状态字段作为历史来源记录保留，当前公开版本说明见 `PUBLICATION.md`。模型和来源材料沿用各自的权利声明。

包中不含 API key、SSH 密码、虚拟环境或模型权重。原配置中的历史路径只用于来源追踪；可执行入口均相对本包定位。纸面结果、逐阶段重新评分、形式化示例与新模型运行分别对应不同的复现层次。
