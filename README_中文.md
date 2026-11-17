# ScopeForm：Agent 代码与 Benchmark

本仓库提供 ScopeForm 的形式化、求解与纠错代码，LLM Agent 对照实现，以及 CPM-Sandbox 的数据、运行环境和评分器。当前发布内容不包含已记录的实验结果、模型回包、执行轨迹或论文结果表。

## 内容

| 目录 | 内容 |
|---|---|
| `benchmark/medical_scale_v06/` | 120 个来源簇、519 条初始命题、四类更新、运行环境和离线评分目标 |
| `code/baseline/` | LLM Agent 对照实现 |
| `code/deepseek/`、`code/qwen/` | 两类模型的形式化、语义库、求解和按 ID 纠错实现 |
| `code/analysis/` | 评分、配对比较和纠错轨迹分析代码 |
| `scripts/` | 可移植运行、评分、演示和服务启动入口 |
| `docs/semantic_library/` | 语义库使用说明与设计依据 |
| `examples/illustration/` | 构造的教学图与修改输入；不是真实实验回包 |
| `provenance/` | 运行配置、代码与模型哈希、依赖版本和原始辅助脚本 |

Benchmark 包含 72 个咨询对话来源和 48 个药品来源簇。五种设置为 Fresh、Labels、Labels+Index、Labels+Guard 和 Index+Guard。Guard 是共同的证据获取设置。每个阶段的读取、形式化、诊断和反馈调用共用最多 40 次模型请求与 40 次动作。

## 安装与离线演示

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/demo.py
python scripts/demo.py --backbone qwen
```

演示使用构造的用药案例执行语境绑定、关系编译、SMT 求解和按 ID 修改，不调用模型 API。输出保存在本地 `output/`，该目录不纳入 Git。

## 运行新的实验

```bash
# 准备 10 个来源的小批任务，不调用模型。
python scripts/run_experiment.py --backbone deepseek --scope pilot --out runs/ds --prepare-only

# 正式执行；API key 从环境变量 DEEPSEEK_API_KEY 或隐藏输入取得。
python scripts/run_experiment.py --backbone deepseek --scope pilot --out runs/ds

# 对新生成的运行记录评分。
python scripts/reproduce.py --run-dir runs/ds --out runs/ds/evaluation
```

默认重新执行初始阶段，再让同一来源的两个方法共享该初始检查点。`--scope full` 选择全部 120 个来源；`--source-ids` 指定来源子集；`--settings fresh labels index guard index_guard` 选择评测设置。新运行的输出写入本地 `runs/`。

Qwen 使用相同入口并指定 `--backbone qwen --endpoint http://127.0.0.1:18227/v1`。`scripts/serve_qwen.sh` 从已存在的本地权重启动 vLLM；可通过 `MODEL_PATH`、`SERVER_PYTHON` 和 GPU 相关环境变量调整部署。服务依赖见 `requirements-server.txt`。

如需使用自己保存的初始检查点，显式传入 `--initials recorded --initials-archive /path/to/shared_initials.tar.gz`。回放代码保留，但需要自行提供记录文件，参数见 `python scripts/replay_case.py --help`。

## 代码与评分目标

两类模型的冻结实现、benchmark 数据和评分器保留。文件名中的 `private`、`LOCAL_ONLY` 表示评分目标与 Agent 可见证据的隔离；推理入口不加载评分目标。配置与文件哈希用于核对实现，`manifest.json` 和 `checksums.sha256` 对应当前发布文件。

模型权重与密钥不随仓库提供。发布说明见 [PUBLICATION.md](PUBLICATION.md)，方法入口见 [code/README.md](code/README.md)，数据说明见 [benchmark/README.md](benchmark/README.md)。
