# 可执行语义定义与角色框架

语义库包含两层：[core.json](../../code/deepseek/frozen_code/definitions/core.json) 的
12 条计算定义，以及 [frames.json](../../code/deepseek/frozen_code/definitions/frames.json)
的 6 个角色化语义框架。框架规定参数从哪类记录读取、哪些字段共同绑定，再展开为
计算定义；记录选择器和 Z3 求解器负责绑定与判定。
[DESIGN.md](DESIGN.md) 记录设计的理论依据与表示约定。

[理论分类定义](../../code/deepseek/frozen_code/definitions/error_taxonomy.json) 包括
表示合法性、概念与角色保真、语境保真、逻辑与知识状态保真。每类记录判定要求、
边界、示例及理论来源；同一模型据此诊断图并输出局部修改，详见 DESIGN.md 的 P6。

以下代码链接指向 DeepSeek 冻结实现。Qwen 对应实现位于
[code/qwen/frozen_code](../../code/qwen/frozen_code)，目录结构相同。
本页中的数值和对象例子是构造的语义教学材料。

## 模型接入：一个表达式协议

[run_methods.py](../../code/deepseek/frozen_code/run_methods.py) 集成 Agent 读取、
形式化与提交流程；[frame_direct.py](../../code/deepseek/frozen_code/methods/frame_direct.py)
提供 `solve_and_submit`。模型生成带 `evidence_ids` 的形式化查询，程序求解并执行
保存、回读与提交。每条 query 的标签来自求解器，引用来自同一次生成。

[frame_protocol.py](../../code/deepseek/frozen_code/methods/frame_protocol.py) 使用
嵌套 `op/args` 协议：query 为 `{item_id,text,expr,evidence_ids}`。框架按 `slots`
访问语义槽，原子 `ref` 使用实际属性名。所有调用写为 `{"op":操作名,"args":[参数]}`：

```json
{"op":"and","args":[
  {"op":"single_dose_conforms","args":["current","advice"]},
  {"op":"ge","args":[{"op":"ref","args":["current","duration_days"]},6]}
]}
```

模型输出的引用评分为 `scores: {record_id: score}`。同一引用的所有字段共享这个
语义分；程序展开为选择器需要的字段索引格式。`decode` 将统一表达式转换成下文
内部框架格式，再执行 `lower`、共享记录选择与 Z3 求解。

查询可以直接读取已明确的来源记录，例如
`{"op":"record","args":["rule_record","threshold"]}`；需要消歧或共享绑定时使用
`ref`。两种地址可出现在同一查询中。来源 rules 使用固定 record 地址，来源事实
不随查询绑定变化。缺失字段保持未知。

查询及 NOT/AND/OR 等明确布尔位置可推导变量为 Bool；这不补充真假值，也不推测
不明确的数值类型。比较参照按原文保留 order 或 recommendation 等意图；条件规范
使用 when、modality、action，与实际事实分开。

从仓库根目录准备一个不调用模型的小批量运行：

```sh
python scripts/run_experiment.py --backbone deepseek --scope pilot --out runs/demo --prepare-only
```

## 6 个角色化框架

| 框架 ID | 角色 | 只判断什么 |
|---|---|---|
| `object_matches_reference` | 实际方案 observed、推荐方案 reference | 对象身份是否相同 |
| `object_in_allowed_set` | 实际方案 observed、完整推荐集合 allowed | 实际对象是否属于集合 |
| `single_dose_conforms` | 实际方案 observed、推荐方案 reference | 单次量是否在推荐闭区间内 |
| `dose_frequency_conforms` | 实际方案 observed、推荐方案 reference | 单次量和日频次是否分别符合 |
| `daily_total_conforms` | 实际方案 observed、推荐方案 reference | 名义日总量是否在推荐乘积区间内 |
| `event_completed` | 实际事件状态 event | 事件是否已实际完成 |

每个框架保存含义、能力问题、角色及字段类型、来源要求、可执行 body、易混淆含义
与适用边界。`theory_basis` 对应 DESIGN.md 的 P1–P4，避免每条框架重复文献说明。

输入沿用现有 `properties / records / references / rules / queries` 图结构。
例如一条查询：

```json
{
  "item_id": "Q1",
  "text": "当前实际单次量符合指定推荐",
  "expr": {
    "frame": "single_dose_conforms",
    "roles": {"observed": "current", "reference": "advice"}
  }
}
```

`current` 与 `advice` 是已有引用 ID。`lower` 为前者添加
`kind=regimen, modality=actual`，为后者添加
`kind=regimen`，保留该引用原文指定的 modality（如 order 或 recommendation），然后生成：

```text
advice.dose_lower ≤ current.dose ≤ advice.dose_upper
```

同一引用被多个框架使用时仍共同绑定，不按查询复制引用。角色与引用的明确定位
条件相冲突时，该表示不能编译；程序不会偷偷修改意图或创建替代引用。
`allowed` 接受引用列表，例如 `["advice1", "advice2"]`，表示明确允许的多个成员。
它不是一个含糊引用的候选集合：候选表示“究竟指哪个”，允许集合表示“这些都允许”。

框架可以放在已有 `negate / all_of / any_of` 定义或原有 AND/OR/NOT 表达式中。
只有选择了对象比较框架才判断药品身份，单次量框架不会暗加身份一致等条件。

### 语境与字段的最小约定

`records.subject` 表示主体；药品或检查身份只有一个真源：角色 `object_id` 槽
指向的来源字段，默认是 `records.fields.object_id`。新增的引用 `where.target`
读取同一字段，不另造身份元数据。既有 `entity` 选择器保留兼容，框架示例采用
明确的 subject/target。

主体、时间、场景及来源定位只使用引用原句给出的条件。框架不默认当前时间，
不把实际患者复制给通用推荐，也不自动要求两侧对象相同。
`where.target` 适用于“药品甲的单次量”等已经指定对象的指称；问“当前药品是否为甲”
时，甲应出现在比较参照中，不能先放入实际角色的定位条件。

本版 `modality` 的规范值为 `actual / recommendation / plan / order`，用于区分
实际状态与行动意图，不是完整 FHIR 意图模型。`reported` 是来源方式：患者自述
正在服用可抽取为 actual。符合性框架比较 actual 与命题指定的参照；比较关系
不限定参照必须是 recommendation。plan/order 参照按原意定位并保留，不能因为
与实际值相同就被改写成 actual，也不能为了框架而改写成 recommendation。
独立模拟场景内给定的实际值也使用 actual，由 scenario 与其他场景隔开。

执行状态只放在 `fields.completed`。计划记录中的 true 表示该计划的目标状态，
actual 记录中的 true 才表示实际完成。`event_completed` 不读取计划目标。
字段为 null/缺省时仍保留候选和符号变量；找不到记录是引用未解决，与属性未知分开。

第一版采用以下语义槽和类型签名，抽取时应按原文语义映射：

| 语义槽（默认同名字段） | `semantic_type` | 底层 `type` | 单位约定 |
|---|---|---|---|
| object_id | Identity | String | 已对齐的身份 |
| dose、dose_lower、dose_upper | Dose | Int 或 Real | 声明 unit，同一框架内一致 |
| frequency、frequency_lower、frequency_upper | Frequency | Int 或 Real | unit 为 `1/d`，每日次数 |
| completed | Completion | Bool | 相对于所属记录的完成状态 |

声明示例：`{"name":"dose","type":"Real","semantic_type":"Dose","unit":"mg"}`。
引用可通过 `slots` 显式映射已有字段名，例如：

```json
{
  "id": "current_tablets",
  "text": "当前药品乙的实际方案",
  "where": {"subject": "P", "target": "药品乙", "time": "current"},
  "slots": {"dose": "dose_tablets", "object_id": "drug_name"},
  "scores": {"dose_tablets": {"R2": 3}}
}
```

其中 `dose_tablets` 声明为 Dose、单位为片；`drug_name` 声明为 Identity。
`where.target` 与对象比较框架都读取 `fields.drug_name`。评分键也使用具体字段名。
这允许同一张图同时表达以 mg 和片计量的不同药品，各自对应自己的推荐引用。
只有参与同一个框架的相同数量类型需要同单位；properties 中同名字段保持同一口径。

单位由公开来源规范化后声明；当前检查单位字符串一致，不做换算或 UCUM 校验。
已知固定推荐 3 对应上下界均为 3；未知边界保留未知。类型检查约束字段声明，
不能验证抽取模型是否忠实理解了原文。

### 执行与构造对照

```python
from methods.frame_library import lower, solve

expanded = lower(graph)  # 既有 record graph，查询已变为原表达式
result = solve(graph)    # lower → shared record selection → compile → Z3
control = solve(graph, use_role_context=False)
```

`result` 包含 graph、selection、compiled、base_status、items。
关闭角色语境只跳过框架添加的 kind/modality 条件；引用原有条件、类型签名、公式、
来源事实和共享选择方法都相同。角色时间/主体等指称仍由引用提供。
改成外层否定不会改变绑定要求，查询真值不参与候选评分。

从仓库根目录运行 [frame_example.json](../../code/deepseek/frozen_code/definitions/frame_example.json)：

```sh
PYTHONPATH=code/deepseek/frozen_code python -B -m methods.frame_library
PYTHONPATH=code/deepseek/frozen_code python -B -m methods.frame_library --without-role-context
```

示例用同一批记录比较当前值、历史值、另一主体、另一模拟场景、通用推荐和检查
计划。历史/另一主体/另一场景被引用定位区分，实际与计划由框架角色区分。
为了观察机制，人工匹配分数有意让计划高于实际记录，两个运行使用相同分数。

| 结果 | 有角色约束 | 关闭角色约束 |
|---|---|---|
| 检查角色绑定 | actual_check | planned_check |
| 本次检查已完成 | not_determined | supported |
| 本次检查未完成 | not_determined | refuted |
| 单次量符合推荐 | refuted | refuted |
| 名义日总量符合推荐 | supported | supported |

这里计划目标被误当实际完成的差异说明角色约束确实参与绑定，不是 benchmark 效果
结论。原始事实没有变化；两种用量含义的区别来自 `2 × 3 = 3 × 2`，而 `2 ≠ 3`。

## 已准备的 12 条定义

| 关系 | 定义 ID | 参数角色与公式 |
|---|---|---|
| 对象匹配 | `object_match` | 待核对身份 `actual`、参照身份 `reference`：`actual = reference` |
| 集合成员 | `member_of` | 待核对身份、完整允许集合：`OR(actual = candidate)` |
| 闭区间 | `within_closed_interval` | 实际值、参照下界与上界：`lower ≤ value ≤ upper` |
| 单侧／严格比较 | `numeric_compare` | 实际值、参照阈值、比较方向：`value operator threshold` |
| 日总量计算 | `daily_total` | 同一记录的单次量与每日次数：`per_event × events_per_day` |
| 单次量与频次分别符合 | `dose_frequency_in_range` | 实际量、实际频次、同一参照的两组边界：两个闭区间判断的 AND |
| 日总量符合 | `daily_total_in_range` | 实际日总量、参照量和频次边界：`dose_lower × frequency_lower ≤ actual_total ≤ dose_upper × frequency_upper` |
| 事件完成 | `event_completed` | 指定事件的实际完成状态：`completed = true` |
| 否定 | `negate` | 被否定的完整命题：`NOT condition` |
| AND | `all_of` | 全部必要条件：`AND(conditions)` |
| OR | `any_of` | 全部备选条件：`OR(conditions)`，允许多项同时成立 |
| 条件规范适用 | `directive_active` | 来源意图与查询意图相同 AND 该规范在所指场景的适用条件成立 |

`Identity` 是已经对齐的 String 身份；`Number` 是 Int/Real 表达式；`Bool` 是布尔
表达式；`Comparison` 枚举 `eq/lt/le/gt/ge`；`List[T]` 是有限的显式列表。
每个参数的具体类型和角色都在 JSON 中，不需要从参数名猜测。
`daily_total` 返回数值，其余定义返回布尔表达式。

单次量、频次、日总量是有意分开的含义：不能仅凭“符合推荐”四个字决定使用哪一个，
要看命题要求核对什么。三个组合定义复用基本定义，不另写一套求解逻辑。

## 单次剂量符合推荐：一个完整展开

自然语言含义：实际单次量位于适用推荐的单次量闭区间内，包含端点。
三个参数均为 `Number`，实际量来自当前实际记录，两个边界来自同一个推荐记录。
原句若要求“不超过上限”，则选择 `numeric_compare` 的 `le`，不虚构下界。

```json
{
  "definition": "within_closed_interval",
  "arguments": {
    "value": ["ref", "actual", "dose"],
    "lower": ["ref", "advice", "dose_lower"],
    "upper": ["ref", "advice", "dose_upper"]
  }
}
```

展开成已有表达式：

```json
["and",
  ["ge", ["ref", "actual", "dose"], ["ref", "advice", "dose_lower"]],
  ["le", ["ref", "actual", "dose"], ["ref", "advice", "dose_upper"]]
]
```

两个推荐边界共享 `advice` 引用。现有记录选择器再将这个引用绑定到具体的来源记录。
固定推荐值 `v` 可用上下界都为 `v` 的区间表示。定义调用也可以嵌套，例如把
`daily_total` 的数值结果交给 `numeric_compare` 或 `daily_total_in_range`。

## 证据、未知值与作用域

参数类型和语义角色是绑定时应遵循的约定。展开器做参数名和枚举检查，Z3 处理底层
表达式类型；这不等于自动识别单位、药品别名、时间、模态或适用人群。上述内容
仍需从公开原文解释。定义库不包含药品推荐数值，也不读取私有评分答案。

未知值保留在有类型的来源字段中，例如 `{"completed": null}`，查询使用
`["ref", "actual_check", "completed"]`。缺省字段也采用同样的开放世界语义。
不要把字面量 `null` 写成表达式参数或生成 `x != null`。缺少完成信息时，
完成命题及其否定都可能为 `not_determined`；明确的完成记录为 true，明确的
未完成记录为 false。计划中的状态与实际完成状态属于不同记录。

未知也不意味着整句必然未知：`false AND unknown` 为 false，
`true OR unknown` 为 true。最终标签来自现有求解器检查事实是否蕴涵命题或其否定。

集合成员定义使用完整的有限枚举。`[]` 是明确的空集合，成员判断为 false，不能
用它表示“集合信息没找到”。单个成员身份未知可以使用有类型的引用；整个集合的
成员数和内容均未知时，本版有限枚举定义不直接表示它。空 AND 为 true，空 OR
为 false，均只适用于明确的空列表。

日总量乘积使用同一记录的单次量和日频次。建议的乘积仍为建议值；按需执行不能
无依据地转成每日实际次数。`daily_total_in_range` 仅在非负、有序区间且原文允许
按乘积区间判断时使用。若来源直接给出日总量范围，应直接用 `within_closed_interval`；
若来源列出多个有配对关系的离散方案，应分别判断每个方案并用 `any_of` 组合，
不能将所有方案合成一个大区间。

## 条件规范：与事实规则、实际事件分开

表达式接口使用嵌套 op/args。rules 中，
事实关系沿用 expr；禁用、建议、医嘱、慎用和明确允许改用如下规范实例：

```json
{
  "id": "N1",
  "text": "潮湿时禁止启动机器A",
  "modality": "prohibition",
  "action": "启动机器A",
  "when": {"op": "record", "args": ["当前设备A", "潮湿"]},
  "sources": [{"location": "manual/M1", "quote": "潮湿时禁止启动"}]
}
```

查询 `{"op":"directive_applies","args":["N1","prohibition"]}` 判断这条禁令
是否适用。程序把normative规则从事实前提中分离，按core.directive_active展开；
原文、action和when保留在record_graph.directives。模态还可为recommendation、
order、caution、permission；询问的模态不同就不是同一规范含义。

潮湿=true且实际启动=true时，两者可以同时成立，Z3前提仍可满足；启动是否发生
仍由实际记录决定。潮湿未知时适用性未知；潮湿=false时只是N1不适用，不能推出
所有规则都允许启动。规范的条件不会因原文提及而成为事实，也不做逆否推理。
条件表达式与动作指称仍需模型正确抽取；这不是自然语言核对器或失败修复流程。

## 在现有代码中使用

```python
from methods.definition_library import expand_queries
from methods import record_solver, scope_solver

expanded = expand_queries(graph)
selection = record_solver.select(expanded, mode="shared")
compiled = record_solver.compile_program(expanded, selection)
result = scope_solver.evaluate(compiled, {})
```

`expand_queries` 返回新图，只展开查询表达式。来源记录、规则与引用描述保持原值。
它不把待判断命题加入事实，也不选择记录。Agent 的框架路径由 `frame_library` 调用定义展开。

从仓库根目录运行构造例子：

```sh
PYTHONPATH=code/deepseek/frozen_code python -B -m methods.definition_library
```

`example.json` 使用虚构对象和数值。实际对象乙，每次 2 单位、每日 3 次；参照
对象甲，每次 3 单位、每日 2 次。计划要求完成检查，但实际完成情况未记录。

| 查询 | 结果 | 原因 |
|---|---|---|
| 对象一致 | `refuted` | 对象乙与对象甲不同 |
| 属于明确的对象甲／对象丙集合 | `refuted` | 对象乙不在集合内 |
| 单次量符合 | `refuted` | 2 不等于参照 3 |
| 单次量和频次分别符合 | `refuted` | 两个维度分别不符 |
| 仅名义日总量符合 | `supported` | `2 × 3 = 3 × 2`，这不表示整体方案符合 |
| 检查已完成 | `not_determined` | 缺少实际完成状态 |
| 检查未完成 | `not_determined` | 缺少状态也不能推出未完成 |
| 对象一致 AND 检查完成 | `refuted` | 已知第一个条件不成立 |
| 实际单次量为 2 OR 检查完成 | `supported` | 已知第一个条件成立 |

这些标签来自构造例子的定义展开。另一个完整的绑定修改与求解演示可从仓库根目录运行：

```sh
python scripts/demo.py --backbone deepseek
python scripts/demo.py --backbone qwen
```

演示读取 [graph.json](../../examples/illustration/graph.json) 和
[illustrative_diagnosis.json](../../examples/illustration/illustrative_diagnosis.json)，
在本地生成输出，不调用模型。
