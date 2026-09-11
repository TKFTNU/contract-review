# 合同智能审查系统

面向房屋租赁/买卖/委托/中介/保管类合同的**解析、结构化与多 Agent 审查**系统。核心目标：把一份合同变成**可追溯、可解释、可复核**的结构化审查报告——每一条结论都能定位到原文条款与法律依据。

**核心能力**

- 三种格式解析（DOCX / 文本型 PDF / TXT），保留页码、坐标与原文映射；
- 规则 + LLM 双引擎条款切分，带结构自校验与自动纠错回流；
- 统一数据契约（条款树 + 要素槽位 + 来源追踪）；
- 五个审查 Agent 组成的 LangGraph 统一图：理解 → 逐条/跨条款审查（并行）→ 法条检索与适用 → LLM 裁判 → 冲突仲裁 → 统一融合；
- 314 条法条的法律知识库 + 混合检索（BGE-M3 向量 + BM25），结论带《法名》第X条引用；
- 整体合理性结论（风险等级 + 0-100 评分 + 概述）与人工复核队列。

**里程碑进度**

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 文档解析与混合切分（规则 + LLM + 结构回流） | ✅ 已封版 |
| M2 | 合同统一数据结构（条款树 + 要素槽位契约） | ✅ 已完成 |
| M3 | 实体与关键要素提取（规则 + 模型双通道） | ✅ 已完成 |
| M4 | 逻辑一致性审查（金额/日期/主体/引用） | ✅ 已完成（已并入 M6 跨条款通道） |
| M5 | 法律知识库与 RAG（混合检索 + 七分类评估） | ✅ 已完成 |
| M6 | 统一图编排 + 跨模块冲突仲裁 + 结果融合 | ✅ 已完成 |
| M7 | 评测、报告与比赛演示 | 待启动 |

---

## 整体架构

```text
合同上传 → 文档解析 → 原子区块 → 规则/LLM切分(L) → 结构校验(C)
                                          ↑              │ 冲突或低置信度
                                          └──────────────┘   （C→L 自动回流）
                                                   │ 通过
                                                   ▼
                                    条款树 → 统一结构(M2) → Contract
                                                   ↓
┌─────────────────────── M6 统一图（LangGraph 编排）───────────────────────┐
│  ① 合同理解（唯一一次，仅语义层）                                          │
│        ↓                                                                  │
│  ┌─────┼──────────┐                                                       │
│ elements  ② 逐条审查  ③ 跨条款审查      ← 三路并行（M3 + 纯模型 × 2）      │
│（要素+数学信号）└──┬──┘                                                    │
│  └─────┼──────────┘                                                       │
│    风险候选合并 → ④ 法律检索与适用（七分类）→ ⑤ LLM 裁判（整体结论）         │
│                        ↑ 检索重试      ↑ 裁判重试 / 证据回环                │
│      → LOOP 跨模块冲突仲裁 →（实质冲突）带反馈回⑤重裁                       │
│      → F 统一融合（去重/交叉印证/排序/证据链）                              │
└───────────────────────────────────────────────────────────────────────────┘
                                                   ↓
                     UnifiedReviewReport + 人工复核队列 → 页面预览 / JSON
```

---

## 一、文档解析与切分（M1）

### 支持格式

| 格式 | 解析方式 | 保留信息 |
|---|---|---|
| DOCX | 标准库直接解析 OOXML（不依赖 python-docx） | 段落/表格顺序、Word 自动编号（numbering.xml）、标题样式、加粗/字号/缩进 |
| 文本型 PDF | PyMuPDF 按页解析 | 页码、文本块、坐标（bbox）；扫描页会告警 |
| TXT | 编码自动识别（UTF-8 / GB18030） | 行号 |

安全限制：文件 ≤25MB、DOCX 解压 ≤100MB、PDF ≤1000 页、文本 ≤500 万字符。

### 条款编号识别

支持：`第X章` `第X节` `第X条` `第X款`、`1.1`、`1.`、`一、`、`（一）`、`（1）`、附件等常见格式；并区分"编号后跟标题"与"编号后直接跟正文"两种情况（后者 heading 只保留编号，避免正文重复）。

### 两种切分引擎

| 引擎 | 原理 | 速度 | 适用 |
|---|---|---|---|
| 规则切分 | 编号/样式/版面规则 + 边界决策（CONTINUATION / SAME_CLAUSE / NEW_SUBCLAUSE / NEW_CLAUSE） | 毫秒级 | 规整合同；可叠加 LLM 复核低置信度边界 |
| LLM 语义分割 | 全部区块交给本地模型聚合为条款（label/title/level/block_ids） | 10~60 秒 | 无编号标题、编号风格混用、复杂版面 |

**LLM 分割的可靠性设计**：模型只负责语义判断，**覆盖完整性与原文一致性由本地保证**——模型漏填的区块归入最近条款、重复/越界 ID 丢弃、超长合同按规则识别的条款起点分批（每批附带上一批尾部区块作上下文）。实测压力测试（强制 6 批次）：区块覆盖 36/36、顺序一致、原文与切分后文本完全相同。

**C→L 结构回流闭环**：切分后执行结构一致性校验（区块遗漏/重复、层级断裂、编号重复、低置信度边界），只把**受影响的局部窗口**连同问题说明交回模型修正。回流默认 1 轮，**仅当可回流问题数严格减少时才采纳**，否则保留上一轮结果——回流只会让结果变好或不变，不会退化。

### 结构校验（`validation.py`）

区块遗漏/重复引用、边界数量匹配、父子层级异常、重复编号、小数编号父级缺失（条款已通过 parent_id 建树时不误报）。

---

## 二、合同统一数据结构（M2）

`Contract` 是全流程的接口契约（`contract_review/models.py`）：

```jsonc
{
  "schema_version": "m2",
  "contract_id": "c-53f3b9aed9e9",     // 文件名+全文哈希派生，同一合同稳定
  "filename": "合同.docx",
  "file_type": "docx",
  "stats": { "clauses": 26, "blocks": 36 },
  "elements": {                         // M3 填充（M6 审查时自动完成）
    "parties": [], "amounts": [], "dates": [], "subjects": []
  },
  "document": { /* 条款树、原文区块、边界决策、来源映射 */ }
}
```

**关键结构**

| 结构 | 字段 | 说明 |
|---|---|---|
| `SourceBlock` | block_id / order / kind / text / raw_text / page / metadata | 段落级原子区块；`text` 规范化文本与 `raw_text` 原文分开保存 |
| `Clause` | clause_id / label / title / level / heading / body / text / parent_id / block_ids / page_start / page_end / source_spans | 条款树节点，全部可回溯到区块与页码 |
| `BoundaryDecision` | boundary_id / 左右区块 / relation / confidence / decision_source / evidence / review_required | 相邻区块关系与判断依据（含 LLM 复核字段） |
| `Element` | element_id / kind / value / normalized / role / clause_id / block_ids / confidence / source | 要素记录，`source` 区分 `m3_rule` / `m3_llm` |

---

## 三、要素抽取（M3，`extractor.py`）

**双通道设计：规则先跑、模型补漏**

- **规则通道**（精确、免费、毫秒级）：当事人（角色词 + 签署区清洗 + 句子内截断）、金额（货币前缀/（小写）/零/负数/就近角色词 → 金额角色：合同总价/首付款/定金/押金/租金/违约金等）、日期（YYYY-MM-DD 归一化 + 就近角色词）、标的（坐落地址、建筑面积）。
- **模型通道**：规则覆盖不到的语义型标的与措辞变体，按 JSON Schema 输出，附原文片段与置信度。
- **合并去重**：同值要素按归一化键合并（`38500` 与 `38500.00` 视为同一），保留置信度更高者；模型不可用时保留规则结果，不阻塞。

**测试集评测**（合成数据集）：

| 引擎 | party F1 | amount F1 | date F1 | subject F1 |
|---|---|---|---|---|
| 仅规则（全量，0.7s） | 0.882 | 0.941 | 0.968 | 0.727 |
| 规则+LLM（对抗子集，93s） | **1.000** | 0.968 | **1.000** | **1.000** |

规则版 subject 召回低的原因是多模板合同的标的（撮合服务、数据集名称）无法用模式表达，恰由模型补齐。

---

## 四、综合审查：M6 统一图（LangGraph）

五个 Agent 收进**同一张图**，共享一次合同理解：

### 节点详解

| 节点 | 文件 | 职责 |
|---|---|---|
| ① understand | `agents/understanding.py` | 纯模型一次通读：合同类型、条款角色、主要义务、价款摘要（**不重复抽取当事人/标的**，那是 elements 的职责） |
| elements | `agents/elements.py` | 封装 M3 管道回填要素槽位；额外计算**数学信号**（金额≤0、极端金额、期限倒置、关键要素缺失） |
| ② clause_review | `agents/clause_review.py` | 逐条审查：显失公平、缺少要素、涉嫌违法、表述歧义等；**风险类型开放**（可自拟英文蛇形 code） |
| ③ cross_review | `agents/cross_review.py` | 跨条款一致性：金额口径冲突、日期矛盾、主体不一致、引用错误、语义互斥 |
| merge | `agents/merge.py` | 两路候选按（code + 条款重叠）去重合并，保留最强证据与全部来源标记 |
| ④ legal_apply | `agents/legal.py` | 混合检索法条 → LLM **七分类评估**；引用只能来自召回池 |
| ⑤ judge | `agents/judge.py` | LLM 逐条裁决（保留/严重度/转人工）+ 整体合理性结论 |
| arbitrate | `agents/arbitrate.py` | 跨模块冲突仲裁：类型/严重度不一致、要素与结论矛盾、结论互斥 |
| fuse | `agents/fuse.py` | 确定性融合：去重 / 交叉印证 / 排序 / 统一证据链 / 复核队列增补 |

### 七分类评估（取代二元"违法/不违法"）

`legal_risk` 法律风险 · `commercial_unreasonable` 商业不合理 · `rights_imbalance` 权利义务失衡 · `performance_risk` 履约风险 · `insufficient_facts` 事实不足 · `need_more_material` 需要补充材料 · `no_substantive_risk` 未发现实质风险。

**不违法 ≠ 合理**——除最后一类外全部进入最终报告。

### 三条护栏（代码层硬约束，模型无法绕过）

1. 评估非「无实质风险」的候选**永不删除**（模型裁决也覆盖不了）；
2. 裁判不可用时**保守保留全部候选**转人工；
3. 裁判未返回裁决的候选项**同样保留**转人工。

### 要素信号与交叉印证

数学信号**不生成风险**（风险检测纯模型），仅用于两处：
- 同一条款有模型风险命中 → 标记 `cross_validated=true` 并提升置信度（数学事实独立佐证）；
- 未被任何风险覆盖 → 以 `element_signal:*` 进入人工复核队列（不静默忽略）。

### 四条有界回环（计数在报告 `stats.loops`）

| 回环 | 触发条件 | 上限 |
|---|---|---|
| 检索重试（④→④） | 候选无评估结论（换宽查询重检） | 1 |
| 裁判重试（⑤→⑤） | 裁判模型调用失败 | 1 |
| 证据回环（⑤→②） | 裁判标记「事实不足/需补充材料」 | 1 |
| 仲裁重裁（arbitrate→⑤） | 仲裁发现实质冲突 | 1 |

### 统一报告（`schema_version = "m6"`）

```jsonc
{
  "schema_version": "m6",
  "contract_id": "c-...", "filename": "...",
  "understanding": { "contract_type": "...", "clause_roles": [...], ... },
  "elements": { "parties": [...], "amounts": [...], "dates": [...], "subjects": [...] },
  "element_signals": [ { "kind": "amount", "note": "租金为 0 元（非正数）", "clause_id": "C0003" } ],
  "m3_stats": { "rule_elements": 11, "llm_elements": 7, "merged_elements": 7, ... },
  "risks": [ {
      "risk_id": "R0001", "code": "zero_rent_with_high_deposit",
      "category": "commercial_unreasonable", "category_label": "商业不合理",
      "severity": "high", "message": "...",
      "citations": [ { "law_name": "中华人民共和国民法典", "article_no": "第七百二十一条", "quote": "..." } ],
      "clause_ids": ["C0003"], "block_ids": ["B0010"],
      "confidence": 0.9, "review_required": false, "cross_validated": true,
      "rationale": "裁判理由", "agent_sources": ["clause_llm"]
  } ],
  "conflicts": [ { "type": "severity_mismatch", "description": "...", "resolution": "..." } ],
  "overall": { "overall_assessment": "high_risk", "overall_score": 5, "summary": "合同整体..." },
  "review_queue": [ { "risk_id": "R0002", "code": "...", "reason": "..." } ],
  "stats": { "...": "...", "loops": { "legal_retries": 0, "judge_retries": 0,
             "evidence_rounds": 1, "arbitration_rounds": 0 } },
  "errors": []
}
```

`overall_score` 为**合理性评分**（0-100，越高越合理）。

---

## 五、法律知识库与检索（M5）

### 法条库（`data/法条结构化数据.jsonl`，314 条）

| 来源 | 条数 |
|---|---|
| 《中华人民共和国民法典》合同编相关 | 272 |
| 最高法商品房买卖合同司法解释 | 25 |
| 最高法城镇房屋租赁合同司法解释 | 17 |

每条字段：`article_uid / law_name / article_no / content / authority_level / jurisdiction / effective_from / effective_to / status / contract_types / risk_tags / source_url / version_id`。

### 混合检索（`legal_kb.py`）

```text
法条 JSONL → Qdrant local（data/legal_index/，无需服务器）
   ├── dense: Ollama bge-m3（1024 维余弦）
   └── bm25:  字级 unigram+bigram BM25
              （文档侧 tf饱和×idf，查询侧全 1，sparse dot 在 Qdrant 内复现 BM25 分数）
查询 → dense + bm25 双路 prefetch → Qdrant Query API RRF 融合 → top-k
```

**索引构建**（法条更新后执行一次）：

```powershell
python scripts/build_legal_index.py   # 输出：载入 314 条 → 词表 → 向量 → 写入索引
```

**召回率评测**（强/弱两级金标准，27 类风险按 issue_code 独立查询）：

```powershell
python scripts/eval_legal_retrieval.py --top-k 5
# 强金标准（法条直接对应风险）：Recall@5=0.567，MRR=0.900
```

**关键设计**：检索默认**不做 contract_types 硬过滤**——元数据标注覆盖不全，硬过滤会漏掉跨类型法条（如 597 条"无处分权"标注为买卖专用，但租赁标的无权属同样适用）。

---

## 六、数据资产（`data/`）

### 合成测试合同（`data/generated_contracts/`）

50 份合同（每份 DOCX + PDF 双格式，共 100 个文件），基于国家市场监督管理总局合同示范文本库结构合成：

| 数据集 | 数量 | 说明 |
|---|---|---|
| `samr_multi_template_2025/` | 20 份 | 五类模板：城镇房屋租赁 / 委托 / 中介 / 数据提供 / 保管 |
| `samr_multi_template_2025/adversarial/` | 10 份 | 对抗样本：注入 27 种风险（见下） |
| `samr_sdf_2025_0002/` | 20 份 | 山东省新建商品房买卖合同（预售）模板 |

每份合同在 `manifest.json` 中记录：模板来源、当事人、金额、日期、`issue_codes`（风险标签）与 `issue_summary`。

### 对抗样本的 27 种风险标签

```text
term_over_20y  zero_rent  negative_fee  extreme_rent  zero_deposit  zero_fee  zero_quantity
zero_records   date_reversed  impossible_deadline  fee_mismatch  quantity_value_mismatch
unverified_property  unilateral_entry  unlimited_authority  no_report  insecure_delivery
commission_100pct  negative_commission  success_guarantee  cash_only  nonexistent_subject
perpetual_license  personal_data_no_consent  public_disclosure  liability_exemption
retrieval_before_storage
```

`issue_codes` 同时是检索评测的金标准（映射到法条 `risk_tags`）。

### 演示样例（`samples/tech-development-contract.txt`）

一份专门用于对比两种切分引擎的样例合同，其中"通知与送达""违约责任""争议解决"是三个**没有编号**的标题：

- **规则引擎**会把它们并入上一条（"第六条 保密义务"吞掉 8 个区块）；
- **LLM 语义分割**会把它们识别为独立条款，并保持 `1.1`/`（一）` 的层级结构。

### 本地 LLM 配置

模型服务地址、模型名、API Key 与 Embedding 配置统一放在项目根目录的 `.env`（模板 `.env.example`，所有键均可省略）。优先级：**命令行参数 > 真实环境变量 > `.env` > 内置默认值**；页面侧栏、命令行与 Web 前端共用这套配置。模型参与切分的两种方式：

1. **LLM 语义分割**（推荐）：全量区块交模型聚合为条款，本地保证覆盖完整性；
2. **低置信度边界复核**（`llm_review.py`）：规则切分后只把低置信度边界连同局部上下文交模型判断关系与标题。

模型服务需单独启动（加载约 22GB 权重），Demo 不会自动下载；服务离线时全部功能自动降级、不阻塞页面。

聊天模型与向量模型各有一份独立的 API Key 配置（`CONTRACT_LLM_API_KEY` / `CONTRACT_EMBED_API_KEY`），本地部署均留空即可。向量端配置了 Key 时，Embedding 请求会自动携带 `Authorization: Bearer <key>` 头，兼容需要鉴权的推理网关。

---

## 七、快速开始

### 环境要求

- Windows（已在 Python 3.13 + PowerShell 验证）/ Linux / macOS
- 虚拟环境 `.venv` 已包含全部依赖（qdrant-client、langgraph、streamlit、PyMuPDF）
- 两个本地模型服务（地址与模型名在 `.env` 中配置，默认如下）：
  - **LM Studio**：`http://localhost:12345`，模型 `qwen3.8-27b-nvfp4-mtp`（审查用）
  - **Ollama**：`http://localhost:11434`，模型 `bge-m3`（检索向量用）

```powershell
# 新环境安装
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 启动 Web 前端（推荐）

```powershell
python -m contract_review.web_server          # 默认 8600 起，被占用自动顺延
python -m contract_review.web_server --port 8700   # 指定端口
```

打开控制台提示的地址 → 「综合审查（M6）」页签 → 运行。**8600 常被 Windows 系统占用**，实际端口以控制台输出为准。

### 停止服务

```powershell
Get-NetTCPConnection -LocalPort 8601 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## 八、使用入口

### Web 单页（`web_server.py` + `webui.html`）

打开即自动载入内置样例合同并完成规则切分，页面能力：

- **条款树**：按层级缩进，关键词筛选，逐条展开看正文与来源区块；
- **边界决策**：可只看待复核；显示关系/置信度/判断来源/证据；
- **原文区块**：全部原子区块与页码；
- **统一结构（M2）**：schema/合同ID/要素槽位状态；「抽取要素（M3）」按钮单独跑要素抽取（快，数秒）；
- **综合审查（M6）**：整体合理性结论（等级+评分+概述）、要素概览、逐条风险（类别/法条引用/交叉印证标记/裁判理由）、仲裁记录、人工复核队列；
- **结构化 JSON** 预览与下载；支持拖拽上传 DOCX / PDF / TXT。

### HTTP API 参考

| 接口 | 方法 | 请求 | 响应 |
|---|---|---|---|
| `/` | GET | — | 前端页面 |
| `/health` | GET | — | `ok` |
| `/api/sample` | POST | `{}` | `{doc_id, filename, result}`（内置样例） |
| `/api/upload` | POST | multipart 文件 | `{doc_id, filename, result}` |
| `/api/parse` | POST | `{doc_id}` | 规则切分结果 |
| `/api/split` | POST | `{doc_id}` | LLM 语义分割结果 + summary |
| `/api/contract` | POST | `{doc_id}` | M2 统一结构 |
| `/api/extract` | POST | `{doc_id, enable_llm?}` | M3 要素抽取结果 + summary |
| `/api/review` | POST | `{doc_id, enable_llm?}` | **M6 统一报告** + summary |
| `/api/legal` | POST | 同上 | 同 `/api/review`（兼容别名） |

### 命令行（`cli.py`）

```powershell
# 仅规则切分
python -m contract_review.cli sample.docx -o sample.parsed.json

# LLM 全量语义分割（含结构回流）
python -m contract_review.cli sample.docx --llm -o sample.llm.json
python -m contract_review.cli sample.docx --llm --reflow-rounds 0   # 关闭回流

# 输出 M2 统一结构 / 单独跑 M3 要素抽取
python -m contract_review.cli sample.docx --contract -o sample.contract.json
python -m contract_review.cli sample.docx --extract -o sample.extracted.json
python -m contract_review.cli sample.docx --extract --rules-only    # 只用规则

# 综合审查（M6 统一图，结果挂在单一 review 字段）
python -m contract_review.cli sample.docx --review -o sample.review.json
python -m contract_review.cli sample.docx --legal                   # 同义（兼容旧参数）
```

**参数说明**

| 参数 | 说明 |
|---|---|
| `--llm` | LLM 语义分割（默认规则切分） |
| `--reflow-rounds N` | 结构回流轮数上限（默认 1，0 关闭） |
| `--contract` / `--extract` | 输出 M2 结构 / 跑 M3 要素抽取（被 `--review` 隐含） |
| `--review` / `--legal` | 运行 M6 综合审查（`--legal` 为同义别名） |
| `--rules-only` | 关闭模型：要素抽取只用规则；审查走保守保留 |
| `--endpoint` / `--model` / `--api-key` / `--timeout` | 模型服务配置（默认读 `.env`，未配置时 `http://localhost:12345` + `qwen3.8-27b-nvfp4-mtp`） |

### Streamlit 界面（`app.py`）

```powershell
python -m streamlit run app.py
```

> 已知兼容性：桌面内置浏览器与 Streamlit 1.63 原生文件上传存在会话问题，Demo 使用仅绑定 `127.0.0.1` 的轻量上传入口传递文件（带 nonce + 一次性令牌 + 30 分钟 TTL），上传后自动返回解析页，避免 `Connection lost`。

---

## 九、项目结构

```text
l:/contract/
├── app.py                          Streamlit 界面（7 个页签，含综合审查）
├── requirements.txt                依赖清单
├── .env.example                    模型服务配置模板（复制为 .env 使用，不进版本控制）
├── contract_review/
│   ├── settings.py                 运行配置读取（.env/环境变量，零依赖）
│   ├── models.py                   M2 数据契约（SourceBlock/Clause/BoundaryDecision/Element/Contract）
│   ├── parsers.py                  DOCX/PDF/TXT 解析 + 上传校验（安全限额）
│   ├── splitter.py                 规则切分（编号识别/边界决策/条款树重建）
│   ├── validation.py               结构一致性校验（供 C→L 回流消费）
│   ├── llm_splitter.py             LLM 语义分割 + 分批 + 上下文 + C→L 回流闭环
│   ├── llm_review.py               低置信度边界复核（复用共享客户端）
│   ├── llm_client.py               共享 LLM 客户端（关思考/JSON Schema/HTTP 重试）
│   ├── extractor.py                M3 要素抽取（规则 + 模型双通道 + 合并去重）
│   ├── reviewer.py                 M4 一致性检查器（历史模块，保留独立可用）
│   ├── legal_kb.py                 M5 检索层（BM25 + Ollama bge-m3 + Qdrant RRF）
│   ├── agents/                     M6 统一图（见下）
│   │   ├── state.py                ReviewState + RiskCandidate/FinalRisk/ElementSignal/Conflict + 配置
│   │   ├── understanding.py        ① 合同理解
│   │   ├── elements.py             要素抽取节点（M3 + 数学信号）
│   │   ├── clause_review.py        ② 逐条审查
│   │   ├── cross_review.py         ③ 跨条款审查
│   │   ├── merge.py                风险候选合并
│   │   ├── legal.py                ④ 法律检索与适用（七分类）
│   │   ├── judge.py                ⑤ 最终裁判（护栏 + 整体结论）
│   │   ├── arbitrate.py            跨模块冲突仲裁
│   │   ├── fuse.py                 统一融合
│   │   └── graph.py                LangGraph 编排 + run_unified_review 入口 + 报告装配
│   ├── web_server.py               标准库 HTTP 服务（页面 + API）
│   ├── webui.html                  单页前端（零构建）
│   ├── upload_server.py            轻量上传服务（Streamlit 兼容层）
│   ├── service.py                  解析编排入口 parse_contract / build_contract
│   └── cli.py                      命令行入口
├── scripts/
│   ├── build_legal_index.py        法条索引构建（314 条 → Qdrant local）
│   ├── eval_legal_retrieval.py     检索召回率评测（强/弱金标准，Recall@k + MRR）
│   └── generate_*.py               合成数据集生成脚本（三类模板）
├── checks/                         88 个校验用例（unittest，全离线可跑）
├── data/
│   ├── 法条结构化数据.jsonl         314 条法条
│   ├── generated_contracts/        50 份合成合同（DOCX+PDF+manifest 标注）
│   └── legal_index/                Qdrant 本地索引（构建产物）
├── samples/                        演示样例合同
└── .review_challengecup/           比赛申请材料
```

---

## 十、关键设计决策记录

| 决策 | 原因 |
|---|---|
| 关闭模型思考用 `reasoning_effort: "none"` | 本机 LM Studio 实测：`chat_template_kwargs.enable_thinking`、`/no_think` 提示词、顶层 `enable_thinking` **均无效**，只有前者生效；客户端自动兼容拒绝该参数的 vLLM/SGLang |
| 检索用 Qdrant **local mode** | 314 条规模无需服务器；`pip install qdrant-client` 即用，数据在 `data/legal_index/`。代价是目录独占锁——由编排层统一创建/`finally` 关闭（`run_unified_review`），Web 服务用进程级共享检索器 + 串行锁 |
| 风险类型**开放集合** | 真实合同的风险远超任何预置清单；模型可自拟英文蛇形 code（如 `waiver_of_termination_right`），只提供建议类别作提示 |
| 要素数学信号**不生成风险** | 风险检测纯模型完成（用户原则）；日期倒置/金额≤0 这类**数学事实**只做独立佐证（交叉印证）与复核队列增补，不做关键词枚举 |
| 七分类 + 三条护栏 | "不违法 ≠ 合理"；防止"商业不合理但未违法"的风险被模型否决后静默删除 |
| 引用只能来自召回池 | 防编造法条；无合适法条时宁可无引用，也不虚构《法名》第X条 |
| 回环全部有界 | 检索/裁判/证据/仲裁各 1 次上限，计数在 state 显式递增，保证图必然终止 |
| `Contract` 稳定 ID | 文件名+全文哈希派生，同一合同跨运行同 ID，便于评测比对 |
| M6 直接替换 M4/M5 分散字段 | 一份合同一份报告（`review` 单一字段），前端/CLI/API 三处入口统一；`--legal` 与 `/api/legal` 保留兼容别名 |
| `reviewer.py` / `extractor.py` 保留 | M4/M3 层的独立实现仍然可用（测试继续覆盖）；统一图只通过 `extract_elements` 复用 M3 管道 |

---

## 十一、校验用例（`checks/`）

```powershell
.venv\Scripts\python.exe -m unittest discover -s checks -p "check_*.py" -v   # 88 个用例
```

| 校验文件 | 覆盖 |
|---|---|
| `check_parser.py` | 正文编号、Word 自动编号、同行正文、动态层级、来源追溯 |
| `check_validation.py` | 校验问题分类、可修复性标记 |
| `check_llm_splitter.py` | 分割容错/回退、C→L 回流（修复/回滚/关闭）、分批切点 |
| `check_llm_review.py` | 低置信度边界复核、失败保留 |
| `check_contract.py` | M2 结构与序列化兼容 |
| `check_extractor.py` | M3 角色识别、归一化、零金额、去重合并、模型降级 |
| `check_reviewer.py` | M4 四类检查器正反例、LLM 合并、降级 |
| `check_legal_kb.py` | BM25 打分、分词、RRF 融合、保存加载 |
| `check_agents.py` | M6 统一图：纯模型无规则来源、七分类护栏、no_substantive_risk 可删、整体结论、要素信号交叉印证、信号进复核队列、仲裁重裁、检索重试、证据回环、裁判失败保守保留、全降级、报告结构 |
| `check_upload_server.py` | multipart 解析、一次性令牌 |

全部用例离线可跑（模型调用均使用 Fake transport / Fake KB）。

---

## 十二、已知边界

**解析层**

- DOCX 不提供稳定页码，按段落/表格顺序定位；
- PDF 复杂双栏/页眉页脚/表格需后续版面分析；扫描件未接 OCR（会提示无文本页）；
- Word 编号解析覆盖常见定义，复杂编号重启与自定义域仍需扩充测试。

**要素层**

- 规则金额抽取只收带货币标记或小数形式的"0 元"，无标记的整数"0 元"被当噪音排除；
- 多模板数据集的 amount_conflict 属"押金值偏离金标准"型，文本内无对照值——规则与模型都无法检出（评测已知边界）。

**审查层**

- 法条库覆盖 5 类合同的核心条文（314 条），未覆盖全部房地产监管规则；
- 检索强金标准 Recall@5=0.567（`unverified_property` 等跨类型引用仍是难点），MRR=0.900；
- 零租金等极端条款若模型未报出，仅以 `element_signal` 形式进入复核队列（不会作为风险结论）；
- 模型服务离线时审查降级为保守保留，不产生实质判定。

**未实现（M7）**

- 边界 F1 / 端到端指标等离线评测体系、HTML/PDF 报告呈现、比赛演示、人工纠错数据回流（H/DATA/EVAL 节点）。

---

## 十三、常见问题

**Q：跑审查需要先手动运行「统一结构（M2）」页签吗？**
不需要。运行综合审查时，M2 构建与 M3 要素抽取都在统一图内部自动完成；「统一结构（M2）」页签是独立的调试/查看工具。

**Q：端口不是 8600？**
Windows 系统常占用 8600，服务会自动顺延（8601、8602…），以控制台输出为准。

**Q：报 `Storage folder ... already accessed by another instance`？**
Qdrant local 目录独占锁冲突——有残留的审查进程未退出。结束对应 python 进程后重试。

**Q：模型服务不可用会怎样？**
不会崩溃：理解返回显式 `unavailable`、检索/裁判失败保守保留候选、全部疑点进人工复核队列并在报告 `errors` 中说明。
