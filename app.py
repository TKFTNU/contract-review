from __future__ import annotations

import json
import html
from hashlib import sha256

import streamlit as st

from contract_review import settings
from contract_review.extractor import M3Config, extract_elements
from contract_review.parsers import ParseError
from contract_review.agents import AgentReviewConfig, run_unified_review
from contract_review.llm_review import LLMReviewConfig, review_contract_boundaries
from contract_review.llm_splitter import LLMSplitConfig, split_contract_with_llm
from contract_review.service import build_contract, parse_contract
from contract_review.upload_server import (
    claim_upload,
    start_upload_server,
    upload_page_url,
)


st.set_page_config(page_title="合同解析 Demo", page_icon="📄", layout="wide")

st.title("合同解析与条款切分")
st.caption("上传 DOCX、文本型 PDF 或 TXT，生成可回溯、可解释的结构化条款。")


@st.cache_data(show_spinner=False)
def parse_uploaded_contract(filename: str, content: bytes):
    """Avoid repeating parsing (and future LLM calls) on every UI rerun."""

    return parse_contract(filename, content)

with st.sidebar:
    st.subheader("当前能力")
    st.markdown(
        """
        - DOCX：按段落和表格提取
        - PDF：按页面和文本块提取
        - TXT：兼容 UTF-8、GB18030
        - 识别正文编号与 Word 自动编号
        - 保存原文、规范化文本和来源范围
        - 输出边界置信度与复核标记
        """
    )
    st.info("扫描型 PDF 暂不自动 OCR；系统会标记未提取到文本的页面。")
    st.divider()
    st.subheader("切分引擎")
    engine = st.radio(
        "选择条款切分方式",
        ["规则切分", "LLM 语义分割"],
        help="规则切分毫秒级完成；LLM语义分割会调用本地模型，"
        "能识别无编号标题、编号风格混用等规则难以判断的边界。",
    )
    llm_endpoint = settings.llm_endpoint()
    llm_model = settings.llm_model()
    llm_api_key = settings.llm_api_key()
    llm_timeout = settings.llm_timeout(300)
    enable_llm = False
    if engine == "LLM 语义分割":
        llm_endpoint = st.text_input("模型服务地址", value=llm_endpoint)
        llm_model = st.text_input("模型名称", value=llm_model)
        llm_api_key = st.text_input(
            "API Key（本地服务通常留空）", type="password"
        )
        llm_timeout = st.number_input(
            "单批超时（秒）", min_value=30, max_value=1800, value=300, step=30
        )
        st.caption(
            "兼容 LM Studio / vLLM / SGLang 的 OpenAI 接口；"
            "超长合同会按条款边界分批请求。"
        )
    else:
        enable_llm = st.checkbox(
            "启用低置信度边界复核",
            value=False,
            help="只有点击运行复核后，才会向配置的模型地址发送待判断文本。",
        )
        if enable_llm:
            llm_endpoint = st.text_input("模型服务地址", value=llm_endpoint)
            llm_model = st.text_input("模型名称", value=llm_model)
            llm_api_key = st.text_input(
                "API Key（本地服务通常留空）", type="password"
            )
            llm_timeout = st.number_input(
                "单批超时（秒）", min_value=30, max_value=900, value=300, step=30
            )
            st.caption("兼容 LM Studio / vLLM / SGLang 的 OpenAI 接口。")

try:
    upload_port, upload_nonce = start_upload_server()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

upload_token = str(st.query_params.get("upload_token", ""))
if upload_token:
    claimed_upload = claim_upload(upload_token)
    if claimed_upload is not None:
        st.session_state["uploaded_contract"] = (
            claimed_upload.filename,
            claimed_upload.content,
        )

uploaded_contract = st.session_state.get("uploaded_contract")
if uploaded_contract is None:
    upload_url = upload_page_url(upload_port, upload_nonce)
    st.markdown(
        f"""
        <a href="{html.escape(upload_url)}" target="_self" style="
            display:block;text-align:center;text-decoration:none;padding:14px 18px;
            border:1px dashed #8fa1b7;border-radius:12px;color:inherit;
            background:#f9fbfd;font-weight:600;">
            选择 DOCX、PDF 或 TXT 合同文件
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.caption("文件通过本机上传服务传递，避免浏览器与Streamlit上传会话断开。")
    st.markdown("### 使用步骤")
    st.write("1. 上传合同；2. 查看切分统计；3. 核对条款；4. 下载 JSON。")
    st.stop()

uploaded_name, uploaded_content = uploaded_contract
upload_columns = st.columns([5, 1])
upload_columns[0].success(
    f"已上传：{uploaded_name}（{len(uploaded_content) / 1024:.1f} KB）"
)
if upload_columns[1].button("更换文件", use_container_width=True):
    st.session_state.pop("uploaded_contract", None)
    st.query_params.clear()
    st.rerun()

try:
    with st.spinner("正在解析文档并识别条款……"):
        base_result = parse_uploaded_contract(uploaded_name, uploaded_content)
except ParseError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # Keep the demo usable while exposing actionable context.
    st.exception(exc)
    st.stop()

result = base_result
review_summary = None
split_summary = None
document_hash = sha256(uploaded_content).hexdigest()[:16]

if engine == "LLM 语义分割":
    split_state_key = f"llm-split:{document_hash}:{llm_endpoint}:{llm_model}"
    run_split = st.button(
        f"运行 LLM 语义分割（{len(base_result.blocks)} 个区块）",
        type="primary",
        use_container_width=True,
    )
    if run_split:
        split_config = LLMSplitConfig(
            endpoint=llm_endpoint,
            model=llm_model,
            api_key=llm_api_key,
            timeout_seconds=int(llm_timeout),
        )
        try:
            with st.spinner("本地模型正在切分全部条款……"):
                split_result, split_summary = split_contract_with_llm(
                    base_result, split_config
                )
            st.session_state[split_state_key] = (split_result, split_summary)
        except Exception as exc:
            st.error(f"LLM分割失败，已保留规则切分结果：{exc}")
    cached_split = st.session_state.get(split_state_key)
    if cached_split:
        result, split_summary = cached_split
elif enable_llm:
    review_state_key = f"llm-review:{document_hash}:{llm_endpoint}:{llm_model}"
    pending_count = base_result.stats["review_boundaries"]
    if pending_count == 0:
        st.success("所有边界都已达到规则自动接受阈值，无需调用LLM。")
    else:
        run_review = st.button(
            f"运行 LLM 复核（{pending_count}个边界）",
            type="primary",
            use_container_width=True,
        )
        if run_review:
            config = LLMReviewConfig(
                endpoint=llm_endpoint,
                model=llm_model,
                api_key=llm_api_key,
                timeout_seconds=int(llm_timeout),
            )
            try:
                with st.spinner("本地模型正在批量复核低置信度边界……"):
                    reviewed_result, review_summary = review_contract_boundaries(
                        base_result, config
                    )
                st.session_state[review_state_key] = (
                    reviewed_result,
                    review_summary,
                )
            except Exception as exc:
                st.error(f"LLM复核失败，已保留规则切分结果：{exc}")
        cached_review = st.session_state.get(review_state_key)
        if cached_review:
            result, review_summary = cached_review

if split_summary is not None:
    st.success(
        f"LLM 语义分割完成：{split_summary.blocks} 个区块 → "
        f"{split_summary.llm_clauses} 条条款"
        f"（规则基线 {split_summary.rule_clauses} 条），"
        f"批次 {split_summary.succeeded_batches}/{split_summary.batches} 成功。"
    )
    if split_summary.issues_before or split_summary.reflow_rounds:
        st.info(
            f"结构回流：可回流问题 {split_summary.issues_before} → "
            f"{split_summary.issues_after}，共 {split_summary.reflow_rounds} 轮、"
            f"重标注 {split_summary.reflowed_blocks} 个区块。"
        )
        for note in split_summary.reflow_notes:
            st.caption(f"· {note}")
    if split_summary.errors:
        with st.expander("LLM分割提示", expanded=True):
            for error in split_summary.errors:
                st.warning(error)

if review_summary is not None:
    if review_summary.accepted:
        st.success(
            f"LLM已接受 {review_summary.accepted}/{review_summary.requested} 个边界判断，"
            f"仍有 {review_summary.unresolved} 个待复核。"
        )
    if review_summary.errors:
        with st.expander("LLM复核提示", expanded=not review_summary.accepted):
            for error in review_summary.errors:
                st.warning(error)

if result.warnings:
    for warning in result.warnings:
        st.warning(warning)

stats = result.stats
metric_columns = st.columns(5)
metric_columns[0].metric("字符数", f"{stats['characters']:,}")
metric_columns[1].metric("原文区块", stats["blocks"])
metric_columns[2].metric("识别条款", stats["clauses"])
metric_columns[3].metric("待复核边界", stats["review_boundaries"])
metric_columns[4].metric("PDF页数", stats["pages"] or "—")

overview_tab, clause_tab, boundary_tab, block_tab, contract_tab, review_tab, json_tab = st.tabs(
    ["结构概览", "条款预览", "边界决策", "原文区块", "统一结构（M2）", "综合审查（M6）", "结构化 JSON"]
)

with overview_tab:
    rows = [
        {
            "条款ID": clause.clause_id,
            "编号": clause.label,
            "标题": clause.title,
            "层级": clause.level,
            "父条款": clause.parent_id or "—",
            "页码": str(
                f"{clause.page_start}-{clause.page_end}"
                if clause.page_start and clause.page_end != clause.page_start
                else clause.page_start or "—"
            ),
            "字符数": len(clause.text),
        }
        for clause in result.clauses
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

with clause_tab:
    query = st.text_input("筛选条款", placeholder="输入编号、标题或正文关键词")
    query = query.strip().lower()
    visible_clauses = [
        clause
        for clause in result.clauses
        if not query
        or query in clause.label.lower()
        or query in clause.title.lower()
        or query in clause.text.lower()
    ]
    st.caption(f"显示 {len(visible_clauses)} / {len(result.clauses)} 个条款")
    for clause in visible_clauses:
        page_label = f" · 第{clause.page_start}页" if clause.page_start else ""
        parent_label = f" · 父条款 {clause.parent_id}" if clause.parent_id else ""
        display_title = clause.heading or clause.title or clause.label
        with st.expander(f"{clause.clause_id} · {display_title}{page_label}{parent_label}"):
            st.text(clause.text)
            st.caption("来源区块：" + "、".join(clause.block_ids))

with boundary_tab:
    only_review = st.toggle("只显示待复核边界", value=False)
    visible_boundaries = [
        boundary
        for boundary in result.boundaries
        if not only_review or boundary.review_required
    ]
    if split_summary is not None:
        boundary_note = "边界关系来自LLM语义分割结果。"
    elif enable_llm:
        boundary_note = "低置信度项可点击上方按钮交给本地模型复核。"
    else:
        boundary_note = "低置信度项建议改用LLM语义分割。"
    st.caption(
        f"显示 {len(visible_boundaries)} / {len(result.boundaries)} 个相邻区块边界；"
        + boundary_note
    )
    boundary_rows = [
        {
            "边界ID": boundary.boundary_id,
            "左区块": boundary.left_block_id,
            "右区块": boundary.right_block_id,
            "关系": boundary.relation,
            "置信度": boundary.confidence,
            "判断来源": boundary.decision_source,
            "规则判断": boundary.rule_relation or "—",
            "待复核": "是" if boundary.review_required else "否",
            "判断依据": "；".join(boundary.evidence),
            "LLM理由": boundary.llm_reason or "—",
            "提示": "；".join(boundary.warnings),
        }
        for boundary in visible_boundaries
    ]
    st.dataframe(boundary_rows, width="stretch", hide_index=True)

with block_tab:
    for block in result.blocks:
        location = f"第{block.page}页" if block.page else f"顺序{block.order}"
        st.markdown(f"**{block.block_id} · {block.kind} · {location}**")
        st.text(block.text)
        source_id = block.metadata.get("source_block_id", block.block_id)
        line_start = block.metadata.get("source_line_start")
        line_end = block.metadata.get("source_line_end")
        source_label = f"来源 {source_id}"
        if line_start is not None:
            source_label += f" · 行 {line_start}-{line_end}"
        st.caption(source_label)
        if block.raw_text != block.text:
            with st.expander("查看未经规范化的原文"):
                st.text(block.raw_text or "")
        st.divider()

with contract_tab:
    contract = build_contract(result)
    st.caption(
        f"schema {contract.schema_version} · 合同ID {contract.contract_id} · "
        "要素抽取先跑规则再调用模型，模型不可用时保留规则结果。"
    )

    extract_columns = st.columns([1, 3])
    if extract_columns[0].button(
        "抽取要素（M3）", type="primary", use_container_width=True
    ):
        try:
            with st.spinner("正在抽取当事人、金额、日期与标的……"):
                extracted, m3_summary = extract_elements(
                    contract,
                    M3Config(
                        endpoint=llm_endpoint,
                        model=llm_model,
                        api_key=llm_api_key,
                        timeout_seconds=int(llm_timeout),
                    ),
                )
            st.session_state["m3-result"] = (document_hash, extracted)
            st.success(
                f"要素抽取完成：规则 {m3_summary.rule_elements} 个 + "
                f"模型 {m3_summary.llm_elements} 个 → 合并 "
                f"{m3_summary.merged_elements} 个。"
            )
            for error in m3_summary.errors:
                st.warning(error)
        except Exception as exc:  # Keep the page usable when the model is down.
            st.error(f"要素抽取失败，保留空槽位：{exc}")

    cached_m3 = st.session_state.get("m3-result")
    if cached_m3 and cached_m3[0] == document_hash:
        contract = cached_m3[1]

    slot_columns = st.columns(4)
    slot_columns[0].metric("当事人", len(contract.elements.parties))
    slot_columns[1].metric("金额", len(contract.elements.amounts))
    slot_columns[2].metric("日期", len(contract.elements.dates))
    slot_columns[3].metric("标的", len(contract.elements.subjects))
    st.json(contract.to_dict(), expanded=False)

with review_tab:
    review_columns = st.columns([1, 3])
    review_enable_llm = review_columns[1].checkbox(
        "启用模型（关闭则保守保留疑点转人工）", value=True, key="m6-llm"
    )
    if review_columns[0].button(
        "运行综合审查（M6）", type="primary", use_container_width=True
    ):
        try:
            with st.spinner(
                "统一图：理解 → 要素 + 双路审查（并行）→ 合并 → 法条检索 → "
                "裁判 → 仲裁 → 融合，通常需要 30~90 秒……"
            ):
                report = run_unified_review(
                    build_contract(result),
                    AgentReviewConfig(
                        endpoint=llm_endpoint,
                        model=llm_model,
                        api_key=llm_api_key,
                        timeout_seconds=int(llm_timeout),
                        enable_llm=review_enable_llm,
                    ),
                )
            st.session_state["m6-result"] = (document_hash, report)
        except Exception as exc:  # Keep the page usable when anything is down.
            st.error(f"综合审查失败：{exc}")

    cached = st.session_state.get("m6-result")
    report = cached[1] if cached and cached[0] == document_hash else None
    if report is None:
        st.caption(
            "点击按钮运行 M6 统一审查图（LangGraph 编排：要素抽取 + 逐条/跨条款审查 + "
            "法条检索与适用 + LLM 裁判 + 冲突仲裁 + 统一融合），输出整体合理性结论、"
            "逐条风险（带法条引用与交叉印证标记）、仲裁记录与人工复核队列。"
        )
    else:
        overall = report.get("overall", {})
        overall_label = {
            "low_risk": "低风险",
            "medium_risk": "中风险",
            "high_risk": "高风险",
            "unknown": "未判定",
        }.get(overall.get("overall_assessment"), "—")
        stats = report.get("stats", {})
        loops = stats.get("loops", {})
        metric_columns = st.columns(4)
        metric_columns[0].metric("整体评估", overall_label)
        metric_columns[1].metric(
            "合理性评分",
            overall.get("overall_score")
            if overall.get("overall_score") is not None
            else "—",
        )
        metric_columns[2].metric("风险数", stats.get("final_risks", 0))
        metric_columns[3].metric("待人工复核", stats.get("review_required", 0))
        if overall.get("summary"):
            st.info(overall["summary"])
        elements = report.get("elements", {})
        st.caption(
            "要素概览："
            f"当事人 {len(elements.get('parties', []))} · "
            f"金额 {len(elements.get('amounts', []))} · "
            f"日期 {len(elements.get('dates', []))} · "
            f"标的 {len(elements.get('subjects', []))}"
            f" ｜ 交叉印证 {stats.get('cross_validated', 0)} 条"
            f" ｜ 回环：检索 {loops.get('legal_retries', 0)} / 裁判 {loops.get('judge_retries', 0)}"
            f" / 证据 {loops.get('evidence_rounds', 0)} / 仲裁 {loops.get('arbitration_rounds', 0)}"
        )
        risks = report.get("risks", [])
        if not risks:
            st.success("未发现风险。")
        else:
            st.dataframe(
                [
                    {
                        "级别": risk["severity"],
                        "类别": risk.get("category_label", ""),
                        "风险类型": risk["code"],
                        "说明": risk["message"],
                        "法条依据": "；".join(
                            f"《{c['law_name']}》{c['article_no']}"
                            for c in risk.get("citations", [])
                        )
                        or "—",
                        "交叉印证": "是" if risk.get("cross_validated") else "否",
                        "待复核": "是" if risk["review_required"] else "否",
                    }
                    for risk in risks
                ],
                width="stretch",
                hide_index=True,
            )
            clause_index = {c.clause_id: c for c in result.clauses}
            with st.expander("风险详情与裁判理由"):
                for risk in risks:
                    st.markdown(
                        f"**{risk['risk_id']} · {risk['code']} · {risk['severity']}**"
                        + ("（已交叉印证）" if risk.get("cross_validated") else "")
                        + ("（待人工复核）" if risk["review_required"] else "")
                    )
                    st.text(risk["message"])
                    for clause_id in risk.get("clause_ids", []):
                        clause = clause_index.get(clause_id)
                        if clause is None:
                            continue
                        head = " ".join(
                            part
                            for part in (clause.label, clause.title)
                            if part
                        )
                        st.markdown(f"合同原文 · {clause_id} {head}")
                        st.text(clause.text)
                    for citation in risk.get("citations", []):
                        st.text(
                            f"依据：《{citation['law_name']}》{citation['article_no']}"
                            f" {citation.get('quote', '')}"
                        )
                    if risk.get("rationale"):
                        st.caption(risk["rationale"])
        conflicts = report.get("conflicts", [])
        if conflicts:
            with st.expander(f"仲裁记录（{len(conflicts)}）"):
                for conflict in conflicts:
                    st.markdown(
                        f"- `{conflict['type']}` {conflict['description']}"
                        + (f"（意见：{conflict['resolution']}）" if conflict.get("resolution") else "")
                    )
        queue = report.get("review_queue", [])
        if queue:
            with st.expander(f"人工复核队列（{len(queue)}）", expanded=True):
                for item in queue:
                    st.markdown(
                        f"- **{item['risk_id']}** `{item['code']}`：{item['reason']}"
                    )
        if report.get("errors"):
            with st.expander("运行警告"):
                for error in report["errors"]:
                    st.warning(error)

payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
with json_tab:
    st.download_button(
        "下载解析结果",
        data=payload.encode("utf-8"),
        file_name=f"{uploaded_name}.parsed.json",
        mime="application/json",
    )
    st.code(payload, language="json")
