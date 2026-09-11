from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BoundaryDecision, Clause, SourceBlock, SourceSpan


CHINESE_NUMBER = "零〇一二三四五六七八九十百千万两"
SENTENCE_ENDINGS = "。；！？!?"
AUTO_ACCEPT_CONFIDENCE = 0.90
BODY_CUES = re.compile(
    r"(?:应当|应于|应在|不得|有权|负责|必须|须在|可以|同意|保证|承诺|支付|交付|承担)"
)


@dataclass(frozen=True, slots=True)
class HeadingMatch:
    label: str
    title: str
    level: int
    heading: str
    inline_body: str = ""
    kind: str = "numbered_text"
    confidence: float = 0.98


@dataclass(slots=True)
class ClauseSplitResult:
    blocks: list[SourceBlock]
    clauses: list[Clause]
    boundaries: list[BoundaryDecision]


HEADING_PATTERNS: tuple[tuple[re.Pattern[str], str, int], ...] = (
    (re.compile(rf"^(第[{CHINESE_NUMBER}0-9]+章)(?:[\s：:、.-]*)(.*)$"), "chapter", 1),
    (re.compile(rf"^(第[{CHINESE_NUMBER}0-9]+节)(?:[\s：:、.-]*)(.*)$"), "section", 2),
    (
        re.compile(
            rf"^(第[{CHINESE_NUMBER}0-9]+条(?:之[{CHINESE_NUMBER}0-9]+)?)(?:[\s：:、.-]*)(.*)$"
        ),
        "article",
        2,
    ),
    (re.compile(rf"^(第[{CHINESE_NUMBER}0-9]+款)(?:[\s：:、.-]*)(.*)$"), "paragraph", 3),
    (re.compile(r"^(\d+(?:\.\d+){1,5})(?:[\s：:、.-]+)(.*)$"), "decimal", 0),
    # (?!\d) keeps decimal values such as "141.78平方米" out of the list
    # pattern: PDF tables often place a bare figure on its own line.
    (re.compile(r"^(\d{1,3}[.、])(?!\d)(?:\s*)(.*)$"), "arabic_list", 2),
    (re.compile(rf"^([{CHINESE_NUMBER}]+、)(?:\s*)(.*)$"), "chinese_list", 2),
    (re.compile(rf"^([（(][{CHINESE_NUMBER}]+[）)])(?:\s*)(.*)$"), "chinese_item", 3),
    (re.compile(r"^([（(]\d{1,3}[）)])(?:\s*)(.*)$"), "arabic_item", 4),
    (re.compile(r"^((?:附件|附录)\s*[一二三四五六七八九十0-9]*)[：:\s、.-]*(.*)$"), "appendix", 1),
)


def _classify_remainder(remainder: str) -> tuple[str, str, bool]:
    """Return title, inline body and whether the choice is inherently ambiguous."""

    remainder = remainder.strip()
    if not remainder:
        return "", "", False
    punctuation_count = sum(remainder.count(mark) for mark in "，,；;。")
    looks_like_body = (
        remainder.endswith(tuple(SENTENCE_ENDINGS))
        or len(remainder) > 36
        or punctuation_count >= 2
        or (len(remainder) >= 12 and BODY_CUES.search(remainder) is not None)
    )
    if looks_like_body:
        return "", remainder, 8 <= len(remainder) <= 40
    return remainder, "", 12 <= len(remainder) <= 36


def _decimal_level(label: str) -> int:
    return min(6, len(label.split(".")) + 1)


def detect_heading(text: str) -> HeadingMatch | None:
    first_line = text.splitlines()[0].strip()
    if len(first_line) > 180:
        return None
    for pattern, kind, fixed_level in HEADING_PATTERNS:
        match = pattern.match(first_line)
        if not match:
            continue
        label = match.group(1).strip()
        remainder = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
        title, inline_body, ambiguous = _classify_remainder(remainder)
        level = _decimal_level(label) if kind == "decimal" else fixed_level
        heading = label if inline_body else first_line
        return HeadingMatch(
            label=label,
            title=title,
            level=level,
            heading=heading,
            inline_body=inline_body,
            kind=kind,
            confidence=0.86 if ambiguous else 0.98,
        )
    return None


def detect_block_heading(block: SourceBlock) -> HeadingMatch | None:
    textual = detect_heading(block.text)
    if textual:
        return textual

    numbering_label = str(block.metadata.get("numbering_label", "")).strip()
    if numbering_label:
        title, inline_body, ambiguous = _classify_remainder(block.text)
        list_level = int(block.metadata.get("list_level", 0))
        return HeadingMatch(
            label=numbering_label,
            title=title,
            level=min(6, list_level + 2),
            heading=numbering_label + (f" {title}" if title else ""),
            inline_body=inline_body,
            kind="word_numbering",
            confidence=0.84 if ambiguous else 0.96,
        )

    style_name = str(block.metadata.get("style_name", ""))
    outline_level = block.metadata.get("outline_level")
    style_is_heading = style_name.lower().startswith("heading") or "标题" in style_name
    if outline_level is not None or style_is_heading:
        if len(block.text) <= 100:
            level = int(outline_level) + 1 if outline_level is not None else 2
            return HeadingMatch(
                label="",
                title=block.text,
                level=min(6, max(1, level)),
                heading=block.text,
                kind="word_heading_style",
                confidence=0.94,
            )
    return None


def prepare_blocks(blocks: list[SourceBlock]) -> list[SourceBlock]:
    """Normalize block granularity while keeping every generated ID traceable."""

    prepared: list[SourceBlock] = []
    for block in blocks:
        lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        raw_lines = [line for line in (block.raw_text or block.text).splitlines() if line.strip()]
        heading_count = sum(detect_heading(line) is not None for line in lines)
        source_block_id = str(block.metadata.get("source_block_id", block.block_id))

        if len(lines) <= 1 or heading_count == 0 or block.kind == "table":
            metadata = {
                **block.metadata,
                "source_block_id": source_block_id,
                "source_line_start": int(block.metadata.get("source_line_start", 1)),
                "source_line_end": int(block.metadata.get("source_line_end", max(1, len(lines)))),
            }
            prepared.append(
                SourceBlock(
                    block_id=block.block_id,
                    order=len(prepared) + 1,
                    kind=block.kind,
                    text=block.text,
                    page=block.page,
                    metadata=metadata,
                    raw_text=block.raw_text,
                )
            )
            continue

        for line_index, line in enumerate(lines, start=1):
            raw_line = raw_lines[line_index - 1] if line_index <= len(raw_lines) else line
            prepared.append(
                SourceBlock(
                    block_id=f"{block.block_id}-L{line_index:03d}",
                    order=len(prepared) + 1,
                    kind=block.kind,
                    text=line,
                    page=block.page,
                    metadata={
                        **block.metadata,
                        "source_block_id": source_block_id,
                        "source_line_start": line_index,
                        "source_line_end": line_index,
                    },
                    raw_text=raw_line,
                )
            )
    return prepared


def _make_boundaries(
    blocks: list[SourceBlock], headings: list[HeadingMatch | None]
) -> list[BoundaryDecision]:
    boundaries: list[BoundaryDecision] = []
    has_structural_anchor = any(heading is not None for heading in headings)
    for index in range(1, len(blocks)):
        left = blocks[index - 1]
        right = blocks[index]
        heading = headings[index]
        evidence: list[str] = []
        warnings: list[str] = []

        if heading:
            relation = "NEW_CLAUSE" if heading.level <= 2 else "NEW_SUBCLAUSE"
            confidence = heading.confidence
            source = heading.kind
            evidence.append(f"右侧识别为{heading.label or '无编号标题'}，层级{heading.level}")
            if heading.confidence < 0.9:
                warnings.append("编号后的内容可能是标题，也可能是同一行正文")
        elif (
            left.metadata.get("source_block_id") == right.metadata.get("source_block_id")
            and left.block_id != right.block_id
        ):
            relation = "CONTINUATION"
            confidence = 0.90
            source = "layout_rule"
            evidence.append("两个文本块来自同一PDF原始区块的相邻行")
        else:
            relation = "SAME_CLAUSE"
            confidence = 0.76 if has_structural_anchor else 0.42
            source = "default_rule"
            evidence.append("未发现新的编号或标题结构")
            if right.kind == "table":
                confidence = min(confidence, 0.68)
                warnings.append("表格归属需要结合上下文确认")
            if not has_structural_anchor:
                warnings.append("全文缺少结构锚点，需要语义模型判断边界")

        boundaries.append(
            BoundaryDecision(
                boundary_id=f"D{index:04d}",
                left_block_id=left.block_id,
                right_block_id=right.block_id,
                relation=relation,
                confidence=round(confidence, 2),
                decision_source=source,
                evidence=evidence,
                review_required=confidence < AUTO_ACCEPT_CONFIDENCE,
                warnings=warnings,
                rule_relation=relation,
                rule_confidence=round(confidence, 2),
            )
        )
    return boundaries


def source_span_of(block: SourceBlock) -> SourceSpan:
    return SourceSpan(
        block_id=block.block_id,
        source_block_id=str(block.metadata.get("source_block_id", block.block_id)),
        page=block.page,
        line_start=int(block.metadata["source_line_start"])
        if block.metadata.get("source_line_start") is not None
        else None,
        line_end=int(block.metadata["source_line_end"])
        if block.metadata.get("source_line_end") is not None
        else None,
    )


def _assemble_clauses(
    blocks: list[SourceBlock],
    headings: list[HeadingMatch | None],
    boundaries: list[BoundaryDecision] | None = None,
) -> list[Clause]:
    groups: list[tuple[HeadingMatch | None, list[SourceBlock]]] = []
    current_level = 0
    for block_index, (block, heading) in enumerate(zip(blocks, headings, strict=True)):
        boundary = boundaries[block_index - 1] if boundaries and block_index > 0 else None
        should_start = (
            block_index == 0
            or (
                boundary is not None
                and boundary.relation in {"NEW_CLAUSE", "NEW_SUBCLAUSE"}
            )
            or (boundaries is None and heading is not None and block.kind != "table")
        )
        if should_start:
            selected_heading = heading
            if selected_heading is None and boundary is not None:
                is_title = boundary.right_block_role == "TITLE"
                inferred_title = (boundary.inferred_title or "").strip()
                if is_title and not inferred_title:
                    inferred_title = block.text
                level = (
                    2
                    if boundary.relation == "NEW_CLAUSE"
                    else min(6, max(3, current_level + 1))
                )
                selected_heading = HeadingMatch(
                    label="未编号",
                    title=inferred_title,
                    level=level,
                    heading=block.text if is_title else "",
                    inline_body="" if is_title else block.text,
                    kind="llm_inferred",
                    confidence=boundary.confidence,
                )
            groups.append((selected_heading, [block]))
            current_level = selected_heading.level if selected_heading else 0
        elif groups:
            groups[-1][1].append(block)
        else:
            groups.append((None, [block]))

    clauses: list[Clause] = []
    parent_stack: list[tuple[int, str]] = []
    for index, (heading, clause_blocks) in enumerate(groups, start=1):
        clause_id = f"C{index:04d}"
        level = heading.level if heading else 0
        label = heading.label if heading and heading.label else ("标题" if heading else "前言")
        title = heading.title if heading else "合同前言"
        heading_text = heading.heading if heading else ""

        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()
        parent_id = parent_stack[-1][1] if parent_stack and level > 0 else None

        raw_texts = [block.text for block in clause_blocks]
        full_text = "\n".join(raw_texts)
        body_parts = raw_texts.copy()
        if heading and body_parts:
            first_lines = body_parts[0].splitlines()
            body_parts = (
                ([heading.inline_body] if heading.inline_body else [])
                + first_lines[1:]
                + body_parts[1:]
            )
        body = "\n".join(part for part in body_parts if part).strip()
        pages = [block.page for block in clause_blocks if block.page is not None]

        clauses.append(
            Clause(
                clause_id=clause_id,
                label=label,
                title=title,
                level=level,
                heading=heading_text,
                body=body,
                text=full_text,
                parent_id=parent_id,
                block_ids=[block.block_id for block in clause_blocks],
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                source_spans=[source_span_of(block) for block in clause_blocks],
            )
        )
        if level > 0:
            parent_stack.append((level, clause_id))
    return clauses


def analyze_clause_structure(blocks: list[SourceBlock]) -> ClauseSplitResult:
    prepared = prepare_blocks(blocks)
    headings = [detect_block_heading(block) if block.kind != "table" else None for block in prepared]
    boundaries = _make_boundaries(prepared, headings)
    return ClauseSplitResult(
        blocks=prepared,
        clauses=_assemble_clauses(prepared, headings, boundaries),
        boundaries=boundaries,
    )


def rebuild_clauses_from_boundaries(
    blocks: list[SourceBlock], boundaries: list[BoundaryDecision]
) -> list[Clause]:
    """Rebuild the clause tree after a semantic reviewer updates boundaries."""

    headings = [detect_block_heading(block) if block.kind != "table" else None for block in blocks]
    return _assemble_clauses(blocks, headings, boundaries)


def split_clauses(blocks: list[SourceBlock]) -> list[Clause]:
    """Backward-compatible clause-only API used by callers and unit tests."""

    return analyze_clause_structure(blocks).clauses
