from __future__ import annotations

import io
import json
import unittest
import zipfile

from contract_review.service import parse_contract
from contract_review.splitter import (
    analyze_clause_structure,
    detect_heading,
    split_clauses,
)
from contract_review.models import SourceBlock


def minimal_docx(paragraphs: list[str]) -> bytes:
    body = "".join(
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def numbered_docx(paragraphs: list[tuple[str, bool]]) -> bytes:
    body_parts: list[str] = []
    for text, numbered in paragraphs:
        num_pr = (
            "<w:pPr><w:numPr><w:ilvl w:val=\"0\"/>"
            "<w:numId w:val=\"1\"/></w:numPr></w:pPr>"
            if numbered
            else ""
        )
        body_parts.append(f"<w:p>{num_pr}<w:r><w:t>{text}</w:t></w:r></w:p>")
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body_parts)}<w:sectPr/></w:body></w:document>"
    )
    numbering_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0">'
        '<w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
        '</w:lvl></w:abstractNum><w:num w:numId="1">'
        '<w:abstractNumId w:val="0"/></w:num></w:numbering>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/numbering.xml", numbering_xml)
    return buffer.getvalue()


class HeadingChecks(unittest.TestCase):
    def test_detects_common_contract_headings(self) -> None:
        self.assertEqual(detect_heading("第一条 合同目的").label, "第一条")
        self.assertEqual(detect_heading("1.1 服务期限").level, 3)
        self.assertEqual(detect_heading("（一）付款方式").level, 3)
        self.assertIsNone(detect_heading("双方本着平等原则订立本合同。"))

    def test_builds_parent_relationships(self) -> None:
        blocks = [
            SourceBlock("B0001", 1, "paragraph", "第一章 总则"),
            SourceBlock("B0002", 2, "paragraph", "第一条 合同目的"),
            SourceBlock("B0003", 3, "paragraph", "双方约定如下。"),
            SourceBlock("B0004", 4, "paragraph", "（一）服务范围"),
            SourceBlock("B0005", 5, "paragraph", "提供技术服务。"),
        ]
        clauses = split_clauses(blocks)
        self.assertEqual(len(clauses), 3)
        self.assertEqual(clauses[1].parent_id, clauses[0].clause_id)
        self.assertEqual(clauses[2].parent_id, clauses[1].clause_id)
        self.assertIn("双方约定如下", clauses[1].body)

    def test_keeps_inline_sentence_as_clause_body(self) -> None:
        blocks = [
            SourceBlock("B0001", 1, "paragraph", "第一条 甲方应当在十日内支付全部款项。")
        ]
        clause = split_clauses(blocks)[0]
        self.assertEqual(clause.title, "")
        self.assertEqual(clause.body, "甲方应当在十日内支付全部款项。")

    def test_decimal_numbering_builds_dynamic_hierarchy(self) -> None:
        blocks = [
            SourceBlock("B0001", 1, "paragraph", "1. 项目"),
            SourceBlock("B0002", 2, "paragraph", "1.1 服务"),
            SourceBlock("B0003", 3, "paragraph", "1.1.1 服务范围"),
        ]
        clauses = split_clauses(blocks)
        self.assertEqual([clause.level for clause in clauses], [2, 3, 4])
        self.assertEqual(clauses[1].parent_id, clauses[0].clause_id)
        self.assertEqual(clauses[2].parent_id, clauses[1].clause_id)

    def test_multiline_expansion_remains_traceable(self) -> None:
        result = analyze_clause_structure(
            [
                SourceBlock(
                    "B0001",
                    1,
                    "text_block",
                    "第一条 标题\n正文内容\n第二条 标题二",
                    page=1,
                )
            ]
        )
        available_ids = {block.block_id for block in result.blocks}
        referenced_ids = {
            block_id for clause in result.clauses for block_id in clause.block_ids
        }
        self.assertEqual(available_ids, referenced_ids)
        self.assertEqual(result.blocks[1].metadata["source_block_id"], "B0001")
        self.assertEqual(len(result.boundaries), len(result.blocks) - 1)


class ParserChecks(unittest.TestCase):
    def test_txt_contract(self) -> None:
        content = "合同名称\n第一条 服务内容\n乙方提供服务。\n第二条 合同价款\n价款为100元。".encode()
        result = parse_contract("sample.txt", content)
        self.assertEqual(result.file_type, "txt")
        self.assertEqual(len(result.clauses), 3)
        self.assertEqual(result.clauses[0].label, "前言")
        self.assertEqual(result.clauses[1].label, "第一条")

    def test_docx_contract(self) -> None:
        content = minimal_docx(["示例合同", "第一条 服务内容", "乙方提供服务。"])
        result = parse_contract("sample.docx", content)
        self.assertEqual(result.file_type, "docx")
        self.assertEqual(len(result.blocks), 3)
        self.assertEqual(len(result.clauses), 2)
        self.assertIn("乙方提供服务", result.clauses[1].body)

    def test_docx_automatic_numbering(self) -> None:
        content = numbered_docx(
            [("服务内容", True), ("乙方提供技术服务。", False), ("付款方式", True)]
        )
        result = parse_contract("numbered.docx", content)
        self.assertEqual([clause.label for clause in result.clauses], ["1.", "2."])
        self.assertEqual(result.blocks[0].metadata["numbering_label"], "1.")
        self.assertEqual(result.blocks[2].metadata["numbering_label"], "2.")
        self.assertIn("乙方提供技术服务", result.clauses[0].body)

    def test_result_exposes_boundaries_and_raw_text(self) -> None:
        content = "  第一条  服务内容  \n乙方提供服务。".encode()
        result = parse_contract("sample.txt", content)
        self.assertEqual(len(result.boundaries), len(result.blocks) - 1)
        self.assertNotEqual(result.raw_text, result.full_text)
        self.assertTrue(all(block.raw_text is not None for block in result.blocks))
        self.assertTrue(result.boundaries[0].review_required)
        json.dumps(result.to_dict(), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
