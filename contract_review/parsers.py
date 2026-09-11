from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .models import SourceBlock


SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf"}
MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_SIZE = 100 * 1024 * 1024
MAX_PDF_PAGES = 1_000
MAX_EXTRACTED_CHARACTERS = 5_000_000

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{WORD_NS}}}"


class ParseError(ValueError):
    """Raised when a document cannot be safely parsed."""


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def validate_upload(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ParseError(f"不支持的文件类型。当前支持：{allowed}")
    if not content:
        raise ParseError("上传的文件为空。")
    if len(content) > MAX_FILE_SIZE:
        raise ParseError("文件超过25 MB，请压缩或拆分后重试。")
    return suffix


def parse_txt(content: bytes) -> tuple[list[SourceBlock], list[str]]:
    decoded: str | None = None
    encoding_used = ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            decoded = content.decode(encoding)
            encoding_used = encoding
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ParseError("无法识别文本编码，请将文件转换为UTF-8后重试。")

    blocks: list[SourceBlock] = []
    for index, paragraph in enumerate(re.split(r"\n\s*\n|\r?\n", decoded), start=1):
        text = clean_text(paragraph)
        if text:
            blocks.append(
                SourceBlock(
                    block_id=f"B{len(blocks) + 1:04d}",
                    order=len(blocks) + 1,
                    kind="paragraph",
                    text=text,
                    raw_text=paragraph,
                    metadata={"encoding": encoding_used, "source_line": index},
                )
            )
    return blocks, []


def _element_raw_text(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag in {f"{W}br", f"{W}cr"}:
            parts.append("\n")
    return "".join(parts)


def _element_text(element: ET.Element) -> str:
    return clean_text(_element_raw_text(element))


def _table_text(table: ET.Element) -> tuple[str, str, int]:
    rows: list[str] = []
    raw_rows: list[str] = []
    for row in table.findall(f"{W}tr"):
        row_cells = row.findall(f"{W}tc")
        cells = [_element_text(cell) for cell in row_cells]
        raw_cells = [_element_raw_text(cell) for cell in row_cells]
        if any(cells):
            rows.append(" | ".join(cells))
            raw_rows.append("\t".join(raw_cells))
    return "\n".join(rows), "\n".join(raw_rows), len(rows)


def _word_value(element: ET.Element | None, attribute: str = "val") -> str | None:
    if element is None:
        return None
    return element.get(f"{W}{attribute}")


def _parse_styles(xml: bytes | None) -> dict[str, dict[str, Any]]:
    if not xml:
        return {}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}
    styles: dict[str, dict[str, Any]] = {}
    for style in root.findall(f"{W}style"):
        style_id = style.get(f"{W}styleId")
        if not style_id:
            continue
        ppr = style.find(f"{W}pPr")
        numpr = ppr.find(f"{W}numPr") if ppr is not None else None
        styles[style_id] = {
            "name": _word_value(style.find(f"{W}name")) or style_id,
            "outline_level": _word_value(
                ppr.find(f"{W}outlineLvl") if ppr is not None else None
            ),
            "num_id": _word_value(
                numpr.find(f"{W}numId") if numpr is not None else None
            ),
            "list_level": _word_value(
                numpr.find(f"{W}ilvl") if numpr is not None else None
            ),
        }
    return styles


def _parse_numbering(xml: bytes | None) -> dict[int, dict[int, dict[str, Any]]]:
    if not xml:
        return {}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}

    abstract: dict[int, dict[int, dict[str, Any]]] = {}
    for abstract_num in root.findall(f"{W}abstractNum"):
        abstract_id = int(abstract_num.get(f"{W}abstractNumId", "0"))
        levels: dict[int, dict[str, Any]] = {}
        for level in abstract_num.findall(f"{W}lvl"):
            ilvl = int(level.get(f"{W}ilvl", "0"))
            start = _word_value(level.find(f"{W}start")) or "1"
            levels[ilvl] = {
                "start": int(start),
                "format": _word_value(level.find(f"{W}numFmt")) or "decimal",
                "template": _word_value(level.find(f"{W}lvlText")) or f"%{ilvl + 1}.",
            }
        abstract[abstract_id] = levels

    numbering: dict[int, dict[int, dict[str, Any]]] = {}
    for num in root.findall(f"{W}num"):
        num_id = int(num.get(f"{W}numId", "0"))
        abstract_id_text = _word_value(num.find(f"{W}abstractNumId"))
        if abstract_id_text is not None:
            numbering[num_id] = abstract.get(int(abstract_id_text), {})
    return numbering


def _chinese_number(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    if value < 20:
        return "十" + (digits[value % 10] if value % 10 else "")
    if value < 100:
        return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    return str(value)


class _NumberingState:
    def __init__(self, definitions: dict[int, dict[int, dict[str, Any]]]) -> None:
        self.definitions = definitions
        self.counters: dict[int, dict[int, int]] = {}

    def next_label(self, num_id: int, ilvl: int) -> tuple[str | None, str | None]:
        definition = self.definitions.get(num_id, {}).get(ilvl)
        if not definition or definition["format"] == "bullet":
            return None, definition["format"] if definition else None

        counters = self.counters.setdefault(num_id, {})
        start = int(definition.get("start", 1))
        counters[ilvl] = counters.get(ilvl, start - 1) + 1
        for deeper_level in [level for level in counters if level > ilvl]:
            counters.pop(deeper_level)

        label = str(definition["template"])
        for level in range(ilvl + 1):
            level_definition = self.definitions.get(num_id, {}).get(level, {})
            value = counters.get(level, int(level_definition.get("start", 1)))
            number_format = str(level_definition.get("format", "decimal"))
            rendered = (
                _chinese_number(value)
                if number_format in {"chineseCounting", "chineseLegalSimplified"}
                else str(value)
            )
            label = label.replace(f"%{level + 1}", rendered)
        return label, str(definition["format"])


def _paragraph_metadata(
    paragraph: ET.Element,
    styles: dict[str, dict[str, Any]],
    numbering_state: _NumberingState,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    ppr = paragraph.find(f"{W}pPr")
    style_id = _word_value(ppr.find(f"{W}pStyle") if ppr is not None else None)
    style = styles.get(style_id or "", {})
    if style_id:
        metadata["style_id"] = style_id
        metadata["style_name"] = style.get("name", style_id)
    if style.get("outline_level") is not None:
        metadata["outline_level"] = int(style["outline_level"])

    numpr = ppr.find(f"{W}numPr") if ppr is not None else None
    num_id_text = _word_value(numpr.find(f"{W}numId") if numpr is not None else None)
    ilvl_text = _word_value(numpr.find(f"{W}ilvl") if numpr is not None else None)
    num_id_text = num_id_text if num_id_text is not None else style.get("num_id")
    ilvl_text = ilvl_text if ilvl_text is not None else style.get("list_level")
    if num_id_text is not None:
        num_id = int(num_id_text)
        ilvl = int(ilvl_text or 0)
        label, number_format = numbering_state.next_label(num_id, ilvl)
        metadata.update({"num_id": num_id, "list_level": ilvl})
        if label:
            metadata["numbering_label"] = label
        if number_format:
            metadata["numbering_format"] = number_format

    if ppr is not None:
        alignment = _word_value(ppr.find(f"{W}jc"))
        if alignment:
            metadata["alignment"] = alignment
        indent = ppr.find(f"{W}ind")
        if indent is not None:
            for name in ("left", "right", "firstLine", "hanging"):
                value = indent.get(f"{W}{name}")
                if value is not None:
                    metadata[f"indent_{name}"] = int(value)

    text_runs = [
        run for run in paragraph.iter(f"{W}r") if _element_text(run)
    ]
    if text_runs:
        bold_values = [run.find(f"{W}rPr/{W}b") is not None for run in text_runs]
        metadata["bold"] = all(bold_values)
        sizes = [
            int(size)
            for run in text_runs
            if (size := _word_value(run.find(f"{W}rPr/{W}sz"))) is not None
        ]
        if sizes:
            metadata["font_size_half_points"] = max(sizes)
    return metadata


def parse_docx(content: bytes) -> tuple[list[SourceBlock], list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_DOCX_UNCOMPRESSED_SIZE:
                raise ParseError("DOCX解压后内容过大，已停止解析。")
            xml = archive.read("word/document.xml")
            numbering_xml = (
                archive.read("word/numbering.xml")
                if "word/numbering.xml" in archive.namelist()
                else None
            )
            styles_xml = (
                archive.read("word/styles.xml")
                if "word/styles.xml" in archive.namelist()
                else None
            )
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ParseError("DOCX文件损坏，或不是有效的Word文档。") from exc

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ParseError("无法读取DOCX正文结构。") from exc

    body = root.find(f"{W}body")
    if body is None:
        raise ParseError("DOCX中未找到正文。")

    styles = _parse_styles(styles_xml)
    numbering_state = _NumberingState(_parse_numbering(numbering_xml))
    blocks: list[SourceBlock] = []
    for child in body:
        kind = ""
        text = ""
        raw_text = ""
        metadata: dict[str, Any] = {}
        if child.tag == f"{W}p":
            kind = "paragraph"
            raw_text = _element_raw_text(child)
            text = clean_text(raw_text)
            metadata = _paragraph_metadata(child, styles, numbering_state)
        elif child.tag == f"{W}tbl":
            kind = "table"
            text, raw_text, rows = _table_text(child)
            metadata["rows"] = rows
        if not text:
            continue
        order = len(blocks) + 1
        blocks.append(
            SourceBlock(
                block_id=f"B{order:04d}",
                order=order,
                kind=kind,
                text=text,
                raw_text=raw_text,
                metadata=metadata,
            )
        )

    warnings = ["DOCX格式通常不包含稳定页码，结果以段落和表格顺序定位。"]
    return blocks, warnings


def _load_pymupdf():
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore

            return pymupdf
        except ImportError as exc:
            raise ParseError("解析PDF需要安装PyMuPDF：pip install PyMuPDF") from exc


def parse_pdf(content: bytes) -> tuple[list[SourceBlock], list[str]]:
    pymupdf = _load_pymupdf()
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ParseError("PDF文件损坏、加密或无法打开。") from exc

    blocks: list[SourceBlock] = []
    warnings: list[str] = []
    empty_pages: list[int] = []
    try:
        if document.page_count > MAX_PDF_PAGES:
            raise ParseError(f"PDF页数超过{MAX_PDF_PAGES}页，已停止解析。")
        extracted_characters = 0
        for page_number, page in enumerate(document, start=1):
            raw_blocks = page.get_text("blocks", sort=True)
            page_has_text = False
            for raw in raw_blocks:
                raw_text = str(raw[4]).strip("\r\n")
                text = clean_text(raw_text)
                if not text:
                    continue
                extracted_characters += len(text)
                if extracted_characters > MAX_EXTRACTED_CHARACTERS:
                    raise ParseError("PDF提取文本超过500万字符，已停止解析。")
                page_has_text = True
                order = len(blocks) + 1
                blocks.append(
                    SourceBlock(
                        block_id=f"B{order:04d}",
                        order=order,
                        kind="text_block",
                        text=text,
                        raw_text=raw_text,
                        page=page_number,
                        metadata={
                            "bbox": [round(float(value), 2) for value in raw[:4]],
                            "block_no": int(raw[5]) if len(raw) > 5 else None,
                        },
                    )
                )
            if not page_has_text:
                empty_pages.append(page_number)
    finally:
        document.close()

    if empty_pages:
        preview = "、".join(str(page) for page in empty_pages[:8])
        suffix = "等" if len(empty_pages) > 8 else ""
        warnings.append(f"第{preview}页{suffix}未提取到文本，可能是扫描页，需要OCR。")
    return blocks, warnings


def parse_blocks(filename: str, content: bytes) -> tuple[str, list[SourceBlock], list[str]]:
    suffix = validate_upload(filename, content)
    if suffix == ".txt":
        blocks, warnings = parse_txt(content)
    elif suffix == ".docx":
        blocks, warnings = parse_docx(content)
    else:
        blocks, warnings = parse_pdf(content)

    if not blocks:
        raise ParseError("文档中没有提取到可用文本；如果是扫描件，请先进行OCR。")
    return suffix.removeprefix("."), blocks, warnings
