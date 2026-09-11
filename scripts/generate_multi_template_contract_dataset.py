from __future__ import annotations

import html
import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


SEED = 20260912
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "generated_contracts" / "samr_multi_template_2025"


TEMPLATES = [
    {
        "key": "housing_rent",
        "title": "城镇房屋租赁合同",
        "template_id": "GF-2025-2614",
        "source_url": "https://htsfwb.samr.gov.cn/View?id=2340996b-882d-47a4-b74d-c30784628737",
        "publisher": "国家市场监督管理总局",
    },
    {
        "key": "entrust",
        "title": "委托合同",
        "template_id": "GF-2025-1001",
        "source_url": "https://htsfwb.samr.gov.cn/View?id=50b57729-0fca-45d2-92c3-fe7e6a989815",
        "publisher": "国家市场监督管理总局",
    },
    {
        "key": "intermediary",
        "title": "中介合同",
        "template_id": "GF-2025-2613",
        "source_url": "https://htsfwb.samr.gov.cn/View?id=b6cdd9b5-4d70-4fd1-8de2-b39a53229b47",
        "publisher": "国家市场监督管理总局",
    },
    {
        "key": "data_provide",
        "title": "数据提供合同",
        "template_id": "GF-2025-2615",
        "source_url": "https://htsfwb.samr.gov.cn/View?id=0912bdd8-c6f0-4c4b-bff1-cb9eb5762f2c",
        "publisher": "国家数据局、市场监督管理总局",
    },
    {
        "key": "custody",
        "title": "保管合同",
        "template_id": "GF-2025-0801",
        "source_url": "https://htsfwb.samr.gov.cn/View?id=dbf08fdc-ec9a-4f4a-9fa7-2a05dd4bbf99",
        "publisher": "国家市场监督管理总局",
    },
]


RISK_TYPES = [
    ("clean", "无预设矛盾，用于正向结构样本"),
    ("amount_conflict", "付款金额与合同摘要中的金额不一致"),
    ("date_conflict", "期限或履行日期在不同条款中不一致"),
    ("party_conflict", "签署区主体名称与当事人信息不一致"),
    ("reference_conflict", "正文引用了不存在或错误的附件编号"),
]


@dataclass(slots=True)
class ContractData:
    contract_id: str
    template_key: str
    template_title: str
    template_id: str
    source_url: str
    party_a: str
    party_b: str
    city: str
    subject: str
    amount: int
    secondary_amount: int
    start_date: str
    end_date: str
    summary_end_date: str
    risk_type: str
    risk_detail: str


def fmt_date(value: date) -> str:
    return f"{value.year}年{value.month}月{value.day}日"


def money_upper(value: int) -> str:
    digits = "零壹贰叁肆伍陆柒捌玖"
    units = ["", "拾", "佰", "仟", "万", "拾", "佰", "仟", "亿"]
    if value == 0:
        return "零元整"
    text, zero = "", False
    for pos, char in enumerate(reversed(str(value))):
        digit = int(char)
        if digit == 0:
            zero = bool(text)
            continue
        if zero and text:
            text = "零" + text
        zero = False
        text = digits[digit] + units[pos] + text
    return text + "元整"


def make_data(rng: random.Random, index: int, spec: dict[str, str]) -> ContractData:
    cities = ["北京市", "上海市", "广州市", "深圳市", "杭州市", "武汉市", "成都市", "西安市"]
    surnames = ["张", "李", "王", "赵", "陈", "刘", "杨", "黄", "周", "吴"]
    given = ["明", "芳", "伟", "静", "磊", "婷", "浩", "雪", "杰", "琳"]
    city = cities[(index - 1) % len(cities)]
    party_a = rng.choice(surnames) + rng.choice(given) + rng.choice(given)
    party_b = rng.choice([s for s in surnames if s != party_a[0]]) + rng.choice(given) + rng.choice(given)
    start = date(2026, 10, 1) + timedelta(days=index * 2)
    end = start + timedelta(days=rng.choice([90, 180, 365, 540]))
    risk_type, risk_detail = RISK_TYPES[(index - 1) % len(RISK_TYPES)]

    if spec["key"] == "housing_rent":
        subject = f"{city}滨江区云水路{18 + index}号{rng.randint(1, 12)}室住宅"
        amount = rng.choice([3200, 4500, 5800, 7200])
        secondary = amount * rng.choice([1, 2])
    elif spec["key"] == "entrust":
        subject = rng.choice(["市场调研与报告编制", "软件系统运维", "展会策划执行", "财税申报代理"])
        amount = rng.choice([28000, 45000, 68000, 92000])
        secondary = rng.choice([3000, 5000, 8000])
    elif spec["key"] == "intermediary":
        subject = rng.choice(["办公楼租赁撮合", "设备采购撮合", "技术服务项目撮合", "二手房交易撮合"])
        amount = rng.choice([180000, 260000, 420000, 680000])
        secondary = int(amount * rng.choice([0.02, 0.03, 0.05]))
    elif spec["key"] == "data_provide":
        subject = rng.choice(["城市交通运行数据集", "零售商品价格数据集", "工业设备传感数据集", "公共服务统计数据集"])
        amount = rng.choice([12000, 26000, 48000, 76000])
        secondary = rng.choice([100000, 250000, 500000, 1000000])
    else:
        subject = rng.choice(["精密仪器一批", "展览器材一批", "电子元器件一批", "艺术品一件"])
        amount = rng.choice([800, 1500, 2600, 4800])
        secondary = rng.randint(1, 20)

    if risk_type == "amount_conflict":
        secondary += rng.choice([1000, 2000, 5000])
    if risk_type == "date_conflict":
        end = end + timedelta(days=30)
    summary_end = fmt_date(end - timedelta(days=30)) if risk_type == "date_conflict" else fmt_date(end)

    return ContractData(
        contract_id=f"SYN-{spec['template_id'].replace('-', '')}-{index:04d}",
        template_key=spec["key"], template_title=spec["title"], template_id=spec["template_id"],
        source_url=spec["source_url"], party_a=party_a, party_b=party_b, city=city, subject=subject,
        amount=amount, secondary_amount=secondary, start_date=fmt_date(start), end_date=fmt_date(end),
        summary_end_date=summary_end, risk_type=risk_type, risk_detail=risk_detail,
    )


def clauses(data: ContractData) -> list[tuple[str, str]]:
    wrong_party = f"林{data.party_b[1:]}" if data.risk_type == "party_conflict" else data.party_b
    attachment = "附件九" if data.risk_type == "reference_conflict" else "附件一"
    amount_note = data.secondary_amount + 3000 if data.risk_type == "amount_conflict" else data.secondary_amount
    A, B = data.party_a, data.party_b
    k = data.template_key
    if k == "housing_rent":
        return [("heading", "第一条 房屋基本状况"), ("body", f"甲方（出租人）：{A}；乙方（承租人）：{B}。甲方将位于{data.subject}的房屋出租给乙方。"), ("heading", "第二条 租赁用途与期限"), ("body", f"房屋用于居住，租赁期限自{data.start_date}起至{data.end_date}止。"), ("heading", "第三条 租金与押金"), ("body", f"月租金为人民币{data.amount:.2f}元，押金为人民币{amount_note:.2f}元，乙方按月支付。"), ("heading", "第四条 房屋交割与维修"), ("body", "双方应共同填写房屋交割单，甲方负责主体结构维修，乙方承担正常使用产生的水电费用。"), ("heading", "第五条 转租与违约责任"), ("body", f"未经甲方书面同意不得转租；逾期支付租金超过十五日的，甲方有权解除合同。相关物品清单见{attachment}。"), ("heading", "第六条 争议解决"), ("body", f"争议协商不成的，向{data.city}有管辖权的人民法院起诉。"), ("body", f"甲方（签章）：{A}    乙方（签章）：{wrong_party}")]
    if k == "entrust":
        return [("heading", "第一条 委托事项"), ("body", f"甲方（委托人）：{A}委托乙方（受托人）：{B}，办理{data.subject}。"), ("heading", "第二条 委托权限"), ("body", "乙方应在授权范围内处理事务，不得擅自变更重要方案或将委托事项转委托给第三人。"), ("heading", "第三条 委托期限"), ("body", f"委托期限自{data.start_date}起至{data.end_date}止，乙方应在期限内提交成果。"), ("heading", "第四条 报酬与费用"), ("body", f"委托报酬为人民币{data.amount:.2f}元，预计必要费用为人民币{amount_note:.2f}元，凭有效凭证结算。"), ("heading", "第五条 报告与保密"), ("body", "乙方应定期报告办理进度，对因履行委托知悉的商业信息承担保密义务。"), ("heading", "第六条 解除与争议"), ("body", f"任一方严重违约，守约方可解除合同；争议协商不成的，向{data.city}有管辖权的人民法院起诉。"), ("body", f"甲方（签章）：{A}    乙方（签章）：{wrong_party}")]
    if k == "intermediary":
        return [("heading", "第一条 中介服务事项"), ("body", f"甲方（委托人）：{A}委托乙方（中介人）：{B}，就{data.subject}提供交易机会、信息核验和撮合服务。"), ("heading", "第二条 服务内容与完成标准"), ("body", "乙方应如实披露交易相对方信息，促成甲方与相对方签署主合同即视为中介成功。"), ("heading", "第三条 服务期限"), ("body", f"服务期限自{data.start_date}起至{data.end_date}止。"), ("heading", "第四条 中介报酬"), ("body", f"主合同签署后，甲方应向乙方支付中介报酬人民币{data.secondary_amount:.2f}元；主交易金额参考为人民币{data.amount:.2f}元。"), ("heading", "第五条 防止绕开中介"), ("body", "甲方不得绕开乙方直接与乙方提供的相对方交易，否则仍应支付约定报酬。"), ("heading", "第六条 违约责任"), ("body", f"因一方违约造成损失的，应依法赔偿；争议协商不成的，向{data.city}有管辖权的人民法院起诉。"), ("body", f"甲方（签章）：{A}    乙方（签章）：{wrong_party}")]
    if k == "data_provide":
        return [("heading", "第一条 数据标的"), ("body", f"甲方（接收方）：{A}，乙方（提供方）：{B}。乙方向甲方提供{data.subject}，预计记录数{data.secondary_amount}条。"), ("heading", "第二条 提供方式与期限"), ("body", f"数据通过API和加密文件两种方式交付，首次交付日为{data.start_date}，服务截止日为{data.end_date}。"), ("heading", "第三条 许可范围"), ("body", "甲方仅可为约定业务目的使用数据，不得向无关第三方转让、出租或公开发布。"), ("heading", "第四条 服务费用"), ("body", f"数据提供服务费为人民币{data.amount:.2f}元，安全审计及技术支持费用为人民币{amount_note:.2f}元。"), ("heading", "第五条 数据安全与个人信息"), ("body", "双方应采取访问控制、加密和脱敏措施；涉及个人信息时，应遵守个人信息保护和数据安全相关法律法规。"), ("heading", "第六条 违约与争议"), ("body", f"数据泄露或超范围使用造成损失的，违约方承担赔偿责任；技术交付清单见{attachment}。"), ("body", f"甲方（签章）：{A}    乙方（签章）：{wrong_party}")]
    return [("heading", "第一条 保管物与交付"), ("body", f"甲方（寄存人）：{A}将{data.subject}交由乙方（保管人）：{B}保管，保管数量为{data.secondary_amount}件。"), ("heading", "第二条 保管期限"), ("body", f"保管期限自{data.start_date}起至{data.end_date}止，乙方应妥善保管并按约返还。"), ("heading", "第三条 保管费用"), ("body", f"保管费为人民币{data.amount:.2f}元，入库、出库和特殊包装费用另行按清单结算。"), ("heading", "第四条 保管责任"), ("body", "乙方应建立出入库记录并采取防火、防潮和防盗措施；因保管不善造成损失的，应依法赔偿。"), ("heading", "第五条 取回与验收"), ("body", "甲方取回保管物时应核对数量、外观和规格，双方在保管单上签字确认。"), ("heading", "第六条 争议解决"), ("body", f"保管单及物品明细见{attachment}；争议协商不成的，向{data.city}有管辖权的人民法院起诉。"), ("body", f"甲方（签章）：{A}    乙方（签章）：{wrong_party}")]


def set_docx_defaults(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(2.2); section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5); section.right_margin = Cm(2.5)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"; normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体"); normal.font.size = Pt(10.5)


def summary_table_docx(doc: Document, data: ContractData) -> None:
    values = [("合同编号", data.contract_id, "模板编号", data.template_id), ("模板类型", data.template_title, "发布机关", "市场监管总局"), ("甲方", data.party_a, "乙方", data.party_b), ("标的", data.subject, "金额", f"{data.amount:.2f}元"), ("起止日期", f"{data.start_date}—{data.summary_end_date}", "风险标签", data.risk_type)]
    table = doc.add_table(rows=len(values), cols=4); table.style = "Table Grid"; table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row, vals in zip(table.rows, values):
        for cell, value in zip(row.cells, vals):
            cell.text = value; cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs: p.paragraph_format.space_after = Pt(0)
    for row in table.rows: row.cells[0].width = Cm(2.4); row.cells[2].width = Cm(2.4)


def create_docx(data: ContractData, path: Path) -> None:
    doc = Document(); set_docx_defaults(doc)
    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(data.template_title); run.bold = True; run.font.size = Pt(16)
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER; sub.add_run(f"基于国家市场监督管理总局合同示范文本结构生成的合成数据（{data.template_id}）").italic = True
    summary_table_docx(doc, data)
    note = doc.add_paragraph("说明：本文件仅用于合同解析、条款切分和风险检测实验，不具有法律效力，不代表官方合同文本。"); note.runs[0].font.size = Pt(8.5)
    for kind, text in clauses(data):
        p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(8 if kind == "heading" else 4)
        if kind == "heading": p.style = doc.styles["Heading 2"]; r = p.add_run(text); r.font.name = "黑体"; r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        else: p.add_run(text)
    footer = doc.sections[0].footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER; footer.add_run(f"{data.contract_id} | Synthetic dataset | No legal effect").font.size = Pt(8)
    doc.save(path)


def register_pdf_font() -> str:
    font_path = Path(r"C:\Windows\Fonts\simfang.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("SimFang", str(font_path))); return "SimFang"
    return "Helvetica"


class FooterCanvas(Canvas):
    def __init__(self, *args, contract_id: str, font_name: str, **kwargs):
        super().__init__(*args, **kwargs); self.contract_id = contract_id; self.font_name = font_name
    def showPage(self) -> None:
        self.saveState(); self.setFont(self.font_name, 7); self.setFillColor(colors.HexColor("#666666")); self.drawCentredString(A4[0] / 2, 1.0 * cm, f"{self.contract_id} | Synthetic dataset | No legal effect"); self.restoreState(); super().showPage()
    def save(self) -> None:
        if self._pageNumber == 0: self.showPage()
        super().save()


def create_pdf(data: ContractData, path: Path) -> None:
    font = register_pdf_font(); styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontName=font, fontSize=16, leading=22, alignment=TA_CENTER, spaceAfter=8)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], fontName=font, fontSize=8.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#666666"), spaceAfter=10)
    head_style = ParagraphStyle("HeadX", parent=styles["Heading2"], fontName=font, fontSize=11.5, leading=16, alignment=TA_LEFT, spaceBefore=8, spaceAfter=5)
    body_style = ParagraphStyle("BodyX", parent=styles["BodyText"], fontName=font, fontSize=9.2, leading=15, alignment=TA_LEFT, spaceAfter=4)
    note_style = ParagraphStyle("NoteX", parent=body_style, fontSize=7.8, leading=11, textColor=colors.HexColor("#666666"))
    cell_style = ParagraphStyle("CellX", parent=body_style, fontName=font, fontSize=8.1, leading=10, spaceAfter=0)
    story = [Paragraph(html.escape(data.template_title), title_style), Paragraph(html.escape(f"基于国家市场监督管理总局合同示范文本结构生成的合成数据（{data.template_id}）"), sub_style)]
    rows = [["合同编号", data.contract_id, "模板编号", data.template_id], ["模板类型", data.template_title, "发布机关", "市场监管总局"], ["甲方", data.party_a, "乙方", data.party_b], ["标的", data.subject, "金额", f"{data.amount:.2f}元"], ["起止日期", f"{data.start_date}—{data.summary_end_date}", "风险标签", data.risk_type]]
    table_data = [[Paragraph(html.escape(str(v)), cell_style) for v in row] for row in rows]
    table = Table(table_data, colWidths=[2.2 * cm, 5.1 * cm, 2.2 * cm, 5.1 * cm]); table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D9D9")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")), ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EEF3F8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [table, Spacer(1, 8), Paragraph("说明：本文件仅用于合同解析、条款切分和风险检测实验，不具有法律效力，不代表官方合同文本。", note_style), Spacer(1, 6)]
    for kind, text in clauses(data): story.append(Paragraph(html.escape(text), head_style if kind == "heading" else body_style))
    def factory(*args, **kwargs): return FooterCanvas(*args, contract_id=data.contract_id, font_name=font, **kwargs)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.8 * cm, bottomMargin=1.7 * cm, title=f"{data.contract_id} {data.template_title}", author="Synthetic contract dataset generator")
    doc.build(story, canvasmaker=factory)


def main() -> None:
    rng = random.Random(SEED); OUT_DIR.mkdir(parents=True, exist_ok=True); records = []
    for index in range(1, 21):
        spec = TEMPLATES[(index - 1) // 4]; data = make_data(rng, index, spec)
        docx_path = OUT_DIR / f"{data.contract_id}.docx"; pdf_path = OUT_DIR / f"{data.contract_id}.pdf"
        create_docx(data, docx_path); create_pdf(data, pdf_path)
        records.append({**asdict(data), "docx": docx_path.name, "pdf": pdf_path.name})
    manifest = {"dataset_name": "samr_multi_template_2025_synthetic_contracts", "generation_seed": SEED, "count": len(records), "template_count": len(TEMPLATES), "templates": TEMPLATES, "source_note": "分别参考国家市场监督管理总局合同示范文本库公开的五类合同结构；主体、金额、日期、标的和风险均为合成数据。", "records": records}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 多模板合成合同数据集", "", "本目录包含20份DOCX和20份PDF格式的合成合同，覆盖5种国家市场监督管理总局示范文本，每种4份。", "", f"- 随机种子：{SEED}", "- 模板：城镇房屋租赁、委托、中介、数据提供、保管", "- 风险标签：clean、amount_conflict、date_conflict、party_conflict、reference_conflict", "", "文件仅用于合同解析、条款切分、实体抽取和风险检测实验，不具有法律效力，也不代表官方合同。", ""]
    for spec in TEMPLATES: lines.append(f"- [{spec['template_id']}] {spec['title']}：{spec['source_url']}")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "count": len(records), "templates": len(TEMPLATES)}, ensure_ascii=False))


if __name__ == "__main__": main()
