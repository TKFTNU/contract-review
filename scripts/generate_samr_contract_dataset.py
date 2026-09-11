from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from docx.oxml.ns import qn
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


SEED = 20260911
SOURCE_URL = "https://htsfwb.samr.gov.cn/View?id=cbb13efd-97cf-4941-a76a-1627faf942cc"
TEMPLATE_ID = "SDF-2025-0002"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "generated_contracts" / "samr_sdf_2025_0002"


@dataclass(slots=True)
class ContractData:
    contract_id: str
    city: str
    seller: str
    buyer: str
    address: str
    area: str
    unit_price: int
    total_price: int
    deposit: int
    balance: int
    signing_date: str
    summary_delivery_date: str
    delivery_date: str
    deed_date: str
    risk_type: str
    risk_detail: str


def money_upper(value: int) -> str:
    digits = "零壹贰叁肆伍陆柒捌玖"
    units = ["", "拾", "佰", "仟", "万", "拾", "佰", "仟", "亿"]
    if value == 0:
        return "零元整"
    text = ""
    s = str(value)
    zero = False
    for offset, char in enumerate(reversed(s)):
        digit = int(char)
        pos = offset
        if digit == 0:
            zero = bool(text)
            continue
        if zero and text:
            text = "零" + text
        zero = False
        text = digits[digit] + units[pos] + text
    return text + "元整"


def fmt_date(value: date) -> str:
    return f"{value.year}年{value.month}月{value.day}日"


def make_data(rng: random.Random, index: int) -> ContractData:
    cities = [
        ("南京市", "建邺区", "某某路88号锦绣花园"),
        ("济南市", "历下区", "文化东路66号泉城雅苑"),
        ("青岛市", "市南区", "香港中路18号海景名苑"),
        ("苏州市", "姑苏区", "平江路28号姑苏府"),
        ("合肥市", "蜀山区", "望江西路108号翡翠湖畔"),
        ("杭州市", "拱墅区", "湖墅南路36号运河华庭"),
        ("武汉市", "武昌区", "中北路128号东湖新城"),
        ("成都市", "锦江区", "东大街52号锦江公馆"),
    ]
    surnames = ["张", "李", "王", "赵", "陈", "刘", "杨", "黄", "周", "吴"]
    given = ["明", "芳", "伟", "静", "磊", "婷", "浩", "雪", "杰", "琳"]
    city, district, community = cities[index % len(cities)]
    seller = rng.choice(surnames) + rng.choice(given) + rng.choice(given)
    buyer = rng.choice([s for s in surnames if s != seller[0]]) + rng.choice(given) + rng.choice(given)
    building = f"{rng.randint(1, 12)}幢{rng.randint(1, 3)}单元{rng.randint(101, 1802)}室"
    address = f"{city}{district}{community}{building}"
    area = f"{rng.uniform(76, 148):.2f}"
    unit_price = rng.randrange(18_000, 42_001, 500)
    total_price = int(round(float(area) * unit_price / 10_000) * 10_000)
    deposit = int(round(total_price * rng.choice([0.03, 0.05, 0.08]) / 1_000) * 1_000)
    balance = total_price - deposit
    signing = date(2026, 9, 11) + timedelta(days=index * 3)
    delivery = signing + timedelta(days=rng.choice([45, 60, 75, 90]))
    deed = delivery + timedelta(days=rng.choice([30, 45, 60]))

    risk_types = [
        ("clean", "无预设矛盾，用于正向结构样本"),
        ("amount_conflict", "将付款分项金额设置为与房屋总价不一致"),
        ("date_conflict", "将交付日期与付款条款中的日期设置为不一致"),
        ("party_conflict", "在签署区替换买受人姓名，制造主体不一致"),
        ("reference_conflict", "将正文引用的附件编号设置为错误编号"),
    ]
    risk_type, risk_detail = risk_types[(index - 1) % len(risk_types)]
    if risk_type == "amount_conflict":
        balance += rng.choice([10_000, 20_000, 30_000])
    elif risk_type == "date_conflict":
        delivery = delivery + timedelta(days=15)
    elif risk_type == "reference_conflict":
        risk_detail = "将正文中的附件四引用改为附件九"

    return ContractData(
        contract_id=f"SYN-SDF-{index:04d}",
        city=city,
        seller=seller,
        buyer=buyer,
        address=address,
        area=area,
        unit_price=unit_price,
        total_price=total_price,
        deposit=deposit,
        balance=balance,
        signing_date=fmt_date(signing),
        summary_delivery_date=fmt_date(delivery - timedelta(days=30)) if risk_type == "date_conflict" else fmt_date(delivery),
        delivery_date=fmt_date(delivery),
        deed_date=fmt_date(deed),
        risk_type=risk_type,
        risk_detail=risk_detail,
    )


def clause_paragraphs(data: ContractData) -> list[tuple[str, str]]:
    # Avoid line-breaking inside grouped digits in narrow PDF paragraphs.
    total_display = f"人民币（小写）{data.total_price:.2f}元，（大写）{money_upper(data.total_price)}"
    deposit_display = f"人民币（小写）{data.deposit:.2f}元，（大写）{money_upper(data.deposit)}"
    balance_display = f"人民币（小写）{data.balance:.2f}元，（大写）{money_upper(data.balance)}"
    attachment_ref = "附件九" if data.risk_type == "reference_conflict" else "附件四"
    delivery_display = data.delivery_date
    signing_buyer = f"林{data.buyer[1:]}" if data.risk_type == "party_conflict" else data.buyer
    amount_note = (
        f"首付款{deposit_display}，余款{balance_display}。"
        if data.risk_type != "amount_conflict"
        else f"首付款{deposit_display}，余款{balance_display}（系统应核对该分项合计与合同总价）。"
    )
    return [
        ("heading", "第一章 合同当事人"),
        ("body", f"出卖人（甲方）：{data.seller}；通讯地址：{data.city}示范区合同路1号；联系电话：1380000{int(data.contract_id[-4:]):04d}。"),
        ("body", f"买受人（乙方）：{data.buyer}；户籍所在地：中国；证件类型：居民身份证；证件号码：3201{int(data.contract_id[-4:]):08d}。"),
        ("heading", "第二章 商品房基本状况"),
        ("heading", "第一条 项目建设依据"),
        ("body", f"甲方以出让方式取得坐落于{data.city}{data.address[:3]}地块的国有建设用地使用权，土地用途为城镇住宅用地。"),
        ("heading", "第二条 商品房基本情况"),
        ("body", f"甲方将坐落于{data.address}的商品房出售给乙方。该房屋建筑面积为{data.area}平方米，规划用途为住宅，不动产权证书号为苏（2020）宁建不动产权第{data.contract_id[-4:]}号。"),
        ("body", "该房屋抵押情况：□有抵押  ☑无抵押；租赁情况：□已出租  ☑未出租。"),
        ("heading", "第三章 商品房价款"),
        ("heading", "第三条 计价方式与价款"),
        ("body", f"双方同意按照建筑面积计算房屋价款，单价为每平方米人民币{data.unit_price:,.2f}元，总价款为{total_display}。"),
        ("heading", "第四条 付款方式及期限"),
        ("body", "（一）双方选择资金监管方式付款，乙方应将购房价款存入监管账户。"),
        ("body", f"（二）签订本合同前，乙方已支付定金{deposit_display}。"),
        ("body", f"（三）付款安排：1. {amount_note} 2. 余款应于{data.delivery_date}前支付。"),
        ("heading", "第四章 商品房交付条件与交付手续"),
        ("heading", "第五条 商品房交付时间和条件"),
        ("body", f"甲方应于{delivery_display}前向乙方交付房屋。交付时房屋应完成竣工验收并具备通水、通电等正常使用条件。"),
        ("body", f"乙方应在交付前完成房屋查验；交付手续、相关证明文件和物业资料应当一并提供。"),
        ("heading", "第六条 逾期交付责任"),
        ("body", "除不可抗力外，甲方逾期交付的，应按日向乙方支付逾期应付款万分之三的违约金；逾期超过六十日的，乙方有权解除合同。"),
        ("heading", "第五章 房屋权利与双方义务"),
        ("heading", "第七条 房屋权利状况承诺"),
        ("body", "甲方保证对该房屋享有合法、完整的处分权，房屋不存在司法查封或其他限制转让情形。"),
        ("body", "（1）甲方应如实披露房屋权属、抵押和租赁情况；（2）乙方应按约支付价款并配合办理登记。"),
        ("heading", "第八条 违约责任"),
        ("body", "一方违约造成对方损失的，应承担继续履行、采取补救措施或赔偿损失等责任。违约金不足以弥补损失的，仍应补足差额。"),
        ("heading", "第六章 其他事项"),
        ("heading", "第九条 税费与不动产登记"),
        ("body", f"双方依法各自承担应缴税费，并应于{data.deed_date}前共同申请办理不动产权转移登记。相关配套约定见{attachment_ref}。"),
        ("heading", "第十条 争议解决方式"),
        ("body", f"本合同履行过程中发生争议，双方协商解决；协商不成的，依法向{data.city}有管辖权的人民法院起诉。"),
        ("heading", "第十一条 合同生效"),
        ("body", "本合同自双方签字或盖章之日起生效。本合同一式三份，甲乙双方及登记机构各执一份。"),
        ("body", f"甲方（签章）：{data.seller}    签约日期：{data.signing_date}"),
        ("body", f"乙方（签章）：{signing_buyer}    签约日期：{data.signing_date}"),
        ("heading", "附件一 房屋平面及基本信息"),
        ("body", f"1. 房屋坐落：{data.address}；2. 建筑面积：{data.area}平方米；3. 房屋用途：住宅。"),
        ("heading", "附件二 交付资料清单"),
        ("body", "1. 竣工验收资料；2. 住宅使用说明书；3. 房屋质量保证书；4. 物业服务资料。"),
        ("heading", "附件三 抵押和权利状况说明"),
        ("body", "本交易标的未设抵押，未被司法查封，未出租；如实际情况发生变化，甲方应及时书面通知乙方。"),
        ("heading", "附件四 付款及资金监管约定"),
        ("body", f"监管账户收款安排：定金{deposit_display}，余款{balance_display}。"),
    ]


def set_docx_defaults(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)


def add_docx_table(doc: Document, data: ContractData) -> None:
    table = doc.add_table(rows=4, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    values = [
        ("合同编号", data.contract_id, "示范文本编号", TEMPLATE_ID),
        ("出卖人", data.seller, "买受人", data.buyer),
        ("房屋坐落", data.address, "建筑面积", f"{data.area}平方米"),
        ("签约日期", data.signing_date, "约定交付日", data.summary_delivery_date),
        ("数据类型", "合成数据", "风险标签", data.risk_type),
    ]
    for row, values_row in zip(table.rows, values):
        for cell, value in zip(row.cells, values_row):
            cell.text = value
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
    for row in table.rows:
        row.cells[0].width = Cm(2.4)
        row.cells[2].width = Cm(2.4)


def create_docx(data: ContractData, path: Path) -> None:
    doc = Document()
    set_docx_defaults(doc)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("山东省新建商品房买卖合同（预售）")
    run.bold = True
    run.font.size = Pt(16)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("基于国家市场监督管理总局合同示范文本结构生成的合成数据").italic = True
    add_docx_table(doc, data)
    note = doc.add_paragraph("说明：本文件仅用于合同解析、条款切分和风险检测实验，不具有法律效力，不代表官方合同文本。")
    note.runs[0].font.size = Pt(8.5)
    for kind, text in clause_paragraphs(data):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(4 if kind == "body" else 8)
        if kind == "heading":
            paragraph.style = doc.styles["Heading 2"]
            heading_run = paragraph.add_run(text)
            heading_run.font.name = "黑体"
            heading_run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        else:
            paragraph.add_run(text)
    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run(f"{data.contract_id} | Synthetic dataset | No legal effect").font.size = Pt(8)
    doc.save(path)


def register_pdf_font() -> str:
    font_path = Path(r"C:\Windows\Fonts\simfang.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("SimFang", str(font_path)))
        return "SimFang"
    return "Helvetica"


class FooterCanvas(Canvas):
    def __init__(self, *args, contract_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_id = contract_id

    def showPage(self) -> None:
        self.saveState()
        self.setFont("SimFang", 7)
        self.setFillColor(colors.HexColor("#666666"))
        self.drawCentredString(A4[0] / 2, 1.0 * cm, f"{self.contract_id} | Synthetic dataset | No legal effect")
        self.restoreState()
        super().showPage()

    def save(self) -> None:
        if self._pageNumber == 0:
            self.showPage()
        super().save()


def create_pdf(data: ContractData, path: Path) -> None:
    font = register_pdf_font()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ContractTitle", parent=styles["Title"], fontName=font, fontSize=16,
        leading=22, alignment=TA_CENTER, spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "ContractSubtitle", parent=styles["Normal"], fontName=font, fontSize=8.5,
        leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#666666"), spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "ContractHeading", parent=styles["Heading2"], fontName=font, fontSize=11.5,
        leading=16, alignment=TA_LEFT, spaceBefore=8, spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ContractBody", parent=styles["BodyText"], fontName=font, fontSize=9.2,
        leading=15, alignment=TA_LEFT, spaceAfter=4,
    )
    note_style = ParagraphStyle(
        "ContractNote", parent=body_style, fontSize=7.8, leading=11,
        textColor=colors.HexColor("#666666"),
    )
    cell_style = ParagraphStyle(
        "ContractCell", parent=body_style, fontName=font, fontSize=8.1,
        leading=10, spaceAfter=0,
    )
    story: list[Flowable] = [
        Paragraph("山东省新建商品房买卖合同（预售）", title_style),
        Paragraph("基于国家市场监督管理总局合同示范文本结构生成的合成数据", subtitle_style),
    ]
    table_data = [
        ["合同编号", data.contract_id, "示范文本编号", TEMPLATE_ID],
        ["出卖人", data.seller, "买受人", data.buyer],
        ["房屋坐落", data.address, "建筑面积", f"{data.area}平方米"],
        ["签约日期", data.signing_date, "约定交付日", data.summary_delivery_date],
        ["数据类型", "合成数据", "风险标签", data.risk_type],
    ]
    table_data = [[Paragraph(str(value), cell_style) for value in row] for row in table_data]
    table = Table(table_data, colWidths=[2.2 * cm, 5.1 * cm, 2.2 * cm, 5.1 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9D9D9")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EEF3F8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 8), Paragraph("说明：本文件仅用于合同解析、条款切分和风险检测实验，不具有法律效力，不代表官方合同文本。", note_style), Spacer(1, 6)])
    for kind, text in clause_paragraphs(data):
        if kind == "heading":
            story.append(Paragraph(text, heading_style))
        else:
            story.append(Paragraph(text.replace("&", "&amp;"), body_style))

    def canvas_factory(*args, **kwargs):
        return FooterCanvas(*args, contract_id=data.contract_id, **kwargs)

    doc = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm,
        topMargin=1.8 * cm, bottomMargin=1.7 * cm,
        title=f"{data.contract_id} 山东省新建商品房买卖合同（预售）",
        author="Synthetic contract dataset generator",
    )
    doc.build(story, canvasmaker=canvas_factory)


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index in range(1, 21):
        data = make_data(rng, index)
        docx_path = OUT_DIR / f"{data.contract_id}.docx"
        pdf_path = OUT_DIR / f"{data.contract_id}.pdf"
        create_docx(data, docx_path)
        create_pdf(data, pdf_path)
        records.append({**asdict(data), "docx": docx_path.name, "pdf": pdf_path.name})
    manifest = {
        "dataset_name": "samr_sdf_2025_0002_synthetic_contracts",
        "template_id": TEMPLATE_ID,
        "source_url": SOURCE_URL,
        "source_note": "结构参考国家市场监督管理总局合同示范文本库公开的山东省2025版商品房预售合同；所有主体、金额、日期和风险均为合成数据。",
        "generation_seed": SEED,
        "count": len(records),
        "records": records,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "# 合成合同数据集\n\n"
        "本目录包含20份DOCX和20份PDF格式的合成商品房预售合同。\n\n"
        f"- 结构来源：{SOURCE_URL}\n"
        f"- 示范文本编号：{TEMPLATE_ID}\n"
        f"- 随机种子：{SEED}\n"
        "- 风险标签：clean、amount_conflict、date_conflict、party_conflict、reference_conflict\n\n"
        "文件仅用于合同解析、条款切分、实体抽取和风险检测实验，不具有法律效力，也不代表国家市场监督管理总局发布的官方合同。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(OUT_DIR), "count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
