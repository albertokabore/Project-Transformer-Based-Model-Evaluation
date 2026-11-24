"""Build one Word report from src/report_template.md and measured CSV tables.

The report contains its required abstract and a separate narrative template.
The notebook keeps its step-based format and 300-word final-insight list.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
REPORT = ROOT / "report"


def fmt(value, digits):
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.1e}" if 0 < abs(value) < 1e-3 else f"{value:.{digits}f}"
    return str(value)


def table(spec):
    parts = spec.split("|")
    df = pd.read_csv(TABLES / f"{parts[0]}.csv")
    if len(parts) > 1 and parts[1]:
        df = df[[c.strip().replace("!", "|") for c in parts[1].split(",")]]
    digits = int(parts[2]) if len(parts) > 2 and parts[2] else 3
    df = df.rename(columns={"Unnamed: 0": "Specialty", "peak memory (MB)": "peak memory (MiB)",
                            "serialized size (MB)": "serialized size (MiB)"})

    def cell(value):
        return fmt(value, digits).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(cell(c) for c in df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(cell(v) for v in row) + " |\n" for row in df.itertuples(index=False))
    return "\n" + head + body + "\n"


def validate_observations(text):
    blocks = re.findall(r"\*\*Observations\*\*\n\n(.*?)(?=\n## )", text, re.S)
    if len(blocks) != 8:
        raise ValueError(f"Expected eight observation blocks, found {len(blocks)}")
    for step, block in enumerate(blocks, 1):
        bullets = [line[2:] for line in block.strip().splitlines() if line.startswith("- ")]
        count = len(" ".join(bullets).split())
        if len(bullets) != 2 or count != 50:
            raise ValueError(f"Step {step} observations: {count} whitespace-separated words, {len(bullets)} bullets")


def render_markdown():
    source = (ROOT / "src/report_template.md").read_text(encoding="utf-8")
    validate_observations(source)
    insights = source.split("## Final insights\n", 1)[1].split("\n## Recommendations", 1)[0]
    bullets = [line[2:] for line in insights.splitlines() if line.startswith("- ")]
    if len(bullets) != 6 or len(" ".join(bullets).split()) != 300:
        raise ValueError("Final insights must contain six bullets totaling 300 words")
    text = re.sub(r"\{\{table:([^}]+)\}\}", lambda match: table(match.group(1)), source)
    if re.search(r"%%|\{\{", text):
        raise ValueError("Report contains unresolved placeholders")
    return text


def render_technical_report():
    """Render the concise report; preserve the notebook's separate step-based format."""
    source = (ROOT / "src/technical_report_template.md").read_text(encoding="utf-8")
    quality = json.loads((ROOT / "results/data_quality_audit.json").read_text(encoding="utf-8"))
    if quality["failed"]:
        raise ValueError("Data-quality audit contains failed checks")
    cleaning = (
        "Quality checks found no missing fields, empty notes, whitespace defects, control characters, "
        "or word-count mismatches in the retained data. No duplicate texts remained within or between "
        "splits, including when case and punctuation were ignored. A second normalization pass would "
        f"change one separator in {quality['repeat_normalization_punctuation_only_documents']} notes. "
        "The one-pass inputs were retained to preserve alignment with the saved predictions."
    )
    references = render_markdown().split("## References", 1)[1].strip()
    references = re.sub(r"(?m)(https?://\S+)$", lambda match: f"[{match.group(1)}]({match.group(1)})", references)
    text = source.replace("{{cleaning_audit}}", cleaning).replace("{{references}}", references)
    text = re.sub(r"\{\{table:([^}]+)\}\}", lambda match: table(match.group(1)), text)
    expected = ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion", "References"]
    if re.findall(r"(?m)^## (.+)$", text) != expected:
        raise ValueError("Technical report section order does not match the assignment")
    conclusion = text.split("## Conclusion\n", 1)[1].split("## References\n", 1)[0]
    if re.search(r"(?m)^#", conclusion):
        raise ValueError("Conclusion must be the last report section before references")
    if "**Observations**" in text or re.search(r"(?m)^## Step ", text):
        raise ValueError("Notebook step formatting leaked into the technical report")
    if re.search(r"%%|\{\{", text):
        raise ValueError("Report contains unresolved placeholders")
    return text



def make_word(body_html, destination):
    from bs4 import BeautifulSoup, NavigableString
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
    from docx.opc.constants import RELATIONSHIP_TYPE

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(.8)
    section.left_margin = section.right_margin = Inches(.8)
    width = section.page_width - section.left_margin - section.right_margin
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Hyperlink", "FollowedHyperlink"):
        hyperlink = document.styles[name] if name in document.styles else document.styles.add_style(name, WD_STYLE_TYPE.CHARACTER)
        hyperlink.font.color.rgb = RGBColor(0, 0, 0)
        hyperlink.font.underline = True
    for style_name, size in (("Title", 19), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)):
        style = document.styles[style_name]
        style.font.name, style.font.size = "Arial", Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(14)
        style.paragraph_format.space_after = Pt(6)
    document.styles["Caption"].font.size = Pt(9)
    document.styles["Caption"].font.color.rgb = RGBColor(0, 0, 0)
    for style in document.styles:
        if hasattr(style, "font"):
            style.font.color.rgb = RGBColor(0, 0, 0)
    document.core_properties.title = "Transformer-Based Model Evaluation"
    document.core_properties.author = "Albert Kabore"
    document.core_properties.subject = "Medical specialty classification: comparative model evaluation"

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Albert Kabore | ").font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    def add_link(paragraph, node, bold, italic):
        link = OxmlElement("w:hyperlink")
        rid = paragraph.part.relate_to(node["href"], RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        link.set(qn("r:id"), rid)
        run = OxmlElement("w:r")
        props = OxmlElement("w:rPr")
        style = OxmlElement("w:rStyle")
        style.set(qn("w:val"), "Hyperlink")
        props.append(style)
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "000000")
        props.append(color)
        for enabled, tag in ((bold, "w:b"), (italic, "w:i")):
            if enabled:
                props.append(OxmlElement(tag))
        run.append(props)
        text = OxmlElement("w:t")
        text.set(qn("xml:space"), "preserve")
        text.text = node.get_text()
        run.append(text)
        link.append(run)
        paragraph._p.append(link)

    def inline(paragraph, node, bold=False, italic=False):
        if isinstance(node, NavigableString):
            run = paragraph.add_run(str(node))
            run.bold, run.italic = bold, italic
            return
        if node.name == "br":
            paragraph.add_run().add_break()
        elif node.name == "a" and node.get("href"):
            add_link(paragraph, node, bold, italic)
        elif node.name == "code":
            run = paragraph.add_run(node.get_text())
            run.font.name, run.font.size = "Consolas", Pt(9)
        else:
            for child in node.children:
                inline(paragraph, child, bold or node.name in ("strong", "b"), italic or node.name in ("em", "i"))

    soup = BeautifulSoup(body_html, "html.parser")
    in_references = False
    before_abstract = True
    for node in soup.children:
        if not getattr(node, "name", None):
            continue
        if node.name in ("h1", "h2", "h3", "h4"):
            label = node.get_text()
            level = int(node.name[1]) - 1
            paragraph = document.add_heading(label, level=level)
            if level == 0:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if label == "Abstract":
                before_abstract = False
            in_references = label == "References" or (in_references and level > 1)
            if label == "References":
                paragraph.paragraph_format.page_break_before = True
        elif node.name == "p":
            picture = node.find("img")
            if picture:
                picture_path = (REPORT / picture["src"]).resolve()
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.keep_with_next = True
                paragraph.add_run().add_picture(str(picture_path), width=width)
                document.add_paragraph(picture.get("alt", ""), style="Caption")
            else:
                is_caption = node.get_text().startswith("Table ")
                paragraph = document.add_paragraph(style="Caption" if is_caption else None)
                inline(paragraph, node)
                if before_abstract:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if in_references:
                    paragraph.paragraph_format.left_indent = Inches(.5)
                    paragraph.paragraph_format.first_line_indent = Inches(-.5)
        elif node.name in ("ul", "ol"):
            for index, item in enumerate(node.find_all("li", recursive=False), 1):
                paragraph = document.add_paragraph(style="List Bullet" if node.name == "ul" else "List Paragraph")
                if node.name == "ol":
                    paragraph.add_run(f"{index}. ")
                    paragraph.paragraph_format.left_indent = Inches(.25)
                    paragraph.paragraph_format.first_line_indent = Inches(-.25)
                inline(paragraph, item)
        elif node.name == "table":
            rows = [row.find_all(["th", "td"], recursive=False) for row in node.find_all("tr")]
            n = len(rows[0])
            grid = document.add_table(rows=0, cols=n)
            grid.style = "Table Grid"
            grid.alignment = WD_TABLE_ALIGNMENT.CENTER
            grid.autofit = False
            widths = [int(width / n)] * n if n < 5 else [int(width * .22)] + [int(width * .78 / (n - 1))] * (n - 1)
            for column, column_width in zip(grid.columns, widths):
                column.width = column_width
            for i, row in enumerate(rows):
                new_row = grid.add_row()
                no_split = OxmlElement("w:cantSplit")
                new_row._tr.get_or_add_trPr().append(no_split)
                if i == 0:
                    new_row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
                for j, (cell, source) in enumerate(zip(new_row.cells, row)):
                    cell.width = widths[j]
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    paragraph = cell.paragraphs[0]
                    paragraph.paragraph_format.space_after = Pt(3)
                    paragraph.paragraph_format.space_before = Pt(3)
                    paragraph.paragraph_format.line_spacing = 1.0
                    inline(paragraph, source, bold=i == 0)
                    for run in paragraph.runs:
                        run.font.size = Pt(7.5 if n >= 7 else 8.5)
                    if i == 0:
                        shade = OxmlElement("w:shd")
                        shade.set(qn("w:fill"), "F2F2F2")
                        cell._tc.get_or_add_tcPr().append(shade)
            document.add_paragraph().paragraph_format.space_after = Pt(2)
    # Remove theme overrides that could recolor headings, links, or visited links.
    for element in (document._element, document.styles.element, section.footer._element):
        for color in element.iter(qn("w:color")):
            color.set(qn("w:val"), "000000")
            for attribute in ("themeColor", "themeTint", "themeShade"):
                color.attrib.pop(qn("w:" + attribute), None)
    document.save(destination)


def build():
    body = markdown.markdown(render_technical_report(), extensions=["tables", "sane_lists", "attr_list"])
    destination = REPORT / "Technical_Report.docx"
    staging = ROOT / "cache/report_revision"
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / destination.name
    make_word(body, temporary)
    try:
        temporary.replace(destination)
    except PermissionError as error:
        raise PermissionError(f"Close {destination.name} in Word before replacing it. The revised file is ready at {temporary}.") from error
    print("Built", destination.name)


if __name__ == "__main__":
    build()
