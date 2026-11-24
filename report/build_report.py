"""Build one Word report from src/report_template.md and measured CSV tables.

The report contains its required abstract; notebooks/build_notebook.py omits that
section from the notebook. Both use the same results and final-insight bullets.
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
    """Arrange the shared evidence into the seven requested report sections.

    The notebook keeps its step-by-step observations and 300-word insight list.
    The Word report uses prose and a conventional methodology/results structure.
    """
    shared = render_markdown()

    def major(title):
        pattern = r"(?ms)^## " + re.escape(title) + r"\n(.*?)(?=^## |\Z)"
        match = re.search(pattern, shared)
        if not match:
            raise ValueError(f"Missing shared section: {title}")
        return match.group(1).strip()

    def subsection(text, title):
        pattern = r"(?ms)^### " + re.escape(title) + r"\n(.*?)(?=^### |\Z)"
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Missing evidence subsection: {title}")
        return match.group(1).strip()

    def take_paragraph(text, starts_with):
        paragraphs = text.split("\n\n")
        selected = [p for p in paragraphs if p.startswith(starts_with)]
        if len(selected) != 1:
            raise ValueError(f"Expected one paragraph starting: {starts_with}")
        return selected[0], "\n\n".join(p for p in paragraphs if p != selected[0]).strip()

    steps = {}
    for number in range(1, 9):
        match = re.search(r"(?ms)^## Step " + str(number) + r"\.[^\n]*\n(.*?)(?=^## |\Z)", shared)
        if not match:
            raise ValueError(f"Missing step {number}")
        steps[number] = match.group(1).split("**Observations**", 1)[0].strip()

    title = shared.split("## Abstract", 1)[0].strip()
    intro = major("Introduction")
    intro, verification = intro.split("\n\n**What was checked.** ", 1)
    performance = subsection(steps[5], "Predictive performance")
    metric_method, performance = take_paragraph(performance, "The test set contains")
    robustness = subsection(steps[5], "Robustness and unseen-specialty detection")
    perturb_method, robustness = take_paragraph(robustness, "The robustness tests introduce")
    ood_method, robustness = take_paragraph(robustness, "Unseen-specialty detection uses")
    fairness = subsection(steps[5], "Exploratory fairness assessment")
    fairness_method, fairness = take_paragraph(fairness, "Demographic parity difference compares")
    efficiency = subsection(steps[6], "Computational efficiency")
    efficiency_method, efficiency = take_paragraph(efficiency, "Each model was measured")
    interpretation = subsection(steps[6], "Interpretability")
    shap_method, interpretation = take_paragraph(interpretation, "Linear SHAP explains")
    local_method, interpretation = take_paragraph(interpretation, "Two selected notes illustrate")
    statistics = steps[7]
    _, statistics = take_paragraph(statistics, "McNemar's test compares")
    cv_method, statistics = take_paragraph(statistics, "Resampling uses")
    error_analysis = subsection(steps[8], "Error analysis")
    ethics = subsection(steps[8], "Ethical reliability and clinical relevance")

    quality_path = ROOT / "results/data_quality_audit.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality["failed"]:
        raise ValueError("Data-quality audit contains failed checks")
    cleanliness = (
        "The repeated cleaning audit rebuilt every processed row from the raw file. Across the training, "
        "validation, test, and unseen-specialty sets, it found no missing cells, blank notes, untrimmed text, "
        "repeated whitespace, control characters, or word-count mismatches. Exact text overlap between "
        "each split pair was zero. Ignoring case and punctuation also produced zero duplicate groups. "
        "Raw input validation now rejects missing columns and removes missing or whitespace-only labels "
        "and notes before grouping; these safeguards did not change the current benchmark.\n\n"
        f"A repeated normalization pass would replace a separator in {quality['repeat_normalization_punctuation_only_documents']} "
        "documents without changing letters or digits. No further matches to the implemented identifier "
        "patterns were found. The original cleaned inputs were retained so that the saved predictions, "
        "row order, and input hashes remain aligned. Neither this check nor zero regex matches establishes "
        "complete de-identification. Counts and checks are recorded in `results/data_quality_audit.json`."
    )
    mcnemar_method = (
        "McNemar's test compares paired correctness on the common test notes. Fewer than 25 discordant "
        "pairs trigger an exact binomial test; otherwise, continuity-corrected chi-square is used. Holm "
        "correction covers all 21 architecture pairs. This tests accuracy differences. Paired bootstrap "
        "intervals compare macro-F1 using 1,000 shared resamples; those intervals are pointwise and "
        "exclude variability from retraining or model selection. The comparison reference is chosen "
        "by validation macro-F1."
    )
    mcnemar_results = (
        "After Holm correction, only three accuracy comparisons fall below p = .05: Logistic Regression, "
        "SVM, and PubMedBERT each outperform Random Forest. The remaining tests leave differences "
        "unresolved; they do not establish equivalence."
    )
    statistics = statistics.replace("\n\n| Accuracy comparison", "\n\n" + mcnemar_results + "\n\n| Accuracy comparison", 1)
    insights = [line[2:] for line in major("Final insights").splitlines() if line.startswith("- ")]
    discussion = "\n\n".join(insights)
    limitations = (
        "The main training limitation is the use of one seed, two learning rates, a 256-token input limit, "
        "and partial transformer fine-tuning. Frozen-embedding probes were resampled, but the final "
        "fine-tuned transformers were not retrained within cross-validation. Classical resampling uses "
        "previously chosen hyperparameters, and the SVM resampling evaluates its uncalibrated classifier. "
        "These differences limit claims about architecture stability and the full tuning procedure. "
        "The demographic audit covers only the classes with sufficient proxy-group support; its missing "
        "values and unverified group assignments prevent a conclusion that any model is fair."
    )
    recommendations = [re.sub(r"^\d+\.\s*", "", line) for line in major("Recommendations").splitlines() if re.match(r"^\d+\. ", line)]
    pipeline = re.search(r"(?m)^!\[Figure 1\..*$", shared).group(0)
    references = re.sub(r"(?m)(https?://\S+)$", lambda match: f"[{match.group(1)}]({match.group(1)})", major("References"))
    report = "\n\n".join([
        title, "## Abstract", major("Abstract"), "## Introduction", intro,
        "## Methodology", pipeline, "### Data collection and cleaning", steps[1], cleanliness,
        "### Text preprocessing and data partitions", steps[2], "### Model architectures", steps[3],
        "### Training and experiment tracking", steps[4],
        "### Performance metrics and statistical design", metric_method, mcnemar_method, cv_method,
        "### Robustness and demographic comparisons", perturb_method, ood_method, fairness_method,
        "### Efficiency measurements and explanation methods", efficiency_method, shap_method, local_method,
        "### Verification and reproducibility", verification,
        "## Results", "### Classification performance", performance,
        "### Robustness and unseen specialties", robustness, "### Subgroup and counterfactual results", fairness,
        "### Computational efficiency", efficiency, "### Explanations and error analysis", interpretation, error_analysis,
        "### Statistical comparisons and optimization", statistics,
        "## Discussion", discussion, "### Limitations and bias assessment", limitations,
        "### Clinical implications and ethical considerations", ethics,
        "## Conclusion", major("Conclusions"), "### Deployment recommendations",
        "\n\n".join(recommendations[i] for i in (1, 2, 3, 4)),
        "### Future research", "\n\n".join(recommendations[i] for i in (0, 5, 6, 7)),
        "## References", references
    ]) + "\n"
    # Moving the error analysis ahead of statistics also moves its per-class table.
    # Keep captions and in-text table references consecutive in the final report.
    table_order = re.findall(r"(?m)^\*Table (\d+)\.", report)
    table_numbers = {old: str(index) for index, old in enumerate(table_order, 1)}
    report = re.sub(r"\bTable (\d+)\b", lambda match: "Table " + table_numbers.get(match.group(1), match.group(1)), report)
    expected = ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion", "References"]
    if re.findall(r"(?m)^## (.+)$", report) != expected:
        raise ValueError("Technical report section order does not match the assignment")
    if "**Observations**" in report or re.search(r"(?m)^## Step ", report):
        raise ValueError("Notebook step formatting leaked into the technical report")
    return report


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
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    if "Hyperlink" not in document.styles:
        hyperlink = document.styles.add_style("Hyperlink", WD_STYLE_TYPE.CHARACTER)
        hyperlink.font.color.rgb = RGBColor.from_string("0563C1")
        hyperlink.font.underline = True
    for style_name, size in (("Title", 19), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)):
        style = document.styles[style_name]
        style.font.name, style.font.size = "Arial", Pt(size)
        style.font.color.rgb = RGBColor.from_string("182536")
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(14)
        style.paragraph_format.space_after = Pt(6)
    document.styles["Caption"].font.size = Pt(9)
    document.styles["Caption"].font.color.rgb = RGBColor.from_string("42546A")
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
                        shade.set(qn("w:fill"), "EAF1F7")
                        cell._tc.get_or_add_tcPr().append(shade)
            document.add_paragraph().paragraph_format.space_after = Pt(2)
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
