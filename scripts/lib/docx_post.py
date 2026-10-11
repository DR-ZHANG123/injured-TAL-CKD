"""Post-process a pandoc .docx so that it follows the journal's manuscript format and the house figure rules.

- Times New Roman 12 pt, double line spacing, A4 page with 2.54 cm margins, continuous line numbers, page numbers;
- each figure starts a new page and is kept with its legend (keep-with-next on the image, keep-lines on the legend);
  legends and tables are single spaced in 10 pt so that a figure and its legend fit on one page;
- table title, table rows and table note are kept together;
- tool metadata (creator, description) are cleared.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

FOOTER = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:ftr xmlns:w="{W}" xmlns:r="{R}"><w:p><w:pPr>'
          '<w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve">'
          ' PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r><w:r>'
          '<w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
SECT = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdPageFooter"/><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:lnNumType w:countBy="1" w:distance="283" w:restart="continuous"/><w:pgNumType w:start="1"/></w:sectPr>')
LEGEND_STYLE = ('<w:style w:type="paragraph" w:customStyle="1" w:styleId="Legend"><w:name w:val="Legend"/>'
                '<w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="120" w:after="0" w:line="240" '
                'w:lineRule="auto"/><w:jc w:val="both"/></w:pPr><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>')


def top_level_blocks(body: str) -> list[str]:
    """Split the body into top-level <w:p>, <w:tbl> and <w:sectPr> elements."""
    out, i = [], 0
    while i < len(body):
        m = re.compile(r"<w:(p|tbl|sectPr|bookmarkStart|bookmarkEnd)\b").search(body, i)
        if not m:
            break
        tag = m.group(1)
        if body[m.start():].startswith(f"<w:{tag}/>") or re.match(rf"<w:{tag}\b[^>]*/>", body[m.start():]):
            end = body.index("/>", m.start()) + 2
        else:
            depth, j = 0, m.start()
            pat = re.compile(rf"<w:{tag}\b[^>]*?(/?)>|</w:{tag}>")
            while True:
                n = pat.search(body, j)
                if n.group(0).startswith("</"):
                    depth -= 1
                elif not n.group(1):
                    depth += 1
                j = n.end()
                if depth == 0:
                    break
            end = j
        out.append(body[m.start():end])
        i = end
    return out


def text_of(xml: str) -> str:
    return "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", xml))


def set_ppr(p: str, add: str, style: str | None = None) -> str:
    """Insert paragraph properties (and optionally replace the style) in a <w:p> element."""
    if "<w:pPr>" not in p:
        p = re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr></w:pPr>", p, count=1)
    if style:
        p = re.sub(r"<w:pStyle w:val=\"[^\"]*\"\s*/>", "", p, count=1)
        p = p.replace("<w:pPr>", f'<w:pPr><w:pStyle w:val="{style}"/>', 1)
    # property order inside pPr: pStyle, keepNext, keepLines, pageBreakBefore, ..., spacing, jc
    return re.sub(r"(<w:pPr>(?:<w:pStyle [^>]*/>)?)", r"\1" + add, p, count=1)


def single_spaced(xml: str) -> str:
    """Single spacing (and keep-together) for every paragraph inside a table."""
    props = '<w:keepNext/><w:keepLines/><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
    return re.sub(r"(<w:p(?: [^>]*)?>)(<w:pPr>(?:<w:pStyle [^>]*/>)?)?", lambda m: _rep_p(m, props), xml)


def _rep_p(m, props: str) -> str:
    if m.group(2):                       # existing pPr (with optional style): insert after it
        return m.group(1) + m.group(2) + props
    return m.group(1) + "<w:pPr>" + props + "</w:pPr>"


def process_document(doc: str, numbered: bool = True) -> str:
    head, rest = doc.split("<w:body>", 1)
    body, tail = rest.rsplit("</w:body>", 1)
    blocks = [b for b in top_level_blocks(body) if not b.startswith("<w:sectPr")]
    out, after_image, n_img = [], False, 0
    for k, b in enumerate(blocks):
        txt = text_of(b)
        if b.startswith("<w:p") and "<w:drawing>" in b:
            n_img += 1
            prev = text_of(blocks[k - 1]) if k else ""
            brk = "" if prev in ("Figures", "Supplementary Figures") else "<w:pageBreakBefore/>"
            b = set_ppr(b, "<w:keepNext/>" + brk)
            after_image = True
        elif b.startswith("<w:p") and after_image:
            b = set_ppr(b, "<w:keepLines/>", style="Legend")
            after_image = False
        elif b.startswith("<w:p") and re.match(r"Table S?\d+ ", txt):
            b = set_ppr(b, "<w:keepNext/><w:keepLines/>", style="Legend")
        elif b.startswith("<w:tbl"):
            b = single_spaced(b)
        elif b.startswith("<w:p") and k and blocks[k - 1].startswith("<w:tbl"):
            b = set_ppr(b, "<w:keepLines/>", style="Legend")
        elif b.startswith("<w:p") and txt in ("Tables", "Figures", "References", "Supplementary Figures",
                                              "Supplementary Tables"):
            b = set_ppr(b, "<w:pageBreakBefore/>")
        out.append(b)
    if f'xmlns:r="{R}"' not in head:
        head = head.replace("<w:document ", f'<w:document xmlns:r="{R}" ', 1)
    sect = SECT if numbered else re.sub(r"<w:lnNumType [^>]*/>", "", SECT)
    return head + "<w:body>" + "".join(out) + sect + "</w:body>" + tail


def process_styles(st: str, line: int = 480) -> str:
    st = re.sub(r"<w:rFonts [^>]*/>", '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
                'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>', st)
    st = re.sub(r"(<w:docDefaults>.*?<w:rPr>)", r"\1", st, flags=re.S)
    st = re.sub(r"<w:sz w:val=\"\d+\"\s*/>", '<w:sz w:val="24"/>', st, count=1)
    # double spacing for body text styles
    for sid in ("BodyText", "FirstParagraph", "Compact", "Abstract"):
        st = re.sub(rf'(<w:style [^>]*w:styleId="{sid}"[^>]*>.*?)(<w:pPr>.*?</w:pPr>|</w:style>)',
                    lambda m: m.group(1) + (re.sub(r"<w:spacing [^>]*/>", "", m.group(2)).replace(
                        "<w:pPr>", '<w:pPr><w:spacing w:before="0" w:after="120" w:line="LINE" w:lineRule="auto"/>')
                        if m.group(2).startswith("<w:pPr>") else
                        '<w:pPr><w:spacing w:before="0" w:after="120" w:line="LINE" w:lineRule="auto"/></w:pPr></w:style>'),
                    st, count=1, flags=re.S)
    st = st.replace('w:line="LINE"', f'w:line="{line}"')
    st = re.sub(r'<w:color w:val="[0-9A-Fa-f]{6}"\s*/>', "", st)       # black headings
    return st.replace("</w:styles>", LEGEND_STYLE + "</w:styles>")


def postprocess(path: Path, manuscript: bool = True) -> None:
    """manuscript=True: double spacing and line numbers (journal manuscript); False: 1.15 spacing (supplement)."""
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = process_document(data.decode(), manuscript).encode()
            elif item.filename == "word/styles.xml":
                data = process_styles(data.decode(), 480 if manuscript else 276).encode()
            elif item.filename == "word/_rels/document.xml.rels":
                data = data.decode().replace("</Relationships>", f'<Relationship Id="rIdPageFooter" Type="{R}/footer" '
                                             'Target="footer1.xml"/></Relationships>').encode()
            elif item.filename == "[Content_Types].xml":
                data = data.decode().replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/'
                                             'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>').encode()
            elif item.filename == "docProps/core.xml":
                d = data.decode()
                d = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator></dc:creator>", d)
                d = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "", d)
                d = re.sub(r"<dc:description>.*?</dc:description>", "", d, flags=re.S)
                data = d.encode()
            elif item.filename == "docProps/custom.xml":
                data = re.sub(r"<property [^>]*>.*?</property>", "", data.decode(), flags=re.S).encode()
            zout.writestr(item, data)
        zout.writestr("word/footer1.xml", FOOTER)
    shutil.move(tmp, path)
