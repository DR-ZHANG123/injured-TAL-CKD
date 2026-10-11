"""Parse the LaTeX manuscript sources and turn them into a pandoc-ready LaTeX copy for the Word versions.

The LaTeX sources (manuscript/main.tex, manuscript/supplement.tex) are the single source of the text. The Word files are
generated from them: citations are replaced by the numbers that BibTeX assigned in the PDF build (read from the .aux
file), the reference list is taken from the .bbl file, and every \\figurepage / \\tablepage block becomes an image (or
table) followed by its own legend, so the PDF and the Word versions carry identical text and numbering.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


def find_group(s: str, start: int) -> tuple[str, int]:
    """Return the content of the balanced {...} group that opens at s[start] and the index after it."""
    assert s[start] == "{", s[start:start + 30]
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "\\":
            continue
        if c == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif c == "}" and s[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
    raise ValueError("unbalanced group")


def macro_args(s: str, name: str, nargs: int) -> list[tuple[int, int, list[str]]]:
    """All occurrences of \\name{..}{..}...: (start, end, [args])."""
    out = []
    for m in re.finditer(r"\\" + name + r"(?![A-Za-z])", s):
        if s[max(0, m.start() - 12):m.start()].endswith("newcommand{"):   # the definition, not a use
            continue
        pos, args = m.end(), []
        if pos < len(s) and s[pos] == "[":                       # optional argument
            pos = s.index("]", pos) + 1
        for _ in range(nargs):
            while s[pos] in " \n%":
                pos = s.index("\n", pos) + 1 if s[pos] == "%" else pos + 1
            a, pos = find_group(s, pos)
            args.append(a)
        out.append((m.start(), pos, args))
    return out


def strip_comments(s: str) -> str:
    return re.sub(r"(?m)(?<!\\)%.*$", "", s)


def cite_numbers(aux: Path) -> dict[str, int]:
    return {k: int(n) for k, n in re.findall(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", aux.read_text())}


def format_cites(nums: list[int]) -> str:
    """natbib numbers with sort&compress: runs of three or more become a range with an en dash."""
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts += [f"{nums[i]}–{nums[j]}"] if j - i >= 2 else [str(n) for n in nums[i:j + 1]]
        i = j + 1
    return "[" + ", ".join(parts) + "]"


def bbl_items(bbl: Path) -> list[str]:
    """Reference list entries in order, as LaTeX fragments pandoc can read."""
    txt = bbl.read_text()
    items = re.split(r"\\bibitem(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?\{[^}]+\}\s*", txt)[1:]
    out = []
    for it in items:
        it = it.split("\\end{thebibliography}")[0]
        it = re.sub(r"\\newblock\s*", " ", it)
        it = re.sub(r"\\doi\{([^}]*)\}", lambda m: "\\url{https://doi.org/" + m.group(1) + "}", it)
        it = re.sub(r"\\(?:href)\{([^}]*)\}\{[^}]*\}", r"\\url{\1}", it)
        it = re.sub(r"\s+", " ", it).strip()
        out.append(it)
    return out


def replace_cites(s: str, nums: dict[str, int]) -> str:
    def rep(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        return format_cites([nums[k] for k in keys])
    return re.sub(r"\\cite[pt]?\{([^}]+)\}", rep, s)


def cite_order(s: str) -> list[str]:
    """Citation keys in order of first appearance in the text."""
    seen = []
    for m in re.finditer(r"\\cite[pt]?\{([^}]+)\}", s):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in seen:
                seen.append(k)
    return seen


def plain(s: str) -> str:
    """LaTeX fragment to plain text (for counting words and comparing with rendered PDF text)."""
    s = strip_comments(s)
    s = re.sub(r"\\cite[pt]?\{[^}]*\}", "", s)
    s = re.sub(r"\\textsubscript\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textsuperscript\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(textbf|textit|emph|mbox)\{", "{", s)
    s = s.replace("\\&", "&").replace("\\%", "%").replace("~", " ").replace("--", "–")
    s = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def figure_blocks(s: str) -> list[dict]:
    """\\figurepage{file}{height}{legend} blocks: file, legend, label ('1' or 'S1'), title (bold lead-in)."""
    out = []
    for a, b, (fn, h, leg) in macro_args(s, "figurepage", 3):
        m = re.match(r"\s*\\textbf\{Fig\. (S?\d+) (.*?)\}", leg, re.S)
        out.append({"start": a, "end": b, "file": fn, "height": h, "legend": leg.strip(), "label": m.group(1),
                    "title": m.group(2).strip()})
    return out


def table_blocks(s: str) -> list[dict]:
    out = []
    for a, b, (title, tab, note) in macro_args(s, "tablepage", 3):
        m = re.match(r"Table (S?\d+) (.*)", title.strip(), re.S)
        out.append({"start": a, "end": b, "title_full": title.strip(), "label": m.group(1), "title": m.group(2),
                    "tabular": tab.strip(), "note": note.strip()})
    return out


def word_image_width(png: Path, max_w_in: float, max_h_in: float) -> float:
    w, h = Image.open(png).size
    return round(min(max_w_in, max_h_in * w / h), 2)


def to_pandoc_tex(src: str, nums: dict[str, int], refs: list[str], texdir: Path, fig_h: dict[str, float],
                  is_main: bool) -> str:
    """Pandoc-ready LaTeX copy of a manuscript source."""
    s = strip_comments(src)
    body_start = s.index("\\maketitle") + len("\\maketitle")
    head = s[:body_start]
    title = macro_args(head, "title", 1)[0][2][0]
    out = ["\\documentclass{article}", "\\begin{document}", "\\title{" + title + "}", "\\maketitle"]
    if is_main:
        esc = lambda t: t.replace("[", "{[}").replace("]", "{]}")
        names = []
        equal = False
        for m in re.finditer(r"\\author(\*?)\[([^\]]*)\]\{\\fnm\{([^}]*)\}\s*\\sur\{([^}]*)\}\}((?:(?!\\author|\\affil).)*)", head, re.S):
            eq = "\\equalcont" in m.group(5)
            equal |= eq
            names.append(esc(f"{m.group(3)} {m.group(4)}") + "\\textsuperscript{" + m.group(2) + ("*" if m.group(1) else "") + (",†" if eq else "") + "}")
        out.append(", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] + "\n")
        for m in re.finditer(r"\\affil\*?\[([^\]]*)\]\{(.*)\}\s*$", head, re.M):
            parts = re.findall(r"\\(?:orgdiv|orgname|city|postcode|country)\{([^}]*)\}", m.group(2))
            out.append("\\textsuperscript{" + m.group(1) + "}" + esc(", ".join(parts)) + "\n")
        mail = re.search(r"\\email\{([^}]*)\}", head)
        out.append("\\textsuperscript{*}Corresponding author: " + esc(mail.group(1) if mail else "") + "\n")
        if equal:
            out.append("\\textsuperscript{†}These authors contributed equally to this work.\n")
        abstract = macro_args(head, "abstract", 1)[0][2][0]
        kw = macro_args(head, "keywords", 1)[0][2][0]
        out.append("\\section*{Abstract}\n" + abstract + "\n")
        out.append("\\textbf{Keywords:} " + kw + "\n")
    else:
        out.append(macro_args(head, "abstract", 1)[0][2][0] + "\n")
    body = s[body_start:s.index("\\end{document}")]
    body = body.replace("\\backmatter", "")
    body = re.sub(r"\\bmhead\{([^}]*)\}", r"\\subsection*{\1}", body)
    body = re.sub(r"\\(section|subsection|subsubsection)\{", r"\\\1*{", body)
    reflist = "\\section*{References}\n\n" + "\n\n".join(f"{i}.\\ {r}" for i, r in enumerate(refs, 1)) + "\n"
    body = body.replace("\\bibliography{refs}", reflist)
    body = replace_cites(body, nums)
    # figures: image paragraph, then legend paragraph
    for fb in reversed(figure_blocks(body)):
        png = (texdir / fb["file"]).resolve()
        w = word_image_width(png, 6.3, fig_h.get(fb["label"], 6.2))
        block = f"\n\n\\includegraphics[width={w}in]{{{png}}}\n\n{fb['legend']}\n\n"
        body = body[:fb["start"]] + block + body[fb["end"]:]
    for tb in reversed(table_blocks(body)):
        block = f"\n\n\\textbf{{{tb['title_full']}}}\n\n{tb['tabular']}\n\n{tb['note']}\n\n"
        body = body[:tb["start"]] + block + body[tb["end"]:]
    body = re.sub(r"\\(clearpage|noindent|medskip|par|small|footnotesize|centering|raggedbottom)(?![A-Za-z])", "\n", body)
    out.append(body)
    out.append("\\end{document}\n")
    return "\n".join(out)
