#!/usr/bin/env python3
"""paper.md -> HTML(嵌入MathJax SVG公式) -> WeasyPrint PDF。"""
import markdown, pathlib, json

src = pathlib.Path("docs/paper.md").read_text(encoding="utf-8")
formulas = json.loads(pathlib.Path("formulas.json").read_text(encoding="utf-8"))

# 先替换 display ($$)，再替换 inline ($)，用 SVG 取代 LaTeX 源码
for key in sorted(formulas, key=len, reverse=True):
    svg = formulas[key]
    if key.startswith("$$"):
        repl = f'<div style="text-align:center;margin:12px 0;">{svg}</div>'
    else:
        repl = f'<span style="vertical-align:middle;">{svg}</span>'
    src = src.replace(key, repl)

body = markdown.markdown(src, extensions=["tables", "fenced_code", "toc", "sane_lists"])

html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 18mm 15mm; }}
body {{ font-family:"Noto Sans CJK SC",sans-serif; line-height:1.7; color:#222; font-size:14px; }}
h1 {{ font-size:24px; border-bottom:3px solid #2c3e50; padding-bottom:10px; }}
h2 {{ font-size:19px; border-bottom:1px solid #ccc; padding-bottom:6px; margin-top:26px; page-break-after:avoid; }}
h3,h4 {{ color:#34495e; page-break-after:avoid; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:12.5px; }}
th,td {{ border:1px solid #bbb; padding:5px 8px; text-align:left; }}
th {{ background:#f0f3f6; }}
pre {{ background:#f6f8fa; padding:12px; border-radius:6px; font-size:12px; line-height:1.4; white-space:pre-wrap; }}
blockquote {{ border-left:4px solid #2c3e50; margin:0; padding-left:14px; color:#555; }}
svg {{ vertical-align:middle; }}
</style></head><body>{body}</body></html>"""

pathlib.Path("docs/paper.html").write_text(html, encoding="utf-8")
from weasyprint import HTML
HTML(string=html, base_url=".").write_pdf("jump_analysis_paper.pdf")
print("PDF generated")
