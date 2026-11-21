from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DESTINATION = BASE / "article" / "R1_Source" / "figures"


def main() -> None:
    inkscape = shutil.which(os.environ.get("INKSCAPE", "inkscape")) or shutil.which("inkscape.com")
    if inkscape is None:
        raise FileNotFoundError("Add Inkscape to PATH or set INKSCAPE to its executable")
    drawio = DESTINATION / "figure_01.drawio"
    svg = DESTINATION / "figure_01.svg"
    pdf = DESTINATION / "figure_01.pdf"
    pdf_tex = DESTINATION / "figure_01.pdf_tex"
    if not drawio.is_file() or not svg.is_file():
        raise FileNotFoundError("The editable inspection-workflow source is incomplete")
    subprocess.run([
        inkscape, str(svg), "--export-filename", str(pdf),
        "--export-area-page",
    ], check=True)
    pdf_tex.write_text(
        "% Inkscape PDF export with a selectable-text vector drawing.\n"
        "\\begingroup\n"
        "\\ifx\\svgwidth\\undefined\\def\\svgwidth{\\linewidth}\\fi\n"
        "\\includegraphics[width=\\svgwidth]{figures/figure_01.pdf}\n"
        "\\endgroup\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Editable diagram: {drawio}")
    print(f"Editable vector: {svg}")
    print(f"Selectable-text PDF: {pdf}")
    print(f"TeX figure wrapper: {pdf_tex}")


if __name__ == "__main__":
    main()
