"""Extract text from PDFs into sibling .txt files."""

import sys
from pathlib import Path

import pymupdf


def main():
    for arg in sys.argv[1:]:
        pdf = Path(arg)
        doc = pymupdf.open(pdf)
        parts = []
        for i, page in enumerate(doc, 1):
            parts.append(f"\n\n===== PAGE {i} =====\n\n")
            parts.append(page.get_text("text"))
        out = pdf.with_suffix(".txt")
        out.write_text("".join(parts), encoding="utf-8")
        print(f"{pdf.name}: {len(doc)} pages -> {out.name} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
