"""Extract text from PDFs in a folder. Saves each as .txt next to the PDF.

Usage: python pdf_extract.py <folder> [--max-pages N] [--out-suffix .txt]
"""
import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF


def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> str:
    doc = fitz.open(str(pdf_path))
    chunks = []
    n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
    for i in range(n_pages):
        page = doc[i]
        text = page.get_text("text")
        chunks.append(f"--- PAGE {i + 1} ---\n{text}")
    doc.close()
    return f"[Total pages: {len(doc) if max_pages is None else f'{n_pages}/{len(fitz.open(str(pdf_path)))}'}]\n\n" + "\n\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Max pages per PDF (default: all)")
    parser.add_argument("--out-suffix", type=str, default=".extracted.txt")
    args = parser.parse_args()

    folder = args.folder
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(folder.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {folder}")

    for pdf in pdfs:
        out_path = pdf.with_suffix(args.out_suffix)
        try:
            text = extract_pdf_text(pdf, max_pages=args.max_pages)
            out_path.write_text(text, encoding="utf-8")
            print(f"  OK: {pdf.name} -> {len(text):,} chars")
        except Exception as e:
            print(f"  FAIL: {pdf.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
