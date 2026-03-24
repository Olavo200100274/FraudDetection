"""Convert PDF articles in results_thesis/relatedWork/ to plain text files."""

import fitz  # PyMuPDF
from pathlib import Path

RELATED_WORK_DIR = Path(__file__).resolve().parent.parent / "results_thesis" / "relatedWork"


def pdf_to_text(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def convert_folder(folder: Path) -> None:
    """Convert all PDFs in a folder to .txt files (same name, same folder)."""
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"  No PDFs found in {folder.name}/")
        return

    for pdf_path in pdfs:
        txt_path = pdf_path.with_suffix(".txt")
        print(f"  {pdf_path.name} -> {txt_path.name}")
        text = pdf_to_text(pdf_path)
        txt_path.write_text(text, encoding="utf-8")
    print(f"  {len(pdfs)} file(s) converted in {folder.name}/")


def main():
    print("Converting related work PDFs to text...\n")
    for subfolder in sorted(RELATED_WORK_DIR.iterdir()):
        if subfolder.is_dir():
            print(f"[{subfolder.name}]")
            convert_folder(subfolder)
            print()
    print("Done.")


if __name__ == "__main__":
    main()
