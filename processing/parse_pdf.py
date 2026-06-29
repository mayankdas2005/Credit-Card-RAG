import pymupdf4llm
import os
from pathlib import Path
import re


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_DIR = os.path.join(ROOT_DIR, "data","pdfs", "MITC-Premium Card Consolidated_27.05.2026 - MITC_Premium.pdf")
MD_DIR = os.path.join(ROOT_DIR, "data", "parsed_md")


#md = pymupdf4llm.to_markdown(DOC_DIR)
#Path(MD_DIR).write_bytes(md.encode())

def extract_markdown_from_pdf(pdf_path: str, md_path: str):
    md = pymupdf4llm.to_markdown(pdf_path)

    cleaned_md = re.sub(r'`(\d+)', r'₹\1', md)
    
    os.makedirs(md_path, exist_ok=True)
    base_file_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    save_path = os.path.join(md_path, base_file_name)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(cleaned_md)
    
    return cleaned_md
    






