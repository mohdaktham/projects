"""Extract text from docx files for scope understanding."""
import os
from pathlib import Path

try:
    from docx import Document
except ImportError:
    print("python-docx not installed")
    exit(1)

base = Path(__file__).resolve().parent
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith(".docx") and not f.startswith("~"):
            path = Path(root) / f
            try:
                doc = Document(path)
                text = "\n".join(p.text for p in doc.paragraphs)
                if text.strip():
                    rel = path.relative_to(base)
                    print("=== ", rel, " ===\n", text[:4000], "\n")
            except Exception as e:
                print("Error:", path, e)
