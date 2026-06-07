# modules/rag_processor.py
import os

def extract_document_context(file_path: str) -> str:
    """Extracts raw text metadata context elements from target attachments safely."""
    if not os.path.exists(file_path):
        return ""
        
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f"\n--- UPLOADED DOCUMENT TEXT ---\n{f.read()[:6000]}"
                
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                extracted_text = ""
                for page in reader.pages[:10]: # Cap text processing buffer window
                    extracted_text += page.extract_text() or ""
                return f"\n--- UPLOADED PDF DATA CONTENT ---\n{extracted_text[:6000]}"
            except ImportError:
                return "\n[SYSTEM NOTICE: pypdf dependency missing on host server runtime environment.]"
    except Exception as e:
        return f"\n[Document parsing runtime anomaly encountered: {e}]"
    return ""
