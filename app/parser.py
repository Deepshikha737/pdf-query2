from pypdf import PdfReader
import io

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 20):
    reader = PdfReader(io.BytesIO(file_bytes))
    text_chunks = []

    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break  # Stop early to limit memory use
        text = page.extract_text()
        if text:
            text_chunks.append(text)
    
    return "\n".join(text_chunks)
