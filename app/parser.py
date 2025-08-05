from pypdf import PdfReader
import io

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 20) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_chunks = []

        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break  # 🔹 Limit number of pages to reduce memory
            text = page.extract_text()
            if text:
                text_chunks.append(text)

        return "\n".join(text_chunks)

    except Exception as e:
        raise ValueError(f"❌ Failed to extract PDF text: {e}")
