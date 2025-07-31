def chunk_text(text, max_tokens=300):
    sentences = text.split(". ")
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len((chunk + sentence).split()) <= max_tokens:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())

    return chunks
