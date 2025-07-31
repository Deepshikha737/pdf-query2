def chunk_text(text, max_tokens=300, max_chunks=10):
    sentences = text.split(". ")
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len((chunk + sentence).split()) <= max_tokens:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            if len(chunks) >= max_chunks:
                break  # Stop early to avoid memory overuse
            chunk = sentence + ". "
    
    if chunk and len(chunks) < max_chunks:
        chunks.append(chunk.strip())

    return chunks
