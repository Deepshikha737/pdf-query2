def chunk_text(text, max_tokens=300, max_chunks=10):
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) <= max_tokens:
            current_chunk.extend(words)
            current_chunk.append("")  # for the period
            current_len += len(words)
        else:
            chunks.append(" ".join(current_chunk).strip() + ".")
            if len(chunks) >= max_chunks:
                break
            current_chunk = words + [""]
            current_len = len(words)
    
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(" ".join(current_chunk).strip() + ".")

    return chunks
