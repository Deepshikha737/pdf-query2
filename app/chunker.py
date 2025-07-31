def chunk_text(text, max_tokens=300, max_chunks=10):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Better sentence splitting

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current_len + len(words) <= max_tokens:
            current_chunk.extend(words)
            current_len += len(words)
        else:
            chunk = " ".join(current_chunk).strip()
            if chunk:
                chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break
            current_chunk = words
            current_len = len(words)

    # Add the last chunk
    if current_chunk and len(chunks) < max_chunks:
        chunk = " ".join(current_chunk).strip()
        if chunk:
            chunks.append(chunk)

    return chunks
