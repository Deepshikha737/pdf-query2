def chunk_text(text, max_tokens=250, max_chunks=10):
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue  # Skip empty sentences

        if current_len + len(words) <= max_tokens:
            current_chunk.extend(words)
            current_len += len(words)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip() + ".")
                if len(chunks) >= max_chunks:
                    break
            current_chunk = words
            current_len = len(words)

    # Add any leftover chunk
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(" ".join(current_chunk).strip() + ".")

    return chunks
