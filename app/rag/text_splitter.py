def split_text(text, chunk_size=700, overlap=120):
    if not text:
        return []

    text = text.replace("\n", " ").strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks