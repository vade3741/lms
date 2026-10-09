def split_text(text, chunk_size=700, overlap=120):
    if not text:
        return []

    text = " ".join(str(text).split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind("; ", start, end),
                text.rfind(" ", start, end)
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks