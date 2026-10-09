import pymupdf


def extract_pdf_pages(pdf_path):
    try:
        with pymupdf.open(pdf_path) as document:
            return [page.get_text().strip() for page in document]
    except Exception as error:
        raise RuntimeError(f"Could not read PDF '{pdf_path}': {error}") from error


def extract_text_from_pdf(uploaded_file):
    text = ""

    try:
        pdf_bytes = uploaded_file.read()
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        for page in document:
            text += page.get_text()

        document.close()
        return text.strip()

    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def extract_text_from_txt(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("latin-1").strip()
    except Exception as e:
        return f"Error reading TXT file: {str(e)}"


def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    return "Unsupported file type. Please upload PDF or TXT file."