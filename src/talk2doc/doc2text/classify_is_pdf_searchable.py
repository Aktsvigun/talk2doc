import fitz  # NB! Install it with `pip install PyMuPDF`, not with `pip install fitz`


def classify_is_pdf_searchable(file_path):
    try:
        doc = fitz.open(file_path)
        for page in doc:
            images = page.get_images()
            if len(images) > 0:
                return False
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False
