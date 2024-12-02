import base64
import io

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_encoded_pages(pdf_path: str, image_format: str = "png") -> list[str]:
    images = []
    doc = fitz.open(pdf_path)

    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert to desired format
        img_buffer = io.BytesIO()
        img.save(img_buffer, format=image_format.upper())
        img_buffer.seek(0)

        # Encode to base64
        base64_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        images.append(base64_image)

    doc.close()
    return images
