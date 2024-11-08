import asyncio
import base64
from typing import Literal
import warnings

from openai import Client

from .classify_is_pdf_searchable import classify_is_pdf_searchable
from .extract_text_from_searchable_pdf import extract_text_from_searchable_pdf
from .pdf_to_encoded_pages import pdf_to_encoded_pages
from ..utils.prompts import IMAGE2TEXT_USER_PROMPT_EXTRACT_DATA
from ..utils.validate_llm_kwargs import validate_llm_kwargs


# Function to encode the image to base64
def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to call Vision LM with the base64 image and a question
async def _acall_model(
    client: Client,
    base64_image: str,
    input_type: Literal["png", "jpg", "jpeg", "gif", "webp"],
    prompt: str,
    model_checkpoint: str = "Qwen/Qwen2-VL-72B-Instruct-AWQ",
    **llm_kwargs,
):
    llm_kwargs = validate_llm_kwargs(llm_kwargs)
    response = client.chat.completions.create(
        model=model_checkpoint,  # Use GPT-4o model identifier
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{input_type};base64,{base64_image}",
                            "detail": "high",  # Adjust to "low" for a less detailed analysis
                        },
                    },
                ],
            }
        ],
        **llm_kwargs,
    )
    # Return the response from the model
    return response.choices[0].message.content.strip()


async def _process_page(
    client, base64_page, input_type, prompt, model_checkpoint, page_num
):
    result = f"\n\n=== Page {page_num} ===\n\n"
    result += await _acall_model(
        client, base64_page, input_type, prompt, model_checkpoint
    )
    return result, page_num


async def _process_all_pages(
    client, base64_pages, input_type, prompt, model_checkpoint
):
    tasks = [
        _process_page(client, page, input_type, prompt, model_checkpoint, i)
        for i, page in enumerate(base64_pages, 1)
    ]

    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[1])
    return "".join(result[0] for result in sorted_results)


async def aextract_text_from_image_or_pdf(
    client: Client,
    image_or_pdf_path: str,
    input_type: Literal["pdf", "png", "jpg", "jpeg", "gif", "webp"] | None = None,
    prompt: str | None = None,
    model_checkpoint: str = "Qwen/Qwen2-VL-72B-Instruct",
) -> str | None:
    if input_type is None:
        input_type = image_or_pdf_path.split(".")[-1].lower()
    if prompt is None:
        prompt = str(IMAGE2TEXT_USER_PROMPT_EXTRACT_DATA)
    if input_type in "pdf":
        input_type = "jpeg"  # OpenAI cannot process the input with the type 'pdf'
        pdf_is_searchable = classify_is_pdf_searchable(image_or_pdf_path)
        if pdf_is_searchable:
            return extract_text_from_searchable_pdf(image_or_pdf_path)
        base64_pages = pdf_to_encoded_pages(image_or_pdf_path)
        if len(base64_pages) > 1:
            return await _process_all_pages(
                client=client,
                base64_pages=base64_pages,
                input_type=input_type,
                prompt=prompt,
                model_checkpoint=model_checkpoint,
            )
        elif len(base64_pages) == 1:
            return await _acall_model(
                client, base64_pages[0], input_type, prompt, model_checkpoint
            )
        else:
            return
    # Otherwise the input is an image
    if not input_type in (
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
    ):
        warnings.warn(
            f"Input should have one of the following types: pdf, png, jpg, jpeg, gif, webp; "
            + "received the type {input_type}. "
            + "Treating the image as `png`"
        )
        input_type = "png"
    base64_image = _encode_image_to_base64(image_or_pdf_path)
    return await _acall_model(
        client, base64_image, input_type, prompt, model_checkpoint
    )
