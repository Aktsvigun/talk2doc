import asyncio
import os
import tempfile

import streamlit as st
from openai import OpenAI

from talk2doc.doc2text.aextract_text_from_image_or_pdf import (
    aextract_text_from_image_or_pdf,
)
from talk2doc.chat.get_llm_response import get_llm_response
from talk2doc.utils.prompts import IMAGE2TEXT_USER_PROMPT_EXTRACT_DATA, CHAT_SYSTEM_PROMPT
from talk2doc.utils.constants import (
    DEFAULT_IMAGE2TEXT_MODEL,
    DEFAULT_CHAT_MODEL,
)

STREAM_CHAT_OUTPUT = os.environ.get("STREAM_CHAT_OUTPUT", "True").lower() == "true"
DEFAULT_BASE_URL = os.getenv(
    "DEFAULT_BASE_URL", "https://api.studio.nebius.ai/v1/"
)
IMAGE2TEXT_MODEL = os.environ.get("IMAGE2TEXT_MODEL", DEFAULT_IMAGE2TEXT_MODEL)
CHAT_MODEL = os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "assistant" not in st.session_state:
    st.session_state.assistant = None

st.set_page_config(page_title="Talk2YourDoc", layout="wide")


def main():
    col1, col2 = st.columns(2)
    with col1:
        client = OpenAI(base_url=DEFAULT_BASE_URL)
        st.title("PDF Assistant")
        uploaded_file = st.file_uploader(
            "Choose a PDF or image file", type=["pdf", "png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            input_type = uploaded_file.name.split(".")[-1]
            progress_bar = st.progress(
                0, text="Extracting information from the provided file..."
            )
            pdf_extracted_content = asyncio.run(
                aextract_text_from_image_or_pdf(
                    client=client,
                    image_or_pdf_path=tmp_file_path,
                    input_type=input_type,
                    prompt=IMAGE2TEXT_USER_PROMPT_EXTRACT_DATA,
                    model_checkpoint=IMAGE2TEXT_MODEL,
                )
            ).strip()
            progress_bar.empty()
            pdf_content = st.text_area(
                "Extracted Text (you can modify if necessary)",
                pdf_extracted_content,
                height=400,
            )
            # Delete the temporary file
            os.unlink(tmp_file_path)
            if st.button("Create Assistant!"):
                st.session_state.assistant = str(CHAT_SYSTEM_PROMPT).format(
                    pdf_content=pdf_content
                )
                # Remove all previous messages
                st.session_state.messages = []

    with col2:
        # Chat interface
        if st.session_state.assistant:
            client_chat = OpenAI(base_url=DEFAULT_BASE_URL)
            if user_input := st.chat_input("Type your message..."):
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                with st.chat_message("assistant"):
                    output = get_llm_response(
                        client=client_chat,
                        messages=st.session_state.messages,
                        system_prompt=st.session_state.assistant,
                        model_checkpoint=CHAT_MODEL,
                        do_stream=STREAM_CHAT_OUTPUT,
                    )
                    if STREAM_CHAT_OUTPUT:
                        response = st.write_stream(output)
                    else:
                        response = output.choices[0].message.content.strip()
                        st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    main()
