CHAT_SYSTEM_PROMPT = """
Act as helpful assistant. \
The user provided a pdf, its content will be given to you below. \
Based on its content, answer the user's questions laconically and to the point.

PDF content:
{pdf_content}
""".strip()


IMAGE2TEXT_USER_PROMPT_EXTRACT_DATA = """
Act as an experienced data extractor. \
Please read all the data & text from the image below and write it down.

For the text, first detect its language; without mentioning it, write the text down as it is. \
Try to preserve the formatting of the text (bold text etc.). \
For other data sources, transform them into a text if they can be transformed. \
For example, if you see an image of a salmon, write: "*An image of a salmon*". \
Otherwise, just ignore the data source.
""".strip()
