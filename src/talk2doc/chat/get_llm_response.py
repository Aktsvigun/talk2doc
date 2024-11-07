from openai import OpenAI

from ..utils.validate_llm_kwargs import validate_llm_kwargs


def get_llm_response(
    client: OpenAI,
    messages: list[dict[str, str]],
    system_prompt: str,
    model_checkpoint: str,
    do_stream: bool = True,
    **llm_kwargs
):
    llm_kwargs = validate_llm_kwargs(llm_kwargs)
    llm_kwargs["stream"] = do_stream
    output = client.chat.completions.create(
        model=model_checkpoint,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        **llm_kwargs,
    )
    return output
