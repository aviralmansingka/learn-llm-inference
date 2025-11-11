import modal

app = modal.App("modal-llm")

image = modal.Image.debian_slim().uv_sync().add_local_python_source("modal_llm")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

with image.imports():
    import torch.nn as nn

@app.function(
    gpu="a100",
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)  # pyright: ignore[reportUnknownMemberType]
@modal.asgi_app()  # pyright: ignore[reportUnknownMemberType]
def modal_llm():
    from modal_llm.web.fastapi import asgi_app

    return asgi_app
