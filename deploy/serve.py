import modal

app = modal.App("modal-llm")

image = modal.Image.debian_slim().uv_sync().add_local_python_source("modal_llm")

with image.imports():
    import torch.nn as nn

@app.function(image=image)  # pyright: ignore[reportUnknownMemberType]
@modal.asgi_app()  # pyright: ignore[reportUnknownMemberType]
def modal_llm():
    from modal_llm.web.fastapi import asgi_app

    return asgi_app
