import modal

app = modal.App("scratch-llm")

image = modal.Image.debian_slim().uv_sync().add_local_python_source("scratch_llm")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

@app.function(
    gpu="a100",
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
@modal.asgi_app()
def asgi_app():
    from scratch_llm.web.fastapi import asgi_app

    return asgi_app
