from fastapi import Depends, FastAPI

from modal_llm.runtime.models.engine import EngineRequest, EngineResponse
from modal_llm.runtime.services.engine import Engine, get_engine

from modal_llm.web.models import ChatCompletionRequest, ChatCompletionResponse, convert_from_engine, convert_to_engine

from loguru import logger

# Create FastAPI instance

# TODO: Add streaming responses

asgi_app = FastAPI()

@asgi_app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    engine: Engine = Depends(get_engine),  # pyright: ignore[reportCallInDefaultInitializer]
) -> ChatCompletionResponse:
    # Convert to internal format
    engine_request: EngineRequest = convert_to_engine(request)

    # TODO: Replace with actual LLM inference

    engine_response: EngineResponse = await engine.process(engine_request)
    logger.info(engine_response)


    return convert_from_engine(engine_response)
