from collections.abc import AsyncGenerator
from fastapi import Depends
from tokenizers import Encoding, Tokenizer

from modal_llm.runtime.models.engine import EngineRequest, EngineResponse, UsageStats
from modal_llm.runtime.models.model import Model, get_model
from modal_llm.runtime.services.tokenizer import get_tokenizer

from loguru import logger



class Engine:
    def __init__(self, tokenizer: Tokenizer, model: Model):
        self.tokenizer = tokenizer
        self.model= model

    async def process(self, req: EngineRequest) -> EngineResponse: 
        response_content = req.messages[-1].content

        encoding: Encoding = self.tokenizer.encode(response_content)
        logger.info(f"{type(encoding.ids[0])=}")

        output_tokens = await self.model.generate(encoding.ids)

        final_content: str = self.tokenizer.decode(output_tokens)

        usage_stats = self._calculate_usage_stats(input_tokens=encoding.ids, output_tokens=output_tokens)
        engine_response = EngineResponse(req.model, prompt_response=final_content, usage_stats=usage_stats)
        return engine_response

    def _calculate_usage_stats(self, input_tokens: list[int], output_tokens: list[int]) -> UsageStats:
        prompt_tokens = len(input_tokens)
        completion_tokens = len(output_tokens)
        total_tokens = prompt_tokens + completion_tokens
        usage_stats = UsageStats(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
        return usage_stats


def get_engine(tokenizer: Tokenizer = Depends(get_tokenizer), model: Model = Depends(get_model)) -> Engine:
    return Engine(tokenizer=tokenizer, model=model)

