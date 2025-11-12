from functools import cache
from fastapi import Depends
from tokenizers import Encoding, Tokenizer

from loguru import logger

from scratch_llm.runtime.models.engine import EngineRequest, EngineResponse, Message, UsageStats
from scratch_llm.runtime.services.model import Model, get_model
from scratch_llm.runtime.services.tokenizer import TokenizerManager, get_tokenizer_manager


class Engine:
    def __init__(self, tokenizer: TokenizerManager, model: Model):
        self.tokenizer: TokenizerManager = tokenizer
        self.model: Model= model

    @staticmethod
    def format_chat(messages: list[Message]) -> str:
        # TODO append to list and use join for better performance
        formatted = ""
        for message in messages:
            role = message.role
            content = message.content
            formatted += f"{role}: {content}\n"
        return formatted.strip()

    async def process(self, req: EngineRequest) -> EngineResponse: 
        response_content = Engine.format_chat(req.messages)
        encoding, attention_mask = self.tokenizer.tokenize(response_content)

        output_tokens = await self.model.generate(encoding.ids, attention_mask=attention_mask)
        final_content: str = self.tokenizer.detokenize(output_tokens)

        usage_stats = self._calculate_usage_stats(input_tokens=encoding.ids, output_tokens=output_tokens)
        engine_response = EngineResponse(req.model, prompt_response=final_content, usage_stats=usage_stats)
        return engine_response

    def _calculate_usage_stats(self, input_tokens: list[int], output_tokens: list[int]) -> UsageStats:
        prompt_tokens = len(input_tokens)
        completion_tokens = len(output_tokens)
        total_tokens = prompt_tokens + completion_tokens
        usage_stats = UsageStats(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
        return usage_stats

@cache
def get_engine(
        tokenizer: TokenizerManager = Depends(get_tokenizer_manager), # pyright: ignore[reportCallInDefaultInitializer]
        model: Model = Depends(get_model), # pyright: ignore[reportCallInDefaultInitializer]
) -> Engine:
    logger.info("Creating Engine(tokenizer, model)")
    return Engine(tokenizer=tokenizer, model=model)
