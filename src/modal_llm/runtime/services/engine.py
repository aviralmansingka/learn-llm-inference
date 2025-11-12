from fastapi import Depends
from tokenizers import Encoding, Tokenizer

from loguru import logger

from modal_llm.runtime.models.engine import EngineRequest, EngineResponse, UsageStats
from modal_llm.runtime.models.model import Model, get_model
from modal_llm.runtime.services.tokenizer import get_tokenizer


class Engine:
    def __init__(self, tokenizer: Tokenizer, model: Model):
        self.tokenizer = tokenizer
        self.model= model

    @staticmethod
    def format_chat(messages: list[dict]) -> str:
        # TODO append to list and use join for better performance
        formatted = ""
        for message in messages:
            role = message.role
            content = message.content
            formatted += f"{role}: {content}\n"
        return formatted.strip()

    async def process(self, req: EngineRequest) -> EngineResponse: 
        response_content = Engine.format_chat(req.messages)
        encoding: Encoding = self.tokenizer.encode(response_content)

        # FIX FOR 
        # The attention mask is not set and cannot be inferred from input because pad token 
        # is same as eos token. As a consequence, you may observe unexpected behavior. 
        # Please pass your input's `attention_mask` to obtain reliable results
        attention_mask = [1 if token_id != self.tokenizer.token_to_id("<pad>") else 0 for token_id in encoding.ids]

        output_tokens = await self.model.generate(encoding.ids, attention_mask=attention_mask)
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
    logger.info("Creating Engine(tokenizer, model)")
    return Engine(tokenizer=tokenizer, model=model)

