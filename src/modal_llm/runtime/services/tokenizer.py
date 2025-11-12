from functools import cache
from typing import Optional
from tokenizers import Encoding, Tokenizer
from loguru import logger
from modal_llm.config import Settings, get_settings
from fastapi import Depends

class TokenizerManager:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str, max_length: Optional[int] = None) -> Encoding:
        encoding: Encoding = self.tokenizer.encode(prompt)
        if max_length:
            self.tokenizer.enable_truncation(max_length=max_length)
        return encoding

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

@cache
def get_tokenizer_manager(settings: Settings = Depends(get_settings)) -> TokenizerManager:
    logger.info("Creating Tokenizer()")
    tokenizer: Tokenizer = Tokenizer.from_pretrained(settings.model_name)

    if tokenizer.token_to_id("<pad>") is None:
        tokenizer.add_special_tokens(["<pad>"])

    # Ensure the pad token is set
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

    return TokenizerManager(tokenizer)
