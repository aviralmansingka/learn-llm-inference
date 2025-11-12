from functools import cache
from tokenizers import Tokenizer
from loguru import logger

@cache
def get_tokenizer() -> Tokenizer:
    logger.info("Creating Tokenizer()")
    return Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
