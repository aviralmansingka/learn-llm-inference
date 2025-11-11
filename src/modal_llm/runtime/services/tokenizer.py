from tokenizers import Tokenizer, Encoding
from loguru import logger

def get_tokenizer() -> Tokenizer:
    logger.info("Creating Tokenizer()")
    tokenizer: Tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return tokenizer
