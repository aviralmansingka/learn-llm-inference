from tokenizers import Tokenizer, Encoding
from loguru import logger

from modal_llm.runtime.models import tokenizer

def get_tokenizer() -> Tokenizer:
    logger.info("Creating Tokenizer()")
    tokenizer: Tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    if tokenizer.token_to_id("<pad>") is None:
        tokenizer.add_special_tokens(["<pad>"])
    
    # Ensure the pad token is set
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")


    return tokenizer
