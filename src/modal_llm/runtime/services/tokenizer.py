from tokenizers import Tokenizer, Encoding

def get_tokenizer() -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    return tokenizer
