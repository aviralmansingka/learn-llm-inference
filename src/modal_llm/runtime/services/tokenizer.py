from tokenizers import Tokenizer, Encoding

def get_tokenizer() -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_pretrained("gpt2")

    return tokenizer
