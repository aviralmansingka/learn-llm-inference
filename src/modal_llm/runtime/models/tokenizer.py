from typing import Optional
from tokenizers import Tokenizer, Encoding

tokenizer: Tokenizer = Tokenizer.from_pretrained("openai/gpt-oss-20b")

def tokenize(prompt: str, max_length: Optional[int] = None):
    encoding: Encoding = tokenizer.encode(prompt)
    if max_length:
        tokenizer.enable_truncation(max_length=max_length)
    return encoding
