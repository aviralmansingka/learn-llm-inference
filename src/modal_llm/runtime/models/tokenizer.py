from typing import _promote
from tokenizers import Tokenizer, Encoding

tokenizer: Tokenizer = Tokenizer.from_pretrained("openai/gpt-oss-20b")

def tokenize(prompt: str):
    encoding: Encoding = tokenizer.encode(prompt)
    return encoding
