from loguru import logger
import torch
from transformers import AutoModelForCausalLM

class Model:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    async def generate(self, token_ids: list[int]) -> list[int]:
        return token_ids

def get_model():
    return Model()
