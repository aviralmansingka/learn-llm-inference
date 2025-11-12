from functools import cache
from loguru import logger
import torch
from transformers import AutoModelForCausalLM

class Model:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        self.device: str = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def generate(self, token_ids: list[int], max_tokens: int = 64) -> list[int]:
        input_tensor = torch.tensor(token_ids, dtype=torch.int32).unsqueeze(0).to(self.device)
        output = self.model.generate(inputs=input_tensor, max_new_tokens=max_tokens)
        return output.squeeze(0).tolist()

@cache
def get_model() -> Model:
    logger.info("Creating Model()")
    return Model()
