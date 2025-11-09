from loguru import logger
import torch
from transformers import AutoModelForCausalLM

class Model:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    async def generate(self, token_ids: list[int]) -> list[int]:
        input_tensor = torch.tensor(token_ids, dtype=torch.int32).unsqueeze(0)
        logger.info(f"{input_tensor.shape=}")
        with torch.no_grad():
            output = self.model.generate(inputs=input_tensor)
        logger.info(f"{output.shape=}")

        return output.squeeze(0).tolist()

def get_model():
    return Model()
