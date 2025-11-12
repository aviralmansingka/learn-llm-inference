from loguru import logger
import torch
from transformers import AutoModelForCausalLM, 

class Model:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def generate(self, token_ids: list[int], max_tokens: int = 1028, attention_mask: list[int] = None) -> list[int]:
        input_tensor = torch.tensor(token_ids, dtype=torch.int32).unsqueeze(0).to(self.device)
        
        # Create attention_mask tensor if provided
        if attention_mask is not None:
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.int32).unsqueeze(0).to(self.device)
        else:
            # Default to attending to all tokens
            attention_mask_tensor = torch.ones_like(input_tensor, dtype=torch.int32).to(self.device)

        output = self.model.generate(input_ids=input_tensor, max_new_tokens=max_tokens, attention_mask=attention_mask_tensor)
        return output.squeeze(0).tolist()

def get_model():
    logger.info("Creating Model()")
    return Model()
