
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class Llama3Model(AbstractModel):
    _name = 'Llama3'
    _default_config = {
        "do_sample":False,
        "max_new_tokens": 1024,
        "seed": 0,
        "repetition_penalty":1.0
    }

    def _render_prompt(self, history):
        prompt = "<|begin_of_text|>"
    
        for message_pair in history:
            msg = message_pair["user"]
            msg_list = [line for line in msg.split('\n') if line.strip() != '']

            assistant_msg = message_pair.get("assistant")

            prompt += (
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{msg_list[0]}\n<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{msg_list[-1]}\n<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )

            if assistant_msg:
                prompt += (
                    f"{assistant_msg}"
                )
    
        return prompt
