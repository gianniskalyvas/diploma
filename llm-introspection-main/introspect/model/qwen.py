
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class Qwen(AbstractModel):
    _name = 'Qwen'
    _default_config = {
        "do_sample":False,
        "max_new_tokens": 1024,
        "seed": 0,
        "repetition_penalty":1.0
    }

    def _render_prompt(self, history):
        prompt = ""
        for message_pair in history:
            msg = message_pair["user"]
            msg_list = [line for line in msg.split('\n') if line.strip() != '']

            prompt += (
                "<|im_start|>system "
                f"{msg_list[0]}\n"
                "<|im_end|> <|im_start|>user "
                f"{msg_list[-1]}\n"
                "<|im_end|><|im_start|>assistant"
            )


        return prompt

