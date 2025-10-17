
from ._abstract_model import AbstractModel
from ..types import ChatHistory

class Gemma(AbstractModel):
    _name = 'Gemma'
    _default_config = {
        "do_sample":False,
        "max_new_tokens": 1024,
        "seed": 0
    }

    def _render_prompt(self, history):
        prompt = "<bos>"
    
        for message_pair in history:
            msg = message_pair["user"]
            msg_list = [line for line in msg.split('\n') if line.strip() != '']

            prompt += (
                "<start_of_turn>user\n"
                f"{msg_list[0]}\n"
                f"{msg_list[-1]}\n<end_of_turn>"
                "<start_of_turn>model\n"
            )


        return prompt

