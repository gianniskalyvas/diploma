from ._abstract_model import AbstractModel

class FalconModel(AbstractModel):
    _name = 'Falcon'
    _default_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "max_new_tokens": 1024,
        "stop": ["User:", "Falcon:", '</s>'],
        "seed": 0
    }

    def _render_prompt(self, history):

        for message_pair in history:
            msg = message_pair["user"]
            msg_list = [line for line in msg.split('\n') if line.strip() != '']

            assistant_msg = message_pair.get("assistant")

        prompt =  f"<|system|>\n{msg_list[0]}\n<|user|>\n{msg_list[-1]}\n<|assistant|>\n"
           
        return prompt
