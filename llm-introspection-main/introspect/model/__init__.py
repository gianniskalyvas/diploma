
__all__ = ['FalconModel', 'Llama2Model',
           'models', 'AbstractModel']

from typing import Type, Mapping

from ._abstract_model import AbstractModel
from .falcon import FalconModel
from .llama2 import Llama2Model
from .llama3 import Llama3Model
from .mistral import MistralModel
from .gemma import Gemma
from .qwen import Qwen

models: Mapping[str, Type[AbstractModel]] = {
    Model._name: Model
    for Model
    in [FalconModel, Llama2Model, Llama3Model, MistralModel, Gemma, Qwen]
}
