import torch
import torch.nn as nn

from common import LogitsAdapter, ElectraWrapper

from fnet_improved.modeling_fnet import FNetForMaskedLM, FNetForPreTraining
import torch.nn as nn
from typing import Tuple, OrderedDict, Optional, Self
import os

from typing import Callable

class ElectraBackbonedWithFNet(ElectraWrapper[FNetForMaskedLM, FNetForPreTraining]):
    @classmethod
    def _tie_embeddings(cls, gen: FNetForMaskedLM, disc: FNetForPreTraining) -> None:
        (
            gen.fnet.embeddings.word_embeddings,
            gen.fnet.embeddings.position_embeddings,
            gen.fnet.embeddings.token_type_embeddings
        ) = (
            disc.fnet.embeddings.word_embeddings,
            disc.fnet.embeddings.position_embeddings,
            disc.fnet.embeddings.token_type_embeddings
        )

    def __init__(
        self,
        model_generator,
        model_discriminator,
        vocab: OrderedDict[str, int],
        mask_prob: float,
        random_token_prob: float=0.,
        wrap_to_logits_adapter: bool = False,
        distributed_enabled: bool = False,
        mask_token: str = '[MASK]',
        pad_token: str = '[PAD]',
        class_token: str = '[CLS]',
        separator_token: str = '[SEP]',
        **kwargs
    ):
        super().__init__(
            model_generator,
            model_discriminator,
            vocab,
            mask_prob,
            random_token_prob,
            wrap_to_logits_adapter,
            distributed_enabled,
            mask_token=mask_token,
            pad_token=pad_token,
            class_token=class_token,
            separator_token=separator_token,
            fix=None,
            **kwargs
        )
    
    def save_pretrained(self, output_dir: str) -> None:
        self.discriminator.fnet.save_pretrained(output_dir)
